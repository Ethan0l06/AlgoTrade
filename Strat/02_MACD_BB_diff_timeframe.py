import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import optuna
import talib
from datetime import datetime
import json
from pathlib import Path
import traceback

# --- All Project Imports ---
from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Config.Enums import TradingMode
from AlgoTrade.Strat.Class.BaseStrategy import BaseStrategy
from AlgoTrade.Factories.IndicatorFactory import IndicatorFactory
from AlgoTrade.Sizing.AtrBandsSizer import AtrBandsSizer
from AlgoTrade.Sizing.PercentBalanceSizer import PercentBalanceSizer
from AlgoTrade.Sizing.FixedAmountSizer import FixedAmountSizer
from AlgoTrade.Optimizer.StudyRunner import run_optimization
from AlgoTrade.Analysis.BacktestAnalysis import BacktestAnalysis
from AlgoTrade.Config.Paths import OPTIMIZER_DIR, ANALYSIS_DIR


class MacdBBStrategy(BaseStrategy):
    """
    Enhanced MACD+BB strategy with full integration features:
    - Proper timing and data validation
    - Parameter optimization support
    - Result persistence and analysis
    - Multi-timeframe signal filtering
    - Volume confirmation option
    """

    def __init__(
        self,
        config: BacktestConfig,
        symbol_ltf: str,
        tframe_ltf: str,
        symbol_htf: str,
        tframe_htf: str,
        start_date: str,
        params: Dict,
        data_manager: Optional[DataManager] = None,
    ):
        super().__init__(config)
        self.dm = data_manager or DataManager(name="bitget")
        self.symbol_ltf = symbol_ltf
        self.tframe_ltf = tframe_ltf
        self.symbol_htf = symbol_htf
        self.tframe_htf = tframe_htf
        self.start_date = start_date
        self.params = self._validate_params(params)

        # Strategy metadata
        self.strategy_name = "MACD_BB_Enhanced"
        self.version = "2.0"

    def _validate_params(self, params: Dict) -> Dict:
        """Validate and set default parameters"""
        default_params = {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "signal_mode": "mean_reversion",
            "volume_filter": False,
            "volume_ma_period": 20,
            "volume_threshold": 1.2,
            "atr_filter": False,
            "atr_period": 14,
            "min_atr_percentile": 20,
        }

        # Merge with defaults
        validated_params = default_params.copy()
        validated_params.update(params)

        # Validate ranges
        if not 5 <= validated_params["bb_period"] <= 100:
            raise ValueError(f"bb_period must be between 5 and 100")
        if not 1.0 <= validated_params["bb_std_dev"] <= 4.0:
            raise ValueError(f"bb_std_dev must be between 1.0 and 4.0")
        if validated_params["macd_fast"] >= validated_params["macd_slow"]:
            raise ValueError("macd_fast must be less than macd_slow")

        return validated_params

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data validation with detailed diagnostics"""
        required_cols = ["open", "high", "low", "close", "signal"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for data quality issues
        quality_report = {
            "total_rows": len(df),
            "date_range": f"{df.index.min()} to {df.index.max()}",
            "missing_values": {},
            "zero_values": {},
            "negative_values": {},
        }

        critical_cols = ["open", "high", "low", "close", "volume"]
        for col in critical_cols:
            if col in df.columns:
                quality_report["missing_values"][col] = df[col].isna().sum()
                quality_report["zero_values"][col] = (df[col] == 0).sum()
                quality_report["negative_values"][col] = (df[col] < 0).sum()

                # Fix issues
                if df[col].isna().any():
                    print(
                        f"Warning: {quality_report['missing_values'][col]} NaN values found in {col}, forward filling..."
                    )
                    df[col] = df[col].fillna(method="ffill")

                # Ensure no negative prices
                if col != "volume" and (df[col] < 0).any():
                    print(
                        f"Warning: Negative values found in {col}, setting to absolute value..."
                    )
                    df[col] = df[col].abs()

        # Validate OHLC relationships
        invalid_candles = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        ).sum()

        if invalid_candles > 0:
            print(f"Warning: {invalid_candles} invalid OHLC relationships detected")

        return df

    def _calculate_volume_filter(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume-based signal filter"""
        if "volume" not in df.columns or not self.params["volume_filter"]:
            return pd.Series(True, index=df.index)

        volume_ma = df["volume"].rolling(window=self.params["volume_ma_period"]).mean()

        return df["volume"] > (volume_ma * self.params["volume_threshold"])

    def _calculate_atr_filter(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR-based volatility filter"""
        if not self.params["atr_filter"]:
            return pd.Series(True, index=df.index)

        if "atr" not in df.columns:
            df["atr"] = talib.ATR(
                df["high"], df["low"], df["close"], timeperiod=self.params["atr_period"]
            )

        # Only trade when ATR is above a certain percentile
        atr_threshold = df["atr"].quantile(self.params["min_atr_percentile"] / 100)
        return df["atr"] > atr_threshold

    def _calculate_signals(self, merged: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Enhanced signal calculation with multiple filters"""
        bb_period = params["bb_period"]
        signal_mode = params.get("signal_mode", "mean_reversion")

        # Get BB column names
        bb_upper_col = f"BB_UPPER_{bb_period}"
        bb_lower_col = f"BB_LOWER_{bb_period}"
        bb_middle_col = f"BB_MIDDLE_{bb_period}"

        if bb_upper_col not in merged.columns or bb_lower_col not in merged.columns:
            raise ValueError(
                f"Bollinger Bands columns not found. Expected: {bb_upper_col}, {bb_lower_col}"
            )

        # Calculate BB position and width
        merged["bb_position"] = (merged["close"] - merged[bb_lower_col]) / (
            merged[bb_upper_col] - merged[bb_lower_col]
        )
        merged["bb_width"] = (merged[bb_upper_col] - merged[bb_lower_col]) / merged[
            bb_middle_col
        ]

        # Apply volume filter
        volume_filter = self._calculate_volume_filter(merged)

        # Apply ATR filter
        atr_filter = self._calculate_atr_filter(merged)

        # Generate signals based on mode
        if signal_mode == "mean_reversion":
            # Buy when price touches lower band in uptrend
            buy_signal = (
                (merged["trend"] == 1)
                & (merged["close"] < merged[bb_lower_col])
                & volume_filter
                & atr_filter
            )
            # Sell when price touches upper band in downtrend
            sell_signal = (
                (merged["trend"] == -1)
                & (merged["close"] > merged[bb_upper_col])
                & volume_filter
                & atr_filter
            )
        elif signal_mode == "trend_following":
            # Buy when price breaks above upper band in uptrend
            buy_signal = (
                (merged["trend"] == 1)
                & (merged["close"] > merged[bb_upper_col])
                & (
                    merged["close"].shift(1) <= merged[bb_upper_col].shift(1)
                )  # Breakout
                & volume_filter
                & atr_filter
            )
            # Sell when price breaks below lower band in downtrend
            sell_signal = (
                (merged["trend"] == -1)
                & (merged["close"] < merged[bb_lower_col])
                & (
                    merged["close"].shift(1) >= merged[bb_lower_col].shift(1)
                )  # Breakdown
                & volume_filter
                & atr_filter
            )
        elif signal_mode == "hybrid":
            # Combine both approaches based on BB width
            wide_bb = (
                merged["bb_width"] > merged["bb_width"].rolling(50).mean()
            ).astype(bool)

            # Mean reversion when BB is wide
            mr_buy = (
                (merged["trend"] == 1)
                & (merged["close"] < merged[bb_lower_col])
                & wide_bb
            )
            mr_sell = (
                (merged["trend"] == -1)
                & (merged["close"] > merged[bb_upper_col])
                & wide_bb
            )

            # Trend following when BB is narrow
            tf_buy = (
                (merged["trend"] == 1)
                & (merged["close"] > merged[bb_upper_col])
                &( ~wide_bb)
            )
            tf_sell = (
                (merged["trend"] == -1)
                & (merged["close"] < merged[bb_lower_col])
                & (~wide_bb)
            )

            buy_signal = (mr_buy | tf_buy) & volume_filter & atr_filter
            sell_signal = (mr_sell | tf_sell) & volume_filter & atr_filter
        else:
            # Default to mean reversion
            buy_signal = (merged["trend"] == 1) & (
                merged["close"] < merged[bb_lower_col]
            )
            sell_signal = (merged["trend"] == -1) & (
                merged["close"] > merged[bb_upper_col]
            )

        # Create signal array
        signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

        # Apply proper shift to prevent lookahead bias
        merged["signal"] = pd.Series(signal, index=merged.index).shift(1).fillna(0)

        # Add signal strength metric
        merged["signal_strength"] = np.where(
            merged["signal"] != 0,
            abs(merged["bb_position"] - 0.5) * 2,  # 0 to 1 scale
            0,
        )

        return merged

    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals with enhanced features"""
        # 1. Load Data
        ltf_data = self.dm.from_local(self.symbol_ltf, self.tframe_ltf, self.start_date)
        htf_data = self.dm.from_local(self.symbol_htf, self.tframe_htf, self.start_date)

        # 2. Calculate LTF Indicators
        ltf_factory = IndicatorFactory(ltf_data)
        df_ltf = (
            ltf_factory.add_bollinger_bands(
                period=self.params["bb_period"],
                nbdevup=self.params["bb_std_dev"],
                nbdevdn=self.params["bb_std_dev"],
            )
            .add_macd(
                fastperiod=self.params["macd_fast"],
                slowperiod=self.params["macd_slow"],
                signalperiod=self.params["macd_signal"],
            )
            .add_atr(14)  # Always add ATR for position sizing
            .add_rsi(14)  # Additional indicator for signal confirmation
            .get_data()
        )

        # 3. Calculate HTF Indicators
        htf_factory = IndicatorFactory(htf_data)
        df_htf = (
            htf_factory.add_macd(
                fastperiod=self.params["macd_fast"],
                slowperiod=self.params["macd_slow"],
                signalperiod=self.params["macd_signal"],
            )
            .add_atr(14)
            .get_data()
        )

        # 4. Calculate HTF trend with enhanced logic
        df_htf["macd_bullish"] = df_htf["MACD"] > df_htf["MACD_signal"]
        df_htf["macd_histogram_positive"] = df_htf["MACD_hist"] > 0
        df_htf["macd_momentum"] = df_htf["MACD_hist"] > df_htf["MACD_hist"].shift(1)
        macd_bullish = df_htf["macd_bullish"].shift(1).fillna(False).astype(bool)
        macd_momentum = df_htf["macd_momentum"].shift(1).fillna(False).astype(bool)

        # Multi-condition trend determination
        df_htf["trend"] = np.where(
            macd_bullish & macd_momentum,
            1,
            np.where(
                ~macd_bullish & ~macd_momentum,
                -1,
                0,
            ),
        )

        # 5. Merge dataframes
        df_htf_resampled = df_htf.reindex(df_ltf.index, method="ffill")
        merged = df_ltf.join(df_htf_resampled, rsuffix="_htf").dropna()

        # 6. Ensure required columns exist
        if "ATR_14" in merged.columns:
            merged["atr"] = merged["ATR_14"]
        elif "atr" not in merged.columns:
            print("Warning: Adding ATR calculation as it's missing")
            merged["atr"] = talib.ATR(
                merged["high"], merged["low"], merged["close"], timeperiod=14
            )

        # 7. Generate signals
        merged = self._calculate_signals(merged, self.params)

        # 8. Add metadata
        merged["strategy_name"] = self.strategy_name
        merged["strategy_version"] = self.version

        # 9. Validate final data
        merged = self._validate_data(merged)

        return merged

    def validate_strategy_data(self) -> Tuple[bool, Dict]:
        """Enhanced validation with detailed reporting"""
        try:
            df = self.generate_signals()

            validation_report = {
                "status": "PASSED",
                "data_shape": df.shape,
                "date_range": f"{df.index.min()} to {df.index.max()}",
                "signal_distribution": df["signal"].value_counts().to_dict(),
                "total_signals": (df["signal"] != 0).sum(),
                "signal_rate": (df["signal"] != 0).sum() / len(df),
                "missing_values": {},
                "issues": [],
            }

            # Check critical columns
            critical_cols = ["open", "high", "low", "close", "signal", "atr"]
            for col in critical_cols:
                if col in df.columns:
                    missing = df[col].isna().sum()
                    validation_report["missing_values"][col] = missing
                    if missing > 0:
                        validation_report["issues"].append(
                            f"{col} has {missing} missing values"
                        )
                else:
                    validation_report["issues"].append(f"{col} column is missing!")
                    validation_report["status"] = "FAILED"

            # Check signal quality
            if validation_report["total_signals"] == 0:
                validation_report["issues"].append("No trading signals generated!")
                validation_report["status"] = "WARNING"
            elif validation_report["signal_rate"] < 0.001:
                validation_report["issues"].append(
                    f"Very low signal rate: {validation_report['signal_rate']:.4%}"
                )
                validation_report["status"] = "WARNING"

            # Print summary
            print(f"\nValidation Status: {validation_report['status']}")
            print(f"Data shape: {validation_report['data_shape']}")
            print(f"Date range: {validation_report['date_range']}")
            print(f"Signal distribution:\n{validation_report['signal_distribution']}")
            print(f"Total signals: {validation_report['total_signals']}")
            print(f"Signal rate: {validation_report['signal_rate']:.2%}")

            if validation_report["issues"]:
                print("\nIssues found:")
                for issue in validation_report["issues"]:
                    print(f"  - {issue}")

            return validation_report["status"] != "FAILED", validation_report

        except Exception as e:
            print(f"Strategy validation failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False, {"status": "FAILED", "error": str(e)}

    def save_optimization_results(
        self, study: optuna.study.Study, filename: str = None
    ):
        """Save optimization results for later analysis"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.strategy_name}_optimization_{timestamp}.json"

        filepath = OPTIMIZER_DIR / filename

        results = {
            "strategy_name": self.strategy_name,
            "version": self.version,
            "optimization_date": datetime.now().isoformat(),
            "symbols": {
                "ltf": f"{self.symbol_ltf}_{self.tframe_ltf}",
                "htf": f"{self.symbol_htf}_{self.tframe_htf}",
            },
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "all_trials": [
                {
                    "number": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": str(trial.state),
                }
                for trial in study.trials
            ],
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Optimization results saved to: {filepath}")
        return filepath


    def analyze_parameter_importance(self, study: optuna.study.Study) -> Dict:
        """Analyze which parameters have the most impact on results"""
        if len(study.trials) < 10:
            return {"error": "Not enough trials for analysis"}

        importance = {}

        try:
            from optuna.importance import get_param_importances

            param_importance = get_param_importances(study)
            importance["parameter_importance"] = param_importance
        except Exception as e:
            importance["parameter_importance"] = f"Could not calculate importance: {str(e)}"

        # Analyze parameter ranges for best trials
        n_best = min(10, len(study.trials))
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:n_best]

        param_stats = {}
        for param_name in study.best_params.keys():
            values = [
                t.params.get(param_name) for t in best_trials if param_name in t.params
            ]

            # Only calculate statistics if values are numeric
            try:
                numeric_values = [float(v) for v in values]
                param_stats[param_name] = {
                    "mean": np.mean(numeric_values),
                    "std": np.std(numeric_values),
                    "min": np.min(numeric_values),
                    "max": np.max(numeric_values),
                }
            except (ValueError, TypeError):
                # Non-numeric parameter, just store unique values
                param_stats[param_name] = {
                    "type": "categorical",
                    "unique_values": list(set(values)),
                }

        importance["best_trials_param_stats"] = param_stats
        return importance


def create_optimizer_strategy_function(
    trial: optuna.trial.Trial, data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Enhanced optimizer function with additional parameters"""
    # 1. Define expanded search space
    params = {
        "bb_period": trial.suggest_int("bb_period", 10, 50),
        "bb_std_dev": trial.suggest_float("bb_std_dev", 1.5, 3.5),
        "macd_fast": trial.suggest_int("macd_fast", 5, 20),
        "macd_slow": trial.suggest_int("macd_slow", 21, 60),
        "macd_signal": trial.suggest_int("macd_signal", 5, 20),
        "signal_mode": trial.suggest_categorical(
            "signal_mode", ["mean_reversion", "trend_following", "hybrid"]
        ),
        "volume_filter": trial.suggest_categorical("volume_filter", [True, False]),
    }

    # Add conditional parameters
    if params["volume_filter"]:
        params["volume_ma_period"] = trial.suggest_int("volume_ma_period", 10, 50)
        params["volume_threshold"] = trial.suggest_float("volume_threshold", 1.0, 2.0)

    # 2. Create strategy instance
    dummy_config = BacktestConfig()
    strategy_instance = MacdBBStrategy(
        config=dummy_config,
        symbol_ltf="TRX/USDT:USDT",
        tframe_ltf="15m",
        symbol_htf="TRX/USDT:USDT",
        tframe_htf="1h",
        start_date="2024-01-01",
        params=params,
    )

    # 3. Override data loading
    def mock_from_local(symbol, timeframe, start_date):
        if timeframe == "15m":
            return data_dict["ltf_data"].copy()
        else:
            return data_dict["htf_data"].copy()

    strategy_instance.dm.from_local = mock_from_local

    return strategy_instance.generate_signals()


def create_custom_objective_functions():
    """Factory for different objective functions"""

    def sharpe_focused(analysis: BacktestAnalysis) -> float:
        """Focus on Sharpe ratio with minimum trade requirement"""
        metrics = analysis.metrics
        sharpe = metrics.get("sharpe_ratio", -1.0)
        total_trades = metrics.get("total_trades", 0)

        if total_trades < 20:
            return -1.0

        return sharpe

    def balanced_score(analysis: BacktestAnalysis) -> float:
        """Balance between multiple metrics"""
        metrics = analysis.metrics

        sharpe = metrics.get("sharpe_ratio", -1.0)
        profit_factor = metrics.get("profit_factor", 0.0)
        win_rate = metrics.get("global_win_rate", 0.0)
        total_trades = metrics.get("total_trades", 0)
        max_dd = metrics.get("max_drawdown_equity", 1.0)

        # Hard constraints
        if total_trades < 25 or profit_factor < 1.1 or sharpe < 0.1:
            return -1.0

        # Weighted score
        score = (
            sharpe * 0.4
            + np.log1p(profit_factor - 1) * 0.3
            + win_rate * 0.2
            + (1 - max_dd) * 0.1
        )

        # Trade frequency bonus
        if total_trades > 100:
            score *= 1.1

        return score if np.isfinite(score) else -1.0

    def risk_adjusted_returns(analysis: BacktestAnalysis) -> float:
        """Focus on risk-adjusted returns"""
        metrics = analysis.metrics

        roi = metrics.get("roi_pct", -100.0)
        max_dd = metrics.get("max_drawdown_equity", 1.0)
        sortino = metrics.get("sortino_ratio", -1.0)
        total_trades = metrics.get("total_trades", 0)

        if total_trades < 20 or roi < 0:
            return -1.0

        # Return over max drawdown ratio
        if max_dd > 0:
            risk_adjusted = (roi / 100) / max_dd
        else:
            risk_adjusted = 0

        # Combine with Sortino
        score = risk_adjusted * 0.6 + sortino * 0.4

        return score if np.isfinite(score) else -1.0

    return {
        "sharpe_focused": sharpe_focused,
        "balanced": balanced_score,
        "risk_adjusted": risk_adjusted_returns,
    }


def main():
    """Enhanced main function with better organization"""
    # --- Configuration ---
    CONFIG = {
        "RUN_VALIDATION": False,
        "RUN_SINGLE_BACKTEST": True,
        "RUN_COMPARATIVE_ANALYSIS": False,
        "RUN_OPTIMIZATION": False,
        "SAVE_RESULTS": False,
        "GENERATE_REPORT": False,
        "DEEP_PLOT": False,
    }

    # --- Strategy Parameters ---
    STRATEGY_CONFIG = {
        "symbol_ltf": "TRX/USDT:USDT",
        "tframe_ltf": "15m",
        "symbol_htf": "TRX/USDT:USDT",
        "tframe_htf": "1h",
        "start_date": "2024-01-01",
    }

    # --- Best Known Parameters ---
    BEST_PARAMS = {
        "bb_period": 42,
        "bb_std_dev": 1.8,
        "macd_fast": 5,
        "macd_slow": 60,
        "macd_signal": 13,
        "signal_mode": "trend_following",
        "volume_filter": True,
        "volume_ma_period": 14,
        "volume_threshold": 1.6,
    }

    print(f"\n{'='*60}")
    print(f"MACD + Bollinger Bands Strategy - Enhanced Version {datetime.now()}")
    print(f"{'='*60}\n")

    # --- Part 0: Validate Strategy Data ---
    if CONFIG["RUN_VALIDATION"]:
        print("\n📊 Validating Strategy Data...")
        print("-" * 50)

        test_config = BacktestConfig()
        test_strategy = MacdBBStrategy(
            config=test_config,
            **STRATEGY_CONFIG,
            params=BEST_PARAMS,
        )

        valid, report = test_strategy.validate_strategy_data()
        if not valid:
            print("❌ Strategy validation failed! Fix issues before proceeding.")
            return
        else:
            print("✅ Strategy validation passed!")

    # --- Part 1: Run Single Backtest ---
    if CONFIG["RUN_SINGLE_BACKTEST"]:
        print("\n💹 Running Single Backtest...")
        print("-" * 50)

        leverage = 10
        sizer_atr = AtrBandsSizer(
            risk_pct=0.01,
            atr_multiplier=2.0,
            risk_reward_ratio=1.5,
            leverage=leverage,
            max_position_pct=0.2,
            min_atr_multiplier=0.5,
        )

        sizer_percent = PercentBalanceSizer(
            percent=0.1
        )

        config_single = BacktestConfig(
            initial_balance=3000.0,
            leverage=leverage,
            trading_mode=TradingMode.CROSS,
            sizing_strategy=sizer_percent,
            # Enhanced realism settings
            enable_slippage=True,
            base_slippage_bps=1.0,
            reduced_weekend_liquidity=True,
            # Trailing stops
            enable_trailing_stop=True,
            breakeven_trigger_pct=0.01,
            breakeven_sl_pct=0.001,
            midpoint_trigger_pct=0.5,
            midpoint_tp_extension_pct=0.3,
            midpoint_sl_adjustment_pct=0.2,
        )

        strategy_single = MacdBBStrategy(
            config=config_single,
            **STRATEGY_CONFIG,
            params=BEST_PARAMS,
        )

        analysis = strategy_single.run_single(
            generate_quantstats_report=CONFIG["GENERATE_REPORT"]
        )

        if CONFIG["DEEP_PLOT"]:
            analysis.show_complete_analysis_dashboard()
        analysis.show_complete_graph_dashboard()

        # Save results
        if CONFIG["SAVE_RESULTS"]:
            results_df = analysis.results_df
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"{ANALYSIS_DIR}\\MACD_BB_results_{timestamp}.csv"
            results_df.to_csv(results_file)
            print(f"\n💾 Results saved to: {results_file}")

    # --- Part 2: Run Comparative Analysis ---
    if CONFIG["RUN_COMPARATIVE_ANALYSIS"]:
        print("\n🔄 Running Comparative Analysis...")
        print("-" * 50)

        config_comparison = BacktestConfig(
            initial_balance=3000.0,
            leverage=10,
            trading_mode=TradingMode.CROSS,
            enable_slippage=True,
            base_slippage_bps=1.0,
            enable_trailing_stop=True,
            breakeven_trigger_pct=0.01,
            breakeven_sl_pct=0.001,
            midpoint_trigger_pct=0.5,
            midpoint_tp_extension_pct=0.3,
            midpoint_sl_adjustment_pct=0.2,
        )

        strategy_comparison = MacdBBStrategy(
            config=config_comparison,
            **STRATEGY_CONFIG,
            params=BEST_PARAMS,
        )

        results = strategy_comparison.run_comparative()

    # --- Part 3: Run Optimization ---
    if CONFIG["RUN_OPTIMIZATION"]:
        print("\n🔧 Running Parameter Optimization...")
        print("-" * 50)

        # Get objective functions
        objectives = create_custom_objective_functions()
        selected_objective = objectives["balanced"]

        # Optimization config
        atr_sizer_opt = AtrBandsSizer(
            risk_pct=0.01, atr_multiplier=2.0, risk_reward_ratio=1.5, leverage=10
        )

        percent_sizer_opt = PercentBalanceSizer(percent=0.1)

        config_optimizer = BacktestConfig(
            initial_balance=30.0,
            leverage=10,
            trading_mode=TradingMode.CROSS,
            sizing_strategy=percent_sizer_opt,
            # Enhanced realism settings
            enable_slippage=True,
            base_slippage_bps=1.0,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0006,
            reduced_weekend_liquidity=True,
            # Trailing stops
            enable_trailing_stop=True,
            breakeven_trigger_pct=0.01,
            breakeven_sl_pct=0.001,
            midpoint_trigger_pct=0.5,
            midpoint_tp_extension_pct=0.3,
            midpoint_sl_adjustment_pct=0.2,
        )

        # Load data for optimization
        dm = DataManager(name="bitget")
        data_for_opt = {
            "ltf_data": dm.from_local(
                STRATEGY_CONFIG["symbol_ltf"],
                STRATEGY_CONFIG["tframe_ltf"],
                STRATEGY_CONFIG["start_date"],
            ),
            "htf_data": dm.from_local(
                STRATEGY_CONFIG["symbol_htf"],
                STRATEGY_CONFIG["tframe_htf"],
                STRATEGY_CONFIG["start_date"],
            ),
        }

        # Run optimization
        study = run_optimization(
            data_dict=data_for_opt,
            config=config_optimizer,
            strategy_function=create_optimizer_strategy_function,
            n_trials=10,
            objective_function=selected_objective,
        )

        print(f"\n🏆 Optimization Results:")
        print(f"Best Score: {study.best_value:.4f}")
        print(f"Best Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save optimization results
        if CONFIG["SAVE_RESULTS"]:
            # Create strategy instance to use save method
            opt_strategy = MacdBBStrategy(
                config=config_optimizer,
                **STRATEGY_CONFIG,
                params=BEST_PARAMS,
            )
            opt_strategy.save_optimization_results(study)

            # Analyze parameter importance
            importance = opt_strategy.analyze_parameter_importance(study)
            print("\n📈 Parameter Importance Analysis:")
            if "parameter_importance" in importance:
                print("Parameter Importance:", importance["parameter_importance"])
            if "best_trials_param_stats" in importance:
                print("\nBest Trials Parameter Statistics:")
                for param, stats in importance["best_trials_param_stats"].items():
                    print(f"  {param}:")
                    for stat_name, stat_value in stats.items():
                        print(f"    {stat_name}: {stat_value}")

    print(f"\n{'='*60}")
    print("Strategy execution completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
