import pandas as pd
import numpy as np
from typing import Dict
import optuna
import talib

# --- All Project Imports ---
from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Config.Enums import TradingMode
from AlgoTrade.Strat.Class.BaseStrategy import BaseStrategy
from AlgoTrade.Factories.IndicatorFactory import IndicatorFactory
from AlgoTrade.Sizing.AtrBandsSizer import AtrBandsSizer
from AlgoTrade.Optimizer.StudyRunner import run_optimization
from AlgoTrade.Analysis.BacktestAnalysis import BacktestAnalysis


class MacdBBStrategy(BaseStrategy):
    """
    Enhanced MACD+BB strategy with proper timing and data validation.
    Uses MACD on HTF for trend and BB on LTF for entries.
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
    ):
        super().__init__(config)
        self.dm = DataManager(name="bitget")
        self.symbol_ltf = symbol_ltf
        self.tframe_ltf = tframe_ltf
        self.symbol_htf = symbol_htf
        self.tframe_htf = tframe_htf
        self.start_date = start_date
        self.params = params

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that dataframe has required columns for backtesting"""
        required_cols = ["open", "high", "low", "close", "signal"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Ensure no NaN values in critical columns
        critical_cols = ["open", "high", "low", "close"]
        for col in critical_cols:
            if df[col].isna().any():
                print(f"Warning: NaN values found in {col}, forward filling...")
                df[col] = df[col].fillna(method="ffill")

        return df

    def _calculate_signals(self, merged: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate signals with enhanced timing logic"""
        bb_period = params["bb_period"]
        signal_mode = params.get("signal_mode", "mean_reversion")

        # Get BB column names
        bb_upper_col = f"BB_UPPER_{bb_period}"
        bb_lower_col = f"BB_LOWER_{bb_period}"

        if bb_upper_col not in merged.columns or bb_lower_col not in merged.columns:
            raise ValueError(
                f"Bollinger Bands columns not found. Expected: {bb_upper_col}, {bb_lower_col}"
            )

        # Generate signals based on mode
        if signal_mode == "mean_reversion":
            buy_signal = (merged["trend"] == 1) & (
                merged["close"] < merged[bb_lower_col]
            )
            sell_signal = (merged["trend"] == -1) & (
                merged["close"] > merged[bb_upper_col]
            )
        elif signal_mode == "trend_following":
            buy_signal = (merged["trend"] == 1) & (
                merged["close"] > merged[bb_upper_col]
            )
            sell_signal = (merged["trend"] == -1) & (
                merged["close"] < merged[bb_lower_col]
            )
        else:
            # Default to mean reversion if unknown mode
            buy_signal = (merged["trend"] == 1) & (
                merged["close"] < merged[bb_lower_col]
            )
            sell_signal = (merged["trend"] == -1) & (
                merged["close"] > merged[bb_upper_col]
            )

        # Create signal array
        signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

        # Apply proper shift to prevent lookhead bias
        merged["signal"] = pd.Series(signal, index=merged.index).shift(1).fillna(0)

        return merged

    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals with enhanced timing and validation"""
        # 1. Load Data
        ltf_data = self.dm.from_local(self.symbol_ltf, self.tframe_ltf, self.start_date)
        htf_data = self.dm.from_local(self.symbol_htf, self.tframe_htf, self.start_date)

        # 2. Calculate LTF Indicators (including ATR for position sizing)
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
            .add_atr(14)  # REQUIRED for enhanced BacktestRunner
            .get_data()
        )

        # 3. Calculate HTF Indicators
        htf_factory = IndicatorFactory(htf_data)
        df_htf = htf_factory.add_macd(
            fastperiod=self.params["macd_fast"],
            slowperiod=self.params["macd_slow"],
            signalperiod=self.params["macd_signal"],
        ).get_data()

        # 4. Calculate HTF trend with proper timing (avoid lookhead bias)
        df_htf["macd_bullish"] = df_htf["MACD"] > df_htf["MACD_signal"]
        # Use previous period's MACD signal to determine current trend
        df_htf["trend"] = np.where(
            df_htf["macd_bullish"].shift(1),
            1,
            np.where(df_htf["macd_bullish"].shift(1) == False, -1, 0),
        )

        # 5. Merge dataframes with proper alignment
        df_htf_resampled = df_htf.reindex(df_ltf.index, method="ffill")
        merged = df_ltf.join(df_htf_resampled, rsuffix="_htf").dropna()

        # 6. Ensure ATR column exists (required by enhanced BacktestRunner)
        if "ATR_14" in merged.columns:
            merged["atr"] = merged["ATR_14"]
        elif "atr" not in merged.columns:
            print("Warning: Adding ATR calculation as it's missing")
            merged["atr"] = talib.ATR(
                merged["high"], merged["low"], merged["close"], timeperiod=14
            )

        # 8. Generate signals with enhanced logic
        merged = self._calculate_signals(merged, self.params)

        # 9. Validate final data
        merged = self._validate_data(merged)

        return merged

    def validate_strategy_data(self) -> bool:
        """Validate that strategy data is correct for backtesting"""
        try:
            df = self.generate_signals()
            
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Signal distribution:\n{df['signal'].value_counts()}")
            print(f"Missing values in critical columns:")
            
            # Only check columns that the strategy should provide
            critical_cols = ["open", "high", "low", "close", "signal", "atr"]
            for col in critical_cols:
                if col in df.columns:
                    missing = df[col].isna().sum()
                    print(f"  {col}: {missing} NaN values")
                else:
                    print(f"  {col}: MISSING COLUMN!")
            
            # Note: next_open will be added by BacktestRunner, so don't check for it here
            print("Note: next_open column will be added automatically by BacktestRunner")
            
            return True
            
        except Exception as e:
            print(f"Strategy validation failed: {e}")
            return False


def create_optimizer_strategy_function(
    trial: optuna.trial.Trial, data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Enhanced optimizer function with data validation"""
    # 1. Define Search Space
    params = {
        "bb_period": trial.suggest_int("bb_period", 10, 50),
        "bb_std_dev": trial.suggest_float("bb_std_dev", 1.5, 3.5),
        "macd_fast": trial.suggest_int("macd_fast", 5, 20),
        "macd_slow": trial.suggest_int("macd_slow", 21, 60),
        "macd_signal": trial.suggest_int("macd_signal", 5, 20),
        "signal_mode": trial.suggest_categorical(
            "signal_mode", ["mean_reversion", "trend_following"]
        ),
    }

    # 2. Create strategy instance with dummy config
    dummy_config = BacktestConfig()
    strategy_instance = MacdBBStrategy(
        config=dummy_config,
        symbol_ltf="TRX/USDT:USDT",  # Match your test symbol
        tframe_ltf="15m",
        symbol_htf="TRX/USDT:USDT",
        tframe_htf="1h",
        start_date="2024-01-01",
        params=params,
    )

    # 3. Override data loading to use provided data
    def mock_from_local(symbol, timeframe, start_date):
        if timeframe == "15m":
            return data_dict["ltf_data"].copy()
        else:
            return data_dict["htf_data"].copy()

    strategy_instance.dm.from_local = mock_from_local

    return strategy_instance.generate_signals()


def main():
    # --- Control Panel ---
    RUN_COMPARATIVE_ANALYSIS = False
    RUN_SINGLE_BACKTEST = True
    RUN_OPTIMIZATION = False
    VALIDATE_STRATEGY = True  # New validation step

    # --- Common Strategy Parameters ---
    symbol_ltf = "TRX/USDT:USDT"
    tframe_ltf = "15m"
    symbol_htf = "TRX/USDT:USDT"
    tframe_htf = "1h"
    start_date = "2024-01-01"

    # --- Part 0: Validate Strategy Data (NEW) ---
    if VALIDATE_STRATEGY:
        print("\n" + "=" * 50 + "\n--- Validating Strategy Data ---\n" + "=" * 50)

        # Test with default params
        test_params = {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "signal_mode": "mean_reversion",
        }

        test_config = BacktestConfig()
        test_strategy = MacdBBStrategy(
            config=test_config,
            symbol_ltf=symbol_ltf,
            tframe_ltf=tframe_ltf,
            symbol_htf=symbol_htf,
            tframe_htf=tframe_htf,
            start_date=start_date,
            params=test_params,
        )

        if not test_strategy.validate_strategy_data():
            print("Strategy validation failed! Fix issues before proceeding.")
            return
        else:
            print("Strategy validation passed!")

    # --- Part 1: Run Single Backtest ---
    if RUN_SINGLE_BACKTEST:
        print("\n" + "=" * 50 + "\n--- Starting Single Backtest ---\n" + "=" * 50)

        leverage = 10
        sizer = AtrBandsSizer(
            risk_pct=0.01,
            atr_multiplier=2.0,
            risk_reward_ratio=1.5,
            leverage=leverage,
        )

        # Use ENHANCED BacktestConfig
        config_for_single = BacktestConfig(
            initial_balance=3000.0,
            leverage=leverage,
            trading_mode=TradingMode.CROSS,
            sizing_strategy=sizer,
            # Enhanced settings for realism
            enable_slippage=True,
            base_slippage_bps=1.0,  # 1 basis point
            enable_market_impact=False,  # Start conservative
            maker_fee_rate=0.0001,  # 0.01%
            taker_fee_rate=0.0004,  # 0.04%
            # Trading hours (optional for crypto)
            enable_trading_hours=False,
            reduced_weekend_liquidity=True,
            # Trailing stops
            enable_trailing_stop=False,
            breakeven_trigger_pct=0.01,
            breakeven_sl_pct=0.001,
        )

        # Best params from optimization or manual tuning
        best_params = {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "signal_mode": "mean_reversion",
        }

        strategy_single = MacdBBStrategy(
            config=config_for_single,
            symbol_ltf=symbol_ltf,
            tframe_ltf=tframe_ltf,
            symbol_htf=symbol_htf,
            tframe_htf=tframe_htf,
            start_date=start_date,
            params=best_params,
        )

        analysis = strategy_single.run_single(generate_quantstats_report=True)

    # --- Part 2: Run Comparative Analysis ---
    if RUN_COMPARATIVE_ANALYSIS:
        print("\n" + "=" * 50 + "\n--- Starting Comparative Analysis ---\n" + "=" * 50)

        config_for_comparison = BacktestConfig(
            initial_balance=3000.0,
            leverage=10,
            trading_mode=TradingMode.CROSS,
            # Use enhanced settings here too
            enable_slippage=True,
            base_slippage_bps=1.0,
        )

        default_params = {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "signal_mode": "mean_reversion",
        }

        strategy_comparison = MacdBBStrategy(
            config=config_for_comparison,
            symbol_ltf=symbol_ltf,
            tframe_ltf=tframe_ltf,
            symbol_htf=symbol_htf,
            tframe_htf=tframe_htf,
            start_date=start_date,
            params=default_params,
        )
        strategy_comparison.run_comparative()

    # --- Part 3: Run Optimizer ---
    if RUN_OPTIMIZATION:
        print(
            "\n" + "=" * 50 + "\n--- Starting Parameter Optimization ---\n" + "=" * 50
        )

        def trade_sharpe_pf_score(analysis: BacktestAnalysis) -> float:
            """Custom objective function balancing multiple metrics"""
            metrics = analysis.metrics

            sharpe = metrics.get("sharpe_ratio", -1.0)
            profit_factor = metrics.get("profit_factor", 0.0)
            total_trades = metrics.get("total_trades", 0)

            # Hard constraints
            if total_trades < 25 or profit_factor < 1.1 or sharpe < 0.1:
                return -1.0

            # Scoring with log scaling
            pf_score = np.log1p(profit_factor - 1)
            trades_score = np.log1p(total_trades)
            score = sharpe * pf_score * trades_score

            return score if np.isfinite(score) else -1.0

        atr_sizer_for_opt = AtrBandsSizer(
            risk_pct=0.01, atr_multiplier=2.0, risk_reward_ratio=1.5, leverage=10
        )

        config_for_optimizer = BacktestConfig(
            initial_balance=3000.0,
            leverage=10,
            trading_mode=TradingMode.CROSS,
            sizing_strategy=atr_sizer_for_opt,
            enable_slippage=True,
            base_slippage_bps=1.0,
        )

        dm = DataManager(name="bitget")
        data_for_opt = {
            "ltf_data": dm.from_local(symbol_ltf, tframe_ltf, start_date),
            "htf_data": dm.from_local(symbol_htf, tframe_htf, start_date),
        }

        study = run_optimization(
            data_dict=data_for_opt,
            config=config_for_optimizer,
            strategy_function=create_optimizer_strategy_function,
            n_trials=50,  # Reduced for testing
            objective_function=trade_sharpe_pf_score,
        )

        print(f"\nBest Score: {study.best_value}")
        print(f"Best Parameters: {study.best_params}")


if __name__ == "__main__":
    main()
