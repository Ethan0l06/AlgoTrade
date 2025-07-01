import pandas as pd
import numpy as np
from typing import Dict, Optional
import optuna
from datetime import datetime

# --- Project Imports ---
from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Utils.VectorizedSignals import VectorizedSignals
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Config.Enums import TradingMode
from AlgoTrade.Strat.Class.BaseStrategy import BaseStrategy
from AlgoTrade.Factories.IndicatorFactory import IndicatorFactory
from AlgoTrade.Sizing.RiskBasedSizer import RiskBasedSizer
from AlgoTrade.Sizing.PercentBalanceSizer import PercentBalanceSizer
from AlgoTrade.Optimizer.StudyRunner import run_optimization
from AlgoTrade.Config.Paths import OPTIMIZER_DIR, ANALYSIS_DIR


class DMIADXRoStrategy(BaseStrategy):
    """
    DMI ADX Williams %R Trend-Following Scalping Strategy using enhanced BaseStrategy.

    Strategy Rules:
    - Long: ADX > 50 + DI+ > DI- + Williams %R < -80 (oversold in uptrend)
    - Short: ADX > 50 + DI- > DI+ + Williams %R > -20 (overbought in downtrend)
    - Stop Loss: Price hits SMA 50 (trend reversal)
    - Take Profit: Highest high of last 30 candles (long) / Lowest low of last 30 candles (short)
    - Early Exit: Trend direction reverses (DI crossover)
    """

    def __init__(
        self,
        config: BacktestConfig,
        symbol: str,
        timeframe: str,
        start_date: str,
        params: Optional[Dict] = None,
        data_manager: Optional[DataManager] = None,
    ):
        super().__init__(config)
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.dm = data_manager or DataManager(name="bitget")

        # Strategy parameters with defaults
        default_params = {
            "adx_period": 14,
            "adx_threshold": 50,
            "di_period": 14,
            "williams_r_period": 14,
            "williams_r_oversold": -80,
            "williams_r_overbought": -20,
            "sma_period": 50,
            "lookback_period": 30,  # For highest high / lowest low
        }

        self.params = default_params.copy()
        if params:
            self.params.update(params)

        # Strategy metadata
        self.strategy_name = "DMI_ADX_Williams_R"
        self.version = "1.0"

    def setup_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators using IndicatorFactory"""
        factory = IndicatorFactory(df)

        df_with_indicators = (
            factory.add_adx(period=self.params["adx_period"])
            .add_minus_di(period=self.params["di_period"])
            .add_plus_di(period=self.params["di_period"])
            .add_williams_r(period=self.params["williams_r_period"])
            .add_sma(period=self.params["sma_period"])
            .add_atr(14)  # For position sizing
            .get_data()
        )

        # Calculate rolling highest high and lowest low for take profit
        df_with_indicators["highest_high"] = (
            df_with_indicators["high"]
            .rolling(window=self.params["lookback_period"])
            .max()
        )
        df_with_indicators["lowest_low"] = (
            df_with_indicators["low"]
            .rolling(window=self.params["lookback_period"])
            .min()
        )

        return df_with_indicators

    def generate_signal_conditions(self, df: pd.DataFrame) -> np.ndarray:
        """Generate vectorized trading signals"""

        # Get column names
        adx_col = f"ADX_{self.params['adx_period']}"
        di_plus_col = f"DI_plus_{self.params['di_period']}"
        di_minus_col = f"DI_minus_{self.params['di_period']}"
        williams_r_col = f"WilliamsR_{self.params['williams_r_period']}"
        sma_col = f"SMA_{self.params['sma_period']}"

        # === TREND STRENGTH CONDITION ===
        strong_trend = df[adx_col] > self.params["adx_threshold"]

        # === TREND DIRECTION CONDITIONS ===
        uptrend = df[di_plus_col] > df[di_minus_col]
        downtrend = df[di_minus_col] > df[di_plus_col]

        # === WILLIAMS %R CONDITIONS ===
        oversold = df[williams_r_col] < self.params["williams_r_oversold"]
        overbought = df[williams_r_col] > self.params["williams_r_overbought"]

        # === LONG CONDITIONS ===
        # Strong uptrend + oversold Williams %R (bounce opportunity)
        long_condition = strong_trend & uptrend & oversold

        # === SHORT CONDITIONS ===
        # Strong downtrend + overbought Williams %R (rejection opportunity)
        short_condition = strong_trend & downtrend & overbought

        # Use VectorizedSignals helper for safe signal generation
        return VectorizedSignals.safe_signals(long_condition, short_condition)

    def generate_exit_conditions(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate exit conditions for stop loss, take profit, and early exit"""

        # Get column names
        di_plus_col = f"DI_plus_{self.params['di_period']}"
        di_minus_col = f"DI_minus_{self.params['di_period']}"
        sma_col = f"SMA_{self.params['sma_period']}"

        # === STOP LOSS CONDITIONS ===
        # Price hits SMA 50 (trend reversal signal)
        long_stop_loss = df["close"] <= df[sma_col]
        short_stop_loss = df["close"] >= df[sma_col]

        # === TAKE PROFIT CONDITIONS ===
        # Long: Price reaches highest high of last 30 candles
        long_take_profit = df["close"] >= df["highest_high"].shift(1)
        # Short: Price reaches lowest low of last 30 candles
        short_take_profit = df["close"] <= df["lowest_low"].shift(1)

        # === EARLY EXIT CONDITIONS ===
        # Trend direction reverses (DI crossover)
        # Long exit: DI- crosses above DI+
        di_bearish_cross = (df[di_minus_col] > df[di_plus_col]) & (
            df[di_minus_col].shift(1) <= df[di_plus_col].shift(1)
        )
        # Short exit: DI+ crosses above DI-
        di_bullish_cross = (df[di_plus_col] > df[di_minus_col]) & (
            df[di_plus_col].shift(1) <= df[di_minus_col].shift(1)
        )

        return {
            "long_stop_loss": long_stop_loss,
            "short_stop_loss": short_stop_loss,
            "long_take_profit": long_take_profit,
            "short_take_profit": short_take_profit,
            "long_early_exit": di_bearish_cross,
            "short_early_exit": di_bullish_cross,
        }

    def configure_sizing(self):
        """Configure risk-based position sizing for scalping"""
        riskSizer = RiskBasedSizer(
            risk_percent=0.03,  # 3% risk per trade as specified
            max_position_pct=0.25,  # Max 25% position size
            default_stop_loss_pct=0.01,  # Default 1% stop loss
        )
        return riskSizer

    def generate_signals(self) -> pd.DataFrame:
        """Main signal generation method"""
        # 1. Load data
        df = self.dm.from_local(self.symbol, self.timeframe, self.start_date)

        # 2. Add indicators
        df = self.setup_indicators(df)

        # 3. Generate entry signals using vectorized approach
        df = self._apply_vectorized_signals(df)

        # 4. Generate exit conditions
        exit_conditions = self.generate_exit_conditions(df)

        # Add exit conditions to dataframe for analysis
        for condition_name, condition_array in exit_conditions.items():
            df[condition_name] = condition_array

        # 5. Add strategy metadata
        df["strategy_name"] = self.strategy_name
        df["strategy_version"] = self.version

        return df

    def validate_strategy(self) -> bool:
        """Quick validation of strategy setup"""
        try:
            df = self.generate_signals()

            # Basic checks
            total_signals = (df["signal"] != 0).sum()
            signal_rate = total_signals / len(df)
            long_signals = (df["signal"] == 1).sum()
            short_signals = (df["signal"] == -1).sum()

            print(f"✅ Strategy validation passed!")
            print(f"   Total signals: {total_signals}")
            print(f"   Long signals: {long_signals}")
            print(f"   Short signals: {short_signals}")
            print(f"   Signal rate: {signal_rate:.2%}")
            print(f"   Data shape: {df.shape}")

            # Check if we have required indicators
            required_indicators = [
                f"ADX_{self.params['adx_period']}",
                f"DI_plus_{self.params['di_period']}",
                f"DI_minus_{self.params['di_period']}",
                f"Williams_R_{self.params['williams_r_period']}",
                f"SMA_{self.params['sma_period']}",
            ]

            missing_indicators = [
                ind for ind in required_indicators if ind not in df.columns
            ]
            if missing_indicators:
                print(f"⚠️  Missing indicators: {missing_indicators}")
                return False

            return total_signals > 0

        except Exception as e:
            print(f"❌ Strategy validation failed: {e}")
            return False


# === OPTIMIZER FUNCTION ===


def create_optimizer_strategy_function(
    trial: optuna.trial.Trial, data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Optimizer function for the DMI ADX Williams R strategy"""

    # Parameter search space
    params = {
        "adx_period": trial.suggest_int("adx_period", 10, 20),
        "adx_threshold": trial.suggest_float("adx_threshold", 40, 60),
        "di_period": trial.suggest_int("di_period", 10, 20),
        "williams_r_period": trial.suggest_int("williams_r_period", 10, 20),
        "williams_r_oversold": trial.suggest_int("williams_r_oversold", -90, -70),
        "williams_r_overbought": trial.suggest_int("williams_r_overbought", -30, -10),
        "sma_period": trial.suggest_int("sma_period", 30, 70),
        "lookback_period": trial.suggest_int("lookback_period", 20, 40),
    }

    # Create strategy instance
    dummy_config = BacktestConfig()
    strategy = DMIADXRoStrategy(
        config=dummy_config,
        symbol="TRX/USDT:USDT",
        timeframe="15m",
        start_date="2024-01-01",
        params=params,
    )

    # Mock data loading
    def mock_from_local(symbol, timeframe, start_date):
        return data_dict["data"].copy()

    strategy.dm.from_local = mock_from_local

    return strategy.generate_signals()


# === MAIN FUNCTION ===


def main():
    """Main function for DMI ADX Williams R strategy"""

    print(f"\n{'='*60}")
    print(f"DMI ADX Williams %R Strategy - TREND FOLLOWING SCALPING")
    print(f"{'='*60}\n")

    # Strategy configuration
    STRATEGY_CONFIG = {
        "symbol": "TRX/USDT:USDT",
        "timeframe": "15m",
        "start_date": "2024-01-01",
    }

    # === Run Single Backtest ===
    print("\n💹 Running Single Backtest...")

    config = BacktestConfig(
        initial_balance=5000.0,
        leverage=5,  # 5x leverage as mentioned in risk management
        trading_mode=TradingMode.CROSS,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0006,
        # Disable trailing stop for this strategy
        enable_trailing_stop=False,
    )

    strategy = DMIADXRoStrategy(config=config, **STRATEGY_CONFIG)

    # Run backtest (sizing config applied automatically)
    analysis = strategy.run_single()


if __name__ == "__main__":
    main()
