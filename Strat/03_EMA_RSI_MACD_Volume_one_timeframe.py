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


class EmaRsiMacdVolumeStrategy(BaseStrategy):
    """
    Simplified 15-Minute Momentum Scalping Strategy using enhanced BaseStrategy.

    Strategy Rules:
    - Long: Price breaks above both EMAs + Volume spike + RSI 30-70 + MACD bullish
    - Short: Price breaks below both EMAs + Volume spike + RSI 30-70 + MACD bearish
    - Exit: 0.5% TP, 0.4% SL, 4-hour time stop
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
            "ema_fast": 9,
            "ema_slow": 21,
            "rsi_period": 14,
            "rsi_min": 30,
            "rsi_max": 70,
            "volume_ma_period": 20,
            "volume_spike_threshold": 1.5,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        }

        self.params = default_params.copy()
        if params:
            self.params.update(params)

        # Strategy metadata
        self.strategy_name = "EMA_RSI_MACD_Volume_Simplified"
        self.version = "2.0"

    def setup_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators using IndicatorFactory"""
        factory = IndicatorFactory(df)

        df_with_indicators = (
            factory.add_ema(period=self.params["ema_fast"])
            .add_ema(period=self.params["ema_slow"])
            .add_rsi(period=self.params["rsi_period"])
            .add_macd(
                fastperiod=self.params["macd_fast"],
                slowperiod=self.params["macd_slow"],
                signalperiod=self.params["macd_signal"],
            )
            .add_atr(14)  # For position sizing
            .add_sma(period=self.params["volume_ma_period"], column="volume")
            .get_data()
        )

        # Calculate volume ratio for spike detection
        volume_ma_col = f"SMA_{self.params['volume_ma_period']}"
        df_with_indicators["volume_ratio"] = (
            df_with_indicators["volume"] / df_with_indicators[volume_ma_col]
        )

        return df_with_indicators

    def generate_signal_conditions(self, df: pd.DataFrame) -> np.ndarray:
        """Generate vectorized trading signals"""

        # Get column names
        ema_fast_col = f"EMA_{self.params['ema_fast']}"
        ema_slow_col = f"EMA_{self.params['ema_slow']}"
        rsi_col = f"RSI_{self.params['rsi_period']}"

        # === LONG CONDITIONS ===
        # Price position
        price_above_fast = df["close"] > df[ema_fast_col]
        price_above_slow = df["close"] > df[ema_slow_col]

        # EMA alignment
        ema_bullish = df[ema_fast_col] > df[ema_slow_col]

        # Price breakout (wasn't above both EMAs in previous candle)
        price_breakout_long = (
            price_above_fast
            & price_above_slow
            & (
                ~VectorizedSignals.safe_condition(price_above_fast.shift(1))
                | ~VectorizedSignals.safe_condition(price_above_slow.shift(1))
            )
        )

        # Volume spike
        volume_spike = df["volume_ratio"] >= self.params["volume_spike_threshold"]

        # RSI filter (avoid extremes)
        rsi_acceptable = (df[rsi_col] >= self.params["rsi_min"]) & (
            df[rsi_col] <= self.params["rsi_max"]
        )
        rsi_rising = df[rsi_col] > df[rsi_col].shift(1)

        # MACD confirmation
        macd_bullish = df["MACD"] > df["MACD_signal"]
        macd_hist_positive = df["MACD_hist"] > 0
        macd_hist_rising = df["MACD_hist"] > df["MACD_hist"].shift(1)

        # === SHORT CONDITIONS ===
        # Price position
        price_below_fast = df["close"] < df[ema_fast_col]
        price_below_slow = df["close"] < df[ema_slow_col]

        # EMA alignment
        ema_bearish = df[ema_fast_col] < df[ema_slow_col]

        # Price breakout (wasn't below both EMAs in previous candle)
        price_breakout_short = (
            price_below_fast
            & price_below_slow
            & (
                ~VectorizedSignals.safe_condition(price_below_fast.shift(1))
                | ~VectorizedSignals.safe_condition(price_below_slow.shift(1))
            )
        )

        # RSI filter
        rsi_falling = df[rsi_col] < df[rsi_col].shift(1)

        # MACD confirmation
        macd_bearish = df["MACD"] < df["MACD_signal"]
        macd_hist_negative = df["MACD_hist"] < 0
        macd_hist_falling = df["MACD_hist"] < df["MACD_hist"].shift(1)

        # === COMBINE CONDITIONS ===
        long_condition = (
            price_breakout_long
            & ema_bullish
            & volume_spike
            & rsi_acceptable
            & rsi_rising
            & macd_bullish
            & macd_hist_positive
            & macd_hist_rising
        )

        short_condition = (
            price_breakout_short
            & ema_bearish
            & volume_spike
            & rsi_acceptable
            & rsi_falling
            & macd_bearish
            & macd_hist_negative
            & macd_hist_falling
        )

        # Use VectorizedSignals helper for safe signal generation
        return VectorizedSignals.safe_signals(long_condition, short_condition)

    def configure_sizing(self):
        """Configure risk-based position sizing for momentum scalping"""
        riskSizer = RiskBasedSizer(
            risk_percent=0.01,  # 1% risk per trade
            max_position_pct=0.2,  # Max 20% position size
            default_stop_loss_pct=0.004,  # 0.4% stop loss
        )
        percentSizer = PercentBalanceSizer(percent=0.15)
        return percentSizer

    def generate_signals(self) -> pd.DataFrame:
        """Main signal generation method"""
        # 1. Load data
        df = self.dm.from_local(self.symbol, self.timeframe, self.start_date)

        # 2. Add indicators
        df = self.setup_indicators(df)

        # 3. Generate signals using vectorized approach
        df = self._apply_vectorized_signals(df)

        # 4. Add strategy metadata
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

            print(f"✅ Strategy validation passed!")
            print(f"   Total signals: {total_signals}")
            print(f"   Signal rate: {signal_rate:.2%}")
            print(f"   Data shape: {df.shape}")

            return total_signals > 0

        except Exception as e:
            print(f"❌ Strategy validation failed: {e}")
            return False


# === OPTIMIZER FUNCTION ===


def create_optimizer_strategy_function(
    trial: optuna.trial.Trial, data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Optimizer function for the simplified strategy"""

    # Parameter search space
    params = {
        "ema_fast": trial.suggest_int("ema_fast", 5, 15),
        "ema_slow": trial.suggest_int("ema_slow", 16, 35),
        "rsi_period": trial.suggest_int("rsi_period", 10, 21),
        "rsi_min": trial.suggest_int("rsi_min", 25, 35),
        "rsi_max": trial.suggest_int("rsi_max", 65, 75),
        "volume_spike_threshold": trial.suggest_float(
            "volume_spike_threshold", 1.2, 2.5
        ),
        "volume_ma_period": trial.suggest_int("volume_ma_period", 10, 30),
    }

    # Create strategy instance
    dummy_config = BacktestConfig()
    strategy = EmaRsiMacdVolumeStrategy(
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
    """Simplified main function using enhanced framework"""

    print(f"\n{'='*60}")
    print(f"EMA RSI MACD Volume Strategy - SIMPLIFIED VERSION")
    print(f"{'='*60}\n")

    # Strategy configuration
    STRATEGY_CONFIG = {
        "symbol": "TRX/USDT:USDT",
        "timeframe": "15m",
        "start_date": "2024-01-01",
    }

    # === Test Strategy Validation ===
    print("🔍 Testing Strategy Validation...")
    test_config = BacktestConfig()
    test_strategy = EmaRsiMacdVolumeStrategy(config=test_config, **STRATEGY_CONFIG)

    if not test_strategy.validate_strategy():
        print("❌ Strategy validation failed!")
        return

    # === Run Single Backtest ===
    print("\n💹 Running Single Backtest...")

    config = BacktestConfig(
        initial_balance=5000.0,
        leverage=15,
        trading_mode=TradingMode.CROSS,
        enable_slippage=True,
        base_slippage_bps=1.5,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0006,
        # Enable trailing stop
        enable_trailing_stop=False,
        breakeven_trigger_pct=0.01,
        breakeven_sl_pct=0.001,
        midpoint_trigger_pct=0.5,
        midpoint_tp_extension_pct=0.3,
        midpoint_sl_adjustment_pct=0.2,
    )

    strategy = EmaRsiMacdVolumeStrategy(config=config, **STRATEGY_CONFIG)

    # Run backtest (sizing config applied automatically)
    analysis = strategy.run_single()
    
if __name__ == "__main__":
    main()
