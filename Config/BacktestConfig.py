# Enhanced Config/BacktestConfig.py
from dataclasses import dataclass, field
from typing import Optional, Tuple
from AlgoTrade.Config.Enums import TradingMode
from AlgoTrade.Sizing.BaseSizer import BaseSizer
from AlgoTrade.Sizing.PercentBalanceSizer import PercentBalanceSizer


@dataclass
class BacktestConfig:
    """
    Enhanced configuration class for the BacktestRunner.
    """

    # --- General Backtest Settings ---
    initial_balance: float = 100.0
    trading_mode: TradingMode = TradingMode.CROSS
    leverage: int = 1

    # --- Enhanced Fee Structure ---
    maker_fee_rate: float = 0.0002  # 0.01% for limit orders
    taker_fee_rate: float = 0.0006  # 0.04% for market orders
    # Legacy support
    open_fee_rate: float = field(init=False)
    close_fee_rate: float = field(init=False)

    # --- Market Microstructure ---
    base_slippage_bps: float = 1.0  # Base slippage in basis points
    enable_slippage: bool = True
    enable_market_impact: bool = False

    # --- Trading Hours ---
    enable_trading_hours: bool = False
    trading_start_hour: int = 0  # 24h format
    trading_end_hour: int = 23
    weekend_trading: bool = True  # For crypto
    reduced_weekend_liquidity: bool = True

    # --- Position Sizing ---
    sizing_strategy: BaseSizer = field(
        default_factory=lambda: PercentBalanceSizer(percent=0.2)
    )

    # --- Exit Logic ---
    exit_on_signal_0: bool = False
    exit_on_signal_opposite: bool = True
    allow_reverse_trade: bool = False

    # --- General Stop-Loss / Take-Profit ---
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    # --- Trailing Stop Logic ---
    enable_trailing_stop: bool = False
    breakeven_trigger_pct: Optional[float] = None
    breakeven_sl_pct: Optional[float] = 0.001
    midpoint_trigger_pct: Optional[float] = None
    midpoint_tp_extension_pct: Optional[float] = None
    midpoint_sl_adjustment_pct: Optional[float] = None

    def __post_init__(self):
        """Post-initialization checks and adjustments."""
        if self.trading_mode == TradingMode.SPOT:
            self.leverage = 1

        # Set legacy fee rates for backward compatibility
        self.open_fee_rate = self.taker_fee_rate  # Assume market orders for entry
        self.close_fee_rate = self.taker_fee_rate  # Assume market orders for exit

        # Validation
        if not 0 <= self.maker_fee_rate < 1:
            raise ValueError("maker_fee_rate must be between 0 and 1.")
        if not 0 <= self.taker_fee_rate < 1:
            raise ValueError("taker_fee_rate must be between 0 and 1.")
        if self.leverage < 1:
            raise ValueError("Leverage must be at least 1.")
        if self.base_slippage_bps < 0:
            raise ValueError("base_slippage_bps must be non-negative.")

    def get_trading_hours(self) -> Tuple[int, int]:
        """Get trading hours tuple"""
        return (self.trading_start_hour, self.trading_end_hour)
