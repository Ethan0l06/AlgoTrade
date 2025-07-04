from dataclasses import dataclass, field
from typing import Optional
from AlgoTrade.Config.Enums import TradingMode
from AlgoTrade.Sizing.BaseSizer import BaseSizer
from AlgoTrade.Sizing.PercentBalanceSizer import PercentBalanceSizer


@dataclass
class BacktestConfig:
    """
    Configuration class for the BacktestRunner.
    Holds all tunable parameters for a backtesting session.
    """

    # --- General Backtest Settings ---
    initial_balance: float = 100.0
    trading_mode: TradingMode = TradingMode.CROSS
    leverage: int = 1
    open_fee_rate: float = 0.0002
    close_fee_rate: float = 0.0006

    # --- Position Sizing ---
    sizing_strategy: BaseSizer = field(
        default_factory=lambda: PercentBalanceSizer(percent=0.2)
    )

    # --- Exit Logic ---
    exit_on_signal_0: bool = False
    exit_on_signal_opposite: bool = True
    allow_reverse_trade: bool = False

    # --- General Stop-Loss / Take-Profit (Percentage-based) ---
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    # --- ▼▼▼ NEW TRAILING STOP LOGIC ▼▼▼ ---
    enable_trailing_stop: bool = False
    breakeven_trigger_pct: Optional[float] = None  # e.g., 0.005 for 0.5% profit
    breakeven_sl_pct: Optional[float] = 0.001  # Move SL to 0.1% profit
    midpoint_trigger_pct: Optional[float] = None  # e.g., 0.5 for 50% to TP
    midpoint_tp_extension_pct: Optional[float] = None  # e.g., 0.5 to extend TP by 50%
    midpoint_sl_adjustment_pct: Optional[float] = (
        None  # e.g., 0.3 to lock in 30% of profit
    )

    def __post_init__(self):
        """Post-initialization checks and adjustments."""
        if self.trading_mode == TradingMode.SPOT:
            self.leverage = 1

        if not 0 <= self.open_fee_rate < 1:
            raise ValueError("open_fee_rate must be between 0 and 1.")
        if not 0 <= self.close_fee_rate < 1:
            raise ValueError("close_fee_rate must be between 0 and 1.")
        if self.leverage < 1:
            raise ValueError("Leverage must be at least 1.")
