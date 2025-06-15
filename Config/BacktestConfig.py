from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class BacktestConfig:
    """
    Configuration class for the BacktestRunner.
    Holds all tunable parameters for a backtesting session.
    """

    # --- General Backtest Settings ---
    initial_balance: float = 1000.0
    trading_mode: Literal["Spot", "Isolated", "Cross"] = "Isolated"
    leverage: int = 1  # Default to 1 for Spot, will be overridden for Futures
    open_fee_rate: float = 0.0002
    close_fee_rate: float = 0.0006

    # --- Position Sizing ---
    # Choose one: 'PercentBalance', 'FixedAmount', 'AtrVolatility', 'KellyCriterion', 'AtrBands'
    position_sizing_method: Literal[
        "PercentBalance", "FixedAmount", "AtrVolatility", "KellyCriterion", "AtrBands"
    ] = "AtrBands"

    # Parameters for 'PercentBalance'
    percent_balance_pct: float = 0.1  # e.g., 10% of balance

    # Parameters for 'FixedAmount'
    fixed_amount_size: float = 100.0  # e.g., $100

    # Parameters for 'AtrVolatility'
    atr_volatility_risk_pct: float = 0.01  # e.g., Risk 1% of equity per trade
    atr_volatility_period: int = 14
    atr_volatility_multiplier: float = 2.0  # SL is atr_value * atr_multiplier

    # Parameters for 'KellyCriterion'
    kelly_criterion_lookback: int = 50
    kelly_criterion_fraction: float = (
        0.5  # e.g., use 50% of the calculated Kelly fraction
    )

    # Parameters for 'AtrBands'
    atr_bands_risk_pct: float = 0.01  # e.g., Risk 1% of equity per trade
    atr_bands_period: int = 14
    atr_bands_multiplier: float = 2.0  # SL is atr_value * atr_multiplier
    atr_bands_risk_reward_ratio: float = 1.5  # e.g., 1.5:1 reward to risk

    exit_on_signal_0: bool = False
    """
    If True, an open position will be closed when the signal changes to 0.
    If False, a signal of 0 is ignored and the position remains open.
    Example: If long and signal becomes 0, the trade is closed.
    """

    exit_on_signal_opposite: bool = True
    """
    If True, an open position will be closed when the signal flips to the opposite direction.
    This is evaluated before 'exit_on_signal_0'.
    Example: If long (signal 1) and signal becomes -1, the trade is closed.
    """

    allow_reverse_trade: bool = False
    """
    If True, the system will perform a "reverse" trade, which is a close-and-reopen in the
    opposite direction. This has the highest precedence.
    If this is True, 'exit_on_signal_opposite' is effectively ignored because reversing
    is a more specific action than just closing.
    Example: If long and signal becomes -1, the system closes the long and immediately
    opens a short position at the same candle.
    """

    # --- General Stop-Loss / Take-Profit (Percentage-based) ---
    # These are used if a sizing method that sets its own SL/TP isn't used.
    stop_loss_pct: Optional[float] = None  # e.g., 0.05 for 5% SL
    take_profit_pct: Optional[float] = None  # e.g., 0.10 for 10% TP

    def __post_init__(self):
        """Post-initialization checks and adjustments."""
        if self.trading_mode == "Spot":
            self.leverage = 1  # Spot trading always has leverage 1

        # Basic validation (can be expanded)
        if not 0 <= self.open_fee_rate < 1:
            raise ValueError("open_fee_rate must be between 0 and 1.")
        if not 0 <= self.close_fee_rate < 1:
            raise ValueError("close_fee_rate must be between 0 and 1.")
        if self.leverage < 1:
            raise ValueError("Leverage must be at least 1.")
