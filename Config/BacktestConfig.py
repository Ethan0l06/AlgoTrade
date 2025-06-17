from dataclasses import dataclass
from typing import Optional
from AlgoTrade.Config.Enums import TradingMode, PositionSizingMethod


@dataclass
class BacktestConfig:
    """
    Configuration class for the BacktestRunner.
    Holds all tunable parameters for a backtesting session.
    """

    # --- General Backtest Settings ---
    initial_balance: float = 1000.0
    trading_mode: TradingMode = TradingMode.ISOLATED
    leverage: int = 1
    open_fee_rate: float = 0.0002
    close_fee_rate: float = 0.0006

    # --- Position Sizing ---
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.ATR_BANDS

    # Parameters for 'PercentBalance'
    percent_balance_pct: float = 0.1

    # Parameters for 'FixedAmount'
    fixed_amount_size: float = 100.0

    # Parameters for 'AtrVolatility'
    atr_volatility_risk_pct: float = 0.01
    atr_volatility_period: int = 14
    atr_volatility_multiplier: float = 2.0

    # Parameters for 'KellyCriterion'
    kelly_criterion_lookback: int = 50
    kelly_criterion_fraction: float = 0.5

    # Parameters for 'AtrBands'
    atr_bands_risk_pct: float = 0.01
    atr_bands_period: int = 14
    atr_bands_multiplier: float = 2.0
    atr_bands_risk_reward_ratio: float = 1.5

    # Parameters for 'RiskParity'
    risk_parity_lookback: int = 20

    # --- Exit Logic ---
    exit_on_signal_0: bool = False
    exit_on_signal_opposite: bool = True
    allow_reverse_trade: bool = False

    # --- General Stop-Loss / Take-Profit (Percentage-based) ---
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

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
