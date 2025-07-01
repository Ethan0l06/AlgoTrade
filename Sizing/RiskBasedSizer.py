from AlgoTrade.Sizing.BaseSizer import BaseSizer
from typing import Optional, Tuple, Dict
import pandas as pd


class RiskBasedSizer(BaseSizer):
    """
    Risk-based position sizing strategy for momentum scalping.
    Calculates position size based on fixed risk percentage and stop loss distance.

    This implements the formula:
    position_size = risk_amount / (entry_price * stop_loss_percent)
    """

    def __init__(
        self,
        risk_percent: float = 0.01,
        max_position_pct: float = 0.15,
        default_stop_loss_pct: float = 0.004,  # 0.4% default stop loss
        min_position_value: float = 10.0,  # Minimum position value
    ):
        """
        Initialize the RiskBasedSizer.

        Args:
            risk_percent: Percentage of equity to risk per trade (0.01 = 1%)
            max_position_pct: Maximum percentage of equity per position (0.15 = 15%)
            default_stop_loss_pct: Default stop loss percentage if not provided
            min_position_value: Minimum position value to avoid tiny trades
        """
        if not 0.001 <= risk_percent <= 0.1:
            raise ValueError("Risk percent must be between 0.1% and 10%")
        if not 0.05 <= max_position_pct <= 1.0:
            raise ValueError("Max position percent must be between 5% and 100%")
        if not 0.001 <= default_stop_loss_pct <= 0.1:
            raise ValueError("Default stop loss percent must be between 0.1% and 10%")

        self.risk_percent = risk_percent
        self.max_position_pct = max_position_pct
        self.default_stop_loss_pct = default_stop_loss_pct
        self.min_position_value = min_position_value

    def calculate_position_size(
        self,
        balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss_percent: float,
    ) -> float:
        """
        Calculate position size using the risk-based formula.

        Args:
            balance: Current account balance
            risk_percent: Risk percentage (0.01 = 1%)
            entry_price: Entry price for the trade
            stop_loss_percent: Stop loss percentage (0.004 = 0.4%)

        Returns:
            Position size in base currency units
        """
        if balance <= 0 or entry_price <= 0 or stop_loss_percent <= 0:
            return 0.0

        risk_amount = balance * risk_percent
        position_size = risk_amount / (entry_price * stop_loss_percent)

        return position_size

    def calculate_trade_parameters(
        self,
        balance: float,
        equity: float,
        price: float,
        side: str,
        trade_history: pd.DataFrame,
        atr_value: Optional[float] = None,
    ) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Calculate margin, stop loss, and take profit for momentum scalping strategy.

        Uses risk-based position sizing with multiple take profit levels:
        - TP1: 0.5% (partial exit)
        - TP2: 1.0% (partial exit)
        - TP3: 1.5% (final exit)
        - SL: 0.4% (risk management)
        """

        # Use equity for risk calculation (includes unrealized P&L)
        account_value = max(equity, balance)

        if account_value <= 0 or price <= 0:
            return 0.0, None, None

        # Calculate risk amount
        risk_amount = account_value * self.risk_percent

        # Use default stop loss percentage for momentum scalping
        stop_loss_percent = self.default_stop_loss_pct

        # Calculate position value based on risk
        position_value = risk_amount / stop_loss_percent

        # Apply maximum position limit
        max_position_value = account_value * self.max_position_pct
        position_value = min(position_value, max_position_value)

        # Ensure minimum position size
        if position_value < self.min_position_value:
            return 0.0, None, None

        # Calculate margin (position value divided by leverage, applied in config)
        margin = position_value

        # Calculate stop loss and take profit prices
        if side == "long":
            # Long position
            sl_price = price * (1 - stop_loss_percent)
            tp_price = price * (1 + 0.005)  # Primary TP at 0.5%

            # Additional TP levels for reference (could be used in advanced exit logic)
            tp_price_2 = price * (1 + 0.01)  # 1.0%
            tp_price_3 = price * (1 + 0.015)  # 1.5%

        else:  # short
            # Short position
            sl_price = price * (1 + stop_loss_percent)
            tp_price = price * (1 - 0.005)  # Primary TP at 0.5%

            # Additional TP levels for reference
            tp_price_2 = price * (1 - 0.01)  # 1.0%
            tp_price_3 = price * (1 - 0.015)  # 1.5%

        return margin, sl_price, tp_price

    def get_exit_levels(self, entry_price: float, side: str) -> Dict[str, float]:
        """
        Get all exit levels for the momentum scalping strategy.

        Args:
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')

        Returns:
            Dictionary with stop loss and take profit levels
        """
        if side == "long":
            return {
                "stop_loss": entry_price * (1 - self.default_stop_loss_pct),
                "take_profit_1": entry_price * (1 + 0.005),  # 0.5%
                "take_profit_2": entry_price * (1 + 0.01),  # 1.0%
                "take_profit_3": entry_price * (1 + 0.015),  # 1.5%
            }
        else:  # short
            return {
                "stop_loss": entry_price * (1 + self.default_stop_loss_pct),
                "take_profit_1": entry_price * (1 - 0.005),  # 0.5%
                "take_profit_2": entry_price * (1 - 0.01),  # 1.0%
                "take_profit_3": entry_price * (1 - 0.015),  # 1.5%
            }

    def calculate_risk_metrics(
        self, balance: float, position_value: float, entry_price: float
    ) -> Dict[str, float]:
        """
        Calculate risk metrics for the position.

        Args:
            balance: Current account balance
            position_value: Total position value
            entry_price: Entry price

        Returns:
            Dictionary with risk metrics
        """
        risk_amount = balance * self.risk_percent
        position_size = position_value / entry_price

        # Risk per unit
        risk_per_unit = entry_price * self.default_stop_loss_pct

        # Maximum loss
        max_loss = position_size * risk_per_unit

        # Position as percentage of balance
        position_percent = (position_value / balance) * 100

        return {
            "risk_amount": risk_amount,
            "position_size": position_size,
            "position_value": position_value,
            "position_percent": position_percent,
            "max_loss": max_loss,
            "risk_per_unit": risk_per_unit,
            "risk_reward_ratio": 0.005 / self.default_stop_loss_pct,  # TP1 vs SL
        }
