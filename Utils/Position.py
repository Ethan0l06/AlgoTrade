from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

# This is a common practice to avoid circular imports while still allowing for type hinting.
# It tells the type checker that a 'Position' class will be available at runtime.
if TYPE_CHECKING:

    class Position:
        pass


class PositionBehavior(ABC):
    """
    An abstract base class that defines the unique behaviors for long and short positions,
    such as how to calculate P&L and liquidation prices.
    """

    @abstractmethod
    def calculate_pnl(self, position: "Position", close_price: float) -> float:
        pass

    @abstractmethod
    def calculate_liquidation_price(
        self, position: "Position", account_balance_for_cross: Optional[float] = None
    ) -> Optional[float]:
        pass

    @abstractmethod
    def check_for_sl(
        self, position: "Position", low_price: float, high_price: float
    ) -> bool:
        pass

    @abstractmethod
    def check_for_tp(
        self, position: "Position", low_price: float, high_price: float
    ) -> bool:
        pass

    @abstractmethod
    def check_for_liquidation(
        self, position: "Position", low_price: float, high_price: float
    ) -> bool:
        pass


class LongPositionBehavior(PositionBehavior):
    """Implements the behavior for a long position."""

    def calculate_pnl(self, position: "Position", close_price: float) -> float:
        return position.amount * (close_price - position.open_price)

    def calculate_liquidation_price(
        self, position: "Position", account_balance_for_cross: Optional[float] = None
    ) -> Optional[float]:
        if position.mode == "cross":
            if account_balance_for_cross is not None and position.amount > 0:
                # Note: This is a simplified formula. A precise exchange formula would
                # also involve maintenance margin rates.
                return position.open_price - (
                    account_balance_for_cross / position.amount
                )
            return None
        else:  # Isolated mode
            if not position.leverage or position.leverage <= 0:
                return None
            return position.open_price * (1 - 1 / position.leverage)

    def check_for_sl(
        self, position: "Position", low_price: float, high_price: float
    ) -> bool:
        return (
            low_price <= position.sl_price if position.sl_price is not None else False
        )

    def check_for_tp(
        self, position: "Position", low_price: float, high_price: float
    ) -> bool:
        return (
            high_price >= position.tp_price if position.tp_price is not None else False
        )

    def check_for_liquidation(
        self, position: "Position", low_price: float, high_price: float
    ) -> bool:
        return (
            low_price <= position.liquidation_price
            if position.liquidation_price is not None
            else False
        )


class ShortPositionBehavior(PositionBehavior):
    """Implements the behavior for a short position."""

    def calculate_pnl(self, position: "Position", close_price: float) -> float:
        return position.amount * (position.open_price - close_price)

    def calculate_liquidation_price(
        self, position: "Position", account_balance_for_cross: Optional[float] = None
    ) -> Optional[float]:
        if position.mode == "cross":
            if account_balance_for_cross is not None and position.amount > 0:
                return position.open_price + (
                    account_balance_for_cross / position.amount
                )
            return None
        else:  # Isolated mode
            if not position.leverage or position.leverage <= 0:
                return None
            return position.open_price * (1 + 1 / position.leverage)

    def check_for_sl(
        self, position: "Position", low_price: float, high_price: float
    ) -> bool:
        return (
            high_price >= position.sl_price if position.sl_price is not None else False
        )

    def check_for_tp(
        self, position: "Position", low_price: float, high_price: float
    ) -> bool:
        return (
            low_price <= position.tp_price if position.tp_price is not None else False
        )

    def check_for_liquidation(
        self, position: "Position", low_price: float, high_price: float
    ) -> bool:
        return (
            high_price >= position.liquidation_price
            if position.liquidation_price is not None
            else False
        )


class Position:
    """
    A class to manage the state and lifecycle of a single trading position.
    It handles opening, closing, and tracking all financial metrics.
    """

    def __init__(
        self,
        leverage: int = 1,
        open_fee_rate: float = 0.0,
        close_fee_rate: float = 0.0,
        mode: str = "cross",  # "isolated" or "cross"
    ) -> None:
        # Configuration
        self.mode = mode
        self.leverage = leverage
        self.open_fee_rate = open_fee_rate
        self.close_fee_rate = close_fee_rate

        # State
        self.side: Optional[str] = None  # 'long' or 'short'
        self.is_open: bool = False
        self.behavior: Optional[PositionBehavior] = None

        # Opening Metrics
        self.open_time: Optional[datetime] = None
        self.open_price: Optional[float] = None
        self.initial_margin: Optional[float] = None
        self.open_notional_value: Optional[float] = None
        self.amount: Optional[float] = None
        self.open_fee: Optional[float] = None
        self.open_reason: Optional[str] = None

        # Risk Management
        self.sl_price: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.liquidation_price: Optional[float] = None

        # Closing Metrics
        self.close_time: Optional[datetime] = None
        self.close_price: Optional[float] = None
        self.close_notional_value: Optional[float] = None
        self.close_fee: Optional[float] = None
        self.pnl_gross: Optional[float] = None
        self.net_pnl: Optional[float] = None
        self.net_pnl_pct: Optional[float] = None
        self.close_reason: Optional[str] = None

    def open(
        self,
        time: datetime,
        side: str,
        open_price: float,
        initial_margin: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        open_reason: str = "Signal",
        account_balance_for_cross: Optional[float] = None,
    ):
        """Opens a new position."""
        if self.is_open:
            # Handle error: Position is already open
            return

        self.side = side
        self.behavior = (
            LongPositionBehavior() if side == "long" else ShortPositionBehavior()
        )

        self.open_time = time
        self.open_price = open_price
        self.initial_margin = initial_margin
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.open_reason = open_reason

        # Calculate core metrics
        self.open_notional_value = self.initial_margin * self.leverage
        self.open_fee = self.open_notional_value * self.open_fee_rate
        effective_notional = self.open_notional_value - self.open_fee
        self.amount = effective_notional / self.open_price if self.open_price > 0 else 0

        self.liquidation_price = self.behavior.calculate_liquidation_price(
            self, account_balance_for_cross
        )
        self.is_open = True

    def close(self, time: datetime, close_price: float, reason: str):
        """Closes the current position."""
        if not self.is_open:
            # Handle error: Position is already closed
            return

        self.close_time = time
        self.close_price = close_price
        self.close_reason = reason

        self.pnl_gross = self.behavior.calculate_pnl(self, self.close_price)
        self.close_notional_value = self.amount * self.close_price
        self.close_fee = self.close_notional_value * self.close_fee_rate

        self.net_pnl = self.pnl_gross - self.open_fee - self.close_fee

        if self.initial_margin > 0:
            self.net_pnl_pct = (self.net_pnl / self.initial_margin) * 100
        else:
            self.net_pnl_pct = 0.0

        self.is_open = False

    def get_trade_info(self) -> Dict[str, Any]:
        """Returns a dictionary summary of the closed trade."""
        return {
            "side": self.side,
            "mode": self.mode,
            "leverage": self.leverage,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "open_reason": self.open_reason,
            "close_reason": self.close_reason,
            "open_price": self.open_price,
            "close_price": self.close_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "liquidation_price": self.liquidation_price,
            "initial_margin": self.initial_margin,
            "open_notional_value": self.open_notional_value,
            "close_notional_value": self.close_notional_value,
            "amount": self.amount,
            "open_fee": self.open_fee,
            "close_fee": self.close_fee,
            "net_pnl": self.net_pnl,
            "net_pnl_pct": self.net_pnl_pct,
        }
