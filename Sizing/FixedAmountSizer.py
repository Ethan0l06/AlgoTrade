from AlgoTrade.Sizing.BaseSizer import BaseSizer
from typing import Optional, Tuple
import pandas as pd


class FixedAmountSizer(BaseSizer):
    """Uses a fixed, absolute currency amount for each trade's margin."""

    def __init__(self, amount: float):
        if amount <= 0:
            raise ValueError("Sizing amount must be positive.")
        self.amount = amount

    def calculate_trade_parameters(
        self,
        balance: float,
        equity: float,
        price: float,
        side: str,
        trade_history: pd.DataFrame,
        atr_value: Optional[float] = None,
    ) -> Tuple[float, Optional[float], Optional[float]]:

        # 'side' is accepted to match the base class, but not used here.
        margin = self.amount
        return margin, None, None
