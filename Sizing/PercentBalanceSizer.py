from AlgoTrade.Sizing.BaseSizer import BaseSizer
from typing import Optional, Tuple
import pandas as pd


class PercentBalanceSizer(BaseSizer):
    """Uses a fixed percentage of the total account balance for each trade's margin.\n
    percent : 0.1 = 10% of balance for margin."""

    def __init__(self, percent: float):
        if not 0 < percent <= 1:
            raise ValueError("Sizing percent must be between 0 and 1.")
        self.percent = percent

    def calculate_trade_parameters(
        self,
        balance: float,
        equity: float,
        price: float,
        side: str,
        trade_history: pd.DataFrame,
        atr_value: Optional[float] = None,
    ) -> Tuple[float, Optional[float], Optional[float]]:

        # 'side' is accepted but not used.
        margin = balance * self.percent
        return margin, None, None
