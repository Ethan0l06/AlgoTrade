from AlgoTrade.Sizing.BaseSizer import BaseSizer
from typing import Optional, Tuple
import pandas as pd


class AtrBandsSizer(BaseSizer):
    """
    Calculates size, SL, and TP based on ATR bands.
    Position size is determined by risking a fixed percentage of equity.
    """

    def __init__(
        self,
        risk_pct: float,
        atr_multiplier: float,
        risk_reward_ratio: float,
        leverage: int,
    ):
        self.risk_pct = risk_pct
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.leverage = leverage

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
        Calculates the initial margin, stop-loss price, and take-profit price for a trade.
        """

        # Bug fix: Added a check for NaN and non-positive ATR values.
        if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
            return 0, None, None  # Cannot size without a valid ATR

        risk_in_dollars = equity * self.risk_pct
        stop_loss_distance = atr_value * self.atr_multiplier

        if stop_loss_distance <= 0:
            return 0, None, None

        notional_value = (risk_in_dollars / stop_loss_distance) * price
        margin = notional_value / self.leverage

        # Calculate SL and TP based on side
        if side == "long":
            sl_price = price - stop_loss_distance
            tp_price = price + (stop_loss_distance * self.risk_reward_ratio)
        else:  # short
            sl_price = price + stop_loss_distance
            tp_price = price - (stop_loss_distance * self.risk_reward_ratio)

        return margin, sl_price, tp_price
