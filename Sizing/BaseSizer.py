from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import pandas as pd


class BaseSizer(ABC):
    """
    Abstract base class for all position sizing strategies.
    Defines the interface for calculating trade margin, stop-loss, and take-profit.
    """

    @abstractmethod
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

        Args:
            balance (float): The current account balance.
            equity (float): The current account equity.
            price (float): The current entry price of the asset.
            trade_history (pd.DataFrame): A DataFrame of past trades.
            atr_value (Optional[float]): The current ATR value, if available.

        Returns:
            Tuple[float, Optional[float], Optional[float]]: A tuple containing:
                - initial_margin (float)
                - stop_loss_price (Optional[float])
                - take_profit_price (Optional[float])
        """
        pass
