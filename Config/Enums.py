# AlgoTrade/Config/enums.py
from enum import Enum


class TradingMode(Enum):
    SPOT = "Spot"
    ISOLATED = "Isolated"
    CROSS = "Cross"
