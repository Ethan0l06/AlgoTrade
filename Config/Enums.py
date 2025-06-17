# AlgoTrade/Config/enums.py
from enum import Enum


class TradingMode(Enum):
    SPOT = "Spot"
    ISOLATED = "Isolated"
    CROSS = "Cross"


class PositionSizingMethod(Enum):
    PERCENT_BALANCE = "PercentBalance"
    FIXED_AMOUNT = "FixedAmount"
    ATR_VOLATILITY = "AtrVolatility"
    KELLY_CRITERION = "KellyCriterion"
    ATR_BANDS = "AtrBands"
