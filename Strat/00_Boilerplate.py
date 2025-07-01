from AlgoTrade.Strat.Class.BaseStrategy import BaseStrategy
from AlgoTrade.Factories.IndicatorFactory import IndicatorFactory
from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Sizing.RiskBasedSizer import RiskBasedSizer
from AlgoTrade.Utils.VectorizedSignals import VectorizedSignals

class MySimpleStrategy(BaseStrategy):
    def __init__(self, config, symbol, timeframe, start_date):
        super().__init__(config)
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.dm = DataManager(name="bitget")

    def setup_indicators(self, df):
        factory = IndicatorFactory(df)
        return factory.add_sma(20).add_rsi(14).get_data()

    def generate_signal_conditions(self, df):
        long_cond = (df["close"] > df["SMA_20"]) & (df["RSI_14"] < 70)
        short_cond = (df["close"] < df["SMA_20"]) & (df["RSI_14"] > 30)
        return VectorizedSignals.safe_signals(long_cond, short_cond)

    def configure_sizing(self):
        return RiskBasedSizer(risk_percent=0.01)

    def generate_signals(self):
        # Load data
        df = self.dm.from_local(self.symbol, self.timeframe, self.start_date)
        # Add indicators
        df = self.setup_indicators(df)
        # Apply signals
        df = self._apply_vectorized_signals(df)
        return df
