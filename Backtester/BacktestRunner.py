# /TradingBacktester/Backtester/BacktestRunner.py

import pandas as pd
import talib
from typing import Dict, Any, List, Optional, Tuple
import datetime
import numpy as np

from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Config.Enums import TradingMode
from AlgoTrade.Utils.Position import Position
from AlgoTrade.Analysis.BacktestAnalysis import BacktestAnalysis
from AlgoTrade.Sizing.AtrBandsSizer import AtrBandsSizer


class BacktestRunner:
    def __init__(self, config: BacktestConfig, data: pd.DataFrame):
        self.config = config
        self.data = data.copy()
        self.balance: float = 0.0
        self.equity: float = 0.0
        self.position: Optional[Position] = None
        self.trades_info: List[Dict[str, Any]] = []
        self.equity_record: List[Dict[str, Any]] = []
        self.is_running: bool = False

        # New: Track volume for market impact
        self.volume_history: List[float] = []

        self._prepare_data()

    def _prepare_data(self):
        """Prepare data with additional columns for realistic execution"""
        # Prepare ATR data if any ATR-based sizer might be used
        if "atr" not in self.data.columns:
            self.data["atr"] = talib.ATR(
                self.data["high"], self.data["low"], self.data["close"], timeperiod=14
            )

        if "signal" not in self.data.columns:
            raise ValueError("Input data must contain a 'signal' column.")

        # Add next candle's open for realistic execution
        self.data["next_open"] = self.data["open"].shift(-1)

        # Calculate average volume for market impact
        if "volume" in self.data.columns:
            self.data["avg_volume_20"] = self.data["volume"].rolling(20).mean()
        else:
            self.data["avg_volume_20"] = 1.0  # Default if no volume data

    def _is_market_open(self, timestamp: datetime.datetime) -> bool:
        """Check if market is open based on config"""
        if not self.config.enable_trading_hours:
            return True

        hour = timestamp.hour
        weekday = timestamp.weekday()

        # Check trading hours
        start_hour, end_hour = self.config.get_trading_hours()
        if not (start_hour <= hour <= end_hour):
            return False

        # Check weekend trading
        if weekday >= 5 and not self.config.weekend_trading:  # Saturday/Sunday
            return False

        return True

    def _get_market_condition_multiplier(self, row: pd.Series) -> float:
        """Get market condition multiplier for slippage/fees"""
        multiplier = 1.0

        # Weekend effect for crypto
        if self.config.reduced_weekend_liquidity:
            timestamp = row.name if hasattr(row, "name") else datetime.datetime.now()
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            if timestamp.weekday() >= 5:  # Weekend
                multiplier *= 1.5

        # Volume-based liquidity
        if "volume" in row.index and "avg_volume_20" in row.index:
            volume_ratio = row["volume"] / max(row["avg_volume_20"], 1)
            if volume_ratio < 0.5:  # Low volume
                multiplier *= 2.0
            elif volume_ratio > 2.0:  # High volume
                multiplier *= 0.7

        return multiplier

    def _calculate_slippage(
        self, price: float, row: pd.Series, gap: bool = False
    ) -> float:
        """Calculate realistic slippage"""
        if not self.config.enable_slippage:
            return 0.0

        base_slippage = price * (self.config.base_slippage_bps / 10000)
        market_multiplier = self._get_market_condition_multiplier(row)

        slippage = base_slippage * market_multiplier

        if gap:
            slippage *= 3  # Higher slippage on gaps

        return slippage

    def _calculate_market_impact(self, notional_value: float, row: pd.Series) -> float:
        """Calculate market impact of the trade"""
        if not self.config.enable_market_impact:
            return 0.0

        avg_volume = row.get("avg_volume_20", 1.0)
        if avg_volume <= 0:
            return 0.0

        # Estimate participation rate (simplified)
        daily_volume_usd = avg_volume * row.get("close", 1.0) * 24  # Rough estimate
        participation_rate = notional_value / max(
            daily_volume_usd * 0.01, 1.0
        )  # 1% of daily volume

        if participation_rate > 0.1:  # More than 10% participation
            return notional_value * 0.0005 * participation_rate  # Linear impact

        return 0.0

    def _calculate_fees(self, notional_value: float, is_maker: bool = False) -> float:
        """Calculate trading fees based on order type"""
        if is_maker:
            fee_rate = self.config.maker_fee_rate
        else:
            fee_rate = self.config.taker_fee_rate

        return notional_value * fee_rate

    def _create_new_position_object(self):
        self.position = Position(
            leverage=self.config.leverage,
            mode=self.config.trading_mode.value,
            open_fee_rate=self.config.taker_fee_rate,  # Use taker rate
            close_fee_rate=self.config.taker_fee_rate,
        )

    def _open_position(self, time: datetime.datetime, row: pd.Series, side: str):
        """Enhanced position opening with realistic execution"""
        # Check if market is open
        if not self._is_market_open(time):
            return

        # Use next candle's open for execution, fallback to current close
        price = row.get("next_open", row["close"])
        if pd.isna(price):
            price = row["close"]

        atr_value = row.get("atr")

        # Calculate position parameters
        sizing_params = {
            "balance": self.balance,
            "equity": self.equity,
            "price": price,
            "side": side,
            "trade_history": pd.DataFrame(self.trades_info),
            "atr_value": atr_value,
        }

        initial_margin, sl_price, tp_price = (
            self.config.sizing_strategy.calculate_trade_parameters(**sizing_params)
        )

        if initial_margin <= 0 or self.balance < initial_margin:
            return

        # Check minimum notional value
        notional_value = initial_margin * self.config.leverage
        min_notional = 10.0  # $10 minimum trade size
        if notional_value < min_notional:
            return

        # Apply slippage to entry price
        slippage = self._calculate_slippage(price, row)
        if side == "long":
            price += slippage
        else:
            price -= slippage

        # Calculate market impact
        market_impact = self._calculate_market_impact(notional_value, row)
        if side == "long":
            price += market_impact / (notional_value / price)  # Impact per unit
        else:
            price -= market_impact / (notional_value / price)

        # Fallback SL/TP logic
        if sl_price is None and self.config.stop_loss_pct is not None:
            sl_price = price * (
                1 - self.config.stop_loss_pct
                if side == "long"
                else 1 + self.config.stop_loss_pct
            )
        if tp_price is None and self.config.take_profit_pct is not None:
            tp_price = price * (
                1 + self.config.take_profit_pct
                if side == "long"
                else 1 - self.config.take_profit_pct
            )

        cross_margin_balance = (
            self.equity if self.config.trading_mode.value == "Cross" else None
        )

        self.position.open(
            time,
            side,
            price,
            initial_margin,
            sl_price,
            tp_price,
            f"Signal {side}",
            cross_margin_balance,
        )

    def _close_position(self, time: datetime.datetime, price: float, reason: str):
        """Enhanced position closing with realistic execution"""
        if not self.position or not self.position.is_open:
            return

        self.position.close(time, price, reason)
        self.balance += self.position.net_pnl
        self.equity = self.balance
        self.trades_info.append(self.position.get_trade_info())
        self._create_new_position_object()

    def _handle_exits_realistic(self, time: datetime.datetime, row: pd.Series) -> bool:
        """Enhanced exit handling with gaps and slippage"""
        if not self.position.is_open:
            return False

        open_price = row["open"]
        high_price = row["high"]
        low_price = row["low"]
        close_price = row["close"]

        exit_price, exit_reason = None, None
        gap_exit = False

        # Check for gaps at market open
        if self.position.side == "long":
            if self.position.sl_price and open_price <= self.position.sl_price:
                exit_price = open_price
                exit_reason = "Stop Loss (Gap)"
                gap_exit = True
            elif self.position.tp_price and open_price >= self.position.tp_price:
                exit_price = open_price
                exit_reason = "Take Profit (Gap)"
                gap_exit = True
            elif (
                self.position.liquidation_price
                and open_price <= self.position.liquidation_price
            ):
                exit_price = max(open_price, 0.01)  # Can't go to zero
                exit_reason = "Liquidated (Gap)"
                gap_exit = True
        else:  # short
            if self.position.sl_price and open_price >= self.position.sl_price:
                exit_price = open_price
                exit_reason = "Stop Loss (Gap)"
                gap_exit = True
            elif self.position.tp_price and open_price <= self.position.tp_price:
                exit_price = open_price
                exit_reason = "Take Profit (Gap)"
                gap_exit = True
            elif (
                self.position.liquidation_price
                and open_price >= self.position.liquidation_price
            ):
                exit_price = open_price
                exit_reason = "Liquidated (Gap)"
                gap_exit = True

        # If no gap, check if levels were hit during the candle
        if exit_reason is None:
            # Check liquidation first (most important)
            if self.position.behavior.check_for_liquidation(
                self.position, low_price, high_price
            ):
                exit_price = self.position.liquidation_price
                exit_reason = "Liquidated"
            elif self.position.behavior.check_for_sl(
                self.position, low_price, high_price
            ):
                exit_price = self.position.sl_price
                exit_reason = "Stop Loss"
            elif self.position.behavior.check_for_tp(
                self.position, low_price, high_price
            ):
                exit_price = self.position.tp_price
                exit_reason = "Take Profit"

        if exit_reason:
            # Apply slippage
            slippage = self._calculate_slippage(exit_price, row, gap=gap_exit)
            if self.position.side == "long":
                exit_price -= slippage
            else:
                exit_price += slippage

            # Ensure price doesn't go negative
            exit_price = max(exit_price, 0.01)

            self._close_position(time, exit_price, exit_reason)
            return True

        return False

    def _handle_signal_changes(self, time: datetime.datetime, row: pd.Series):
        """Handle signal changes with enhanced logic"""
        if not self.position.is_open:
            return

        signal, close_price, side = row["signal"], row["close"], self.position.side
        is_reversal = self.config.allow_reverse_trade and (
            (side == "long" and signal == -1) or (side == "short" and signal == 1)
        )
        is_opposite_exit = self.config.exit_on_signal_opposite and (
            (side == "long" and signal == -1) or (side == "short" and signal == 1)
        )
        is_neutral_exit = self.config.exit_on_signal_0 and signal == 0

        if is_reversal:
            # Apply slippage to exit price
            slippage = self._calculate_slippage(close_price, row)
            if self.position.side == "long":
                close_price -= slippage
            else:
                close_price += slippage

            self._close_position(time, close_price, "Reversed")
            self._open_position(time, row, "short" if signal == -1 else "long")
        elif is_opposite_exit or is_neutral_exit:
            # Apply slippage to exit price
            slippage = self._calculate_slippage(close_price, row)
            if self.position.side == "long":
                close_price -= slippage
            else:
                close_price += slippage

            self._close_position(time, close_price, "Signal Exit")

    def _handle_entries(self, time: datetime.datetime, row: pd.Series):
        """Handle new entries with enhanced logic"""
        if self.position.is_open:
            return

        signal = row["signal"]
        if signal == 1 and self.config.trading_mode != TradingMode.SPOT:
            self._open_position(time, row, "long")
        elif signal == -1 and self.config.trading_mode != TradingMode.SPOT:
            self._open_position(time, row, "short")

    def _update_and_record_state(
        self,
        time: datetime.datetime,
        row: pd.Series,
        timeline_cols: dict,
        position_was_open: bool,
        current_trade_duration: int,
    ) -> int:
        """Update and record state with enhanced tracking"""
        close_price = row["close"]

        if self.position.is_open:
            if not position_was_open:
                current_trade_duration = 1
            else:
                current_trade_duration += 1

            # Update trailing stops if enabled
            if self.config.enable_trailing_stop:
                self.position.update_trailing_levels(close_price, self.config)
        else:
            if position_was_open:
                current_trade_duration = 0

        unrealized_pnl, notional_value, margin_used, unrealized_pnl_pct = 0, 0, 0, 0
        if self.position and self.position.is_open:
            unrealized_pnl = self.position.behavior.calculate_pnl(
                self.position, close_price
            )
            self.equity = self.balance + unrealized_pnl
            notional_value = self.position.amount * close_price
            margin_used = self.position.initial_margin
            unrealized_pnl_pct = (
                (unrealized_pnl / margin_used) * 100 if margin_used > 0 else 0
            )
        else:
            self.equity = self.balance

        # Record all state
        timeline_cols["balance"].append(self.balance)
        timeline_cols["equity"].append(self.equity)
        timeline_cols["position_side"].append(
            self.position.side if self.position.is_open else None
        )
        timeline_cols["unrealized_pnl"].append(unrealized_pnl)
        timeline_cols["position_amount"].append(
            self.position.amount if self.position.is_open else 0
        )
        timeline_cols["position_notional_value"].append(notional_value)
        timeline_cols["margin_used"].append(margin_used)
        timeline_cols["effective_leverage"].append(
            (notional_value / self.equity) if self.equity > 0 else 0
        )
        timeline_cols["active_entry_price"].append(
            self.position.open_price if self.position.is_open else np.nan
        )
        timeline_cols["active_sl_price"].append(
            self.position.sl_price if self.position.is_open else None
        )
        timeline_cols["active_tp_price"].append(
            self.position.tp_price if self.position.is_open else None
        )
        timeline_cols["active_liq_price"].append(
            self.position.liquidation_price if self.position.is_open else None
        )
        timeline_cols["trade_duration_candles"].append(current_trade_duration)
        timeline_cols["unrealized_pnl_pct"].append(unrealized_pnl_pct)

        self.equity_record.append({"time": time, "equity": self.equity})
        return current_trade_duration

    def run(self) -> BacktestAnalysis:
        """Enhanced backtest execution"""
        if self.is_running:
            return BacktestAnalysis(self)

        print("Starting enhanced backtest...")
        self.is_running = True
        self.balance = self.config.initial_balance
        self.equity = self.config.initial_balance
        self.trades_info, self.equity_record = [], []
        self._create_new_position_object()

        timeline_cols = {
            "balance": [],
            "equity": [],
            "position_side": [],
            "unrealized_pnl": [],
            "position_amount": [],
            "position_notional_value": [],
            "margin_used": [],
            "effective_leverage": [],
            "active_sl_price": [],
            "active_tp_price": [],
            "active_liq_price": [],
            "trade_duration_candles": [],
            "unrealized_pnl_pct": [],
            "active_entry_price": [],
        }

        # Add buy and hold benchmark
        self.data["buy_and_hold_equity"] = self.config.initial_balance * (
            self.data["close"] / self.data["close"].iloc[0]
        )

        current_trade_duration = 0

        for time, row in self.data.iterrows():
            position_was_open = self.position.is_open

            # Handle exits first (most important)
            exited = self._handle_exits_realistic(time, row)

            # Handle signal changes if position wasn't closed
            if not exited:
                self._handle_signal_changes(time, row)

            # Handle new entries
            self._handle_entries(time, row)

            # Update and record state
            current_trade_duration = self._update_and_record_state(
                time, row, timeline_cols, position_was_open, current_trade_duration
            )

        # Add timeline data to main dataframe
        for col_name, data_list in timeline_cols.items():
            self.data[col_name] = data_list

        # Calculate drawdown
        equity_ath = self.data["equity"].cummax()
        self.data["drawdown_pct"] = (equity_ath - self.data["equity"]) / equity_ath

        print("Enhanced backtest finished.")
        return BacktestAnalysis(self)
