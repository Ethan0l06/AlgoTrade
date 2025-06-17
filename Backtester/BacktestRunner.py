# /TradingBacktester/Backtester/BacktestRunner.py

import pandas as pd
import talib
from typing import Dict, Any, List, Optional
import datetime
import numpy as np

from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Config.Enums import TradingMode, PositionSizingMethod
from AlgoTrade.Utils.Position import Position
from AlgoTrade.Analysis.BacktestAnalysis import BacktestAnalysis


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
        self._prepare_data()

    def _prepare_data(self):
        if self.config.position_sizing_method in [PositionSizingMethod.ATR_VOLATILITY, PositionSizingMethod.ATR_BANDS]:
            self.data["atr"] = talib.ATR(self.data["high"], self.data["low"], self.data["close"], timeperiod=14)
        if "signal" not in self.data.columns:
            raise ValueError("Input data must contain a 'signal' column.")

    def _get_initial_margin(self, price: float, atr_value: Optional[float]) -> tuple[float, Optional[float]]:
        cfg = self.config
        if cfg.position_sizing_method == PositionSizingMethod.PERCENT_BALANCE:
            margin = self.balance * cfg.percent_balance_pct
        elif cfg.position_sizing_method == PositionSizingMethod.FIXED_AMOUNT:
            margin = cfg.fixed_amount_size
        elif cfg.position_sizing_method in [PositionSizingMethod.ATR_VOLATILITY, PositionSizingMethod.ATR_BANDS]:
            if atr_value is None or atr_value == 0:
                return 0, None

            risk_pct = cfg.atr_volatility_risk_pct if cfg.position_sizing_method == PositionSizingMethod.ATR_VOLATILITY else cfg.atr_bands_risk_pct
            multiplier = cfg.atr_volatility_multiplier if cfg.position_sizing_method == PositionSizingMethod.ATR_VOLATILITY else cfg.atr_bands_multiplier

            risk_in_dollars, stop_loss_distance = self.equity * risk_pct, atr_value * multiplier
            notional_value = (risk_in_dollars / stop_loss_distance) * price if stop_loss_distance > 0 else 0
            margin = notional_value / cfg.leverage if cfg.leverage > 0 else notional_value
        elif cfg.position_sizing_method == PositionSizingMethod.KELLY_CRITERION:
            if len(self.trades_info) < cfg.kelly_criterion_lookback:
                margin = self.balance * 0.01
            else:
                recent_trades = pd.DataFrame(self.trades_info[-cfg.kelly_criterion_lookback:])
                wins, losses = recent_trades[recent_trades["net_pnl"] > 0], recent_trades[recent_trades["net_pnl"] < 0]
                if len(wins) == 0 or len(losses) == 0:
                    margin = self.balance * 0.01
                else:
                    W, R = len(wins) / len(recent_trades), wins["net_pnl"].mean() / abs(losses["net_pnl"].mean())
                    kelly_pct = W - ((1 - W) / R)
                    margin = self.equity * kelly_pct * cfg.kelly_criterion_fraction if kelly_pct > 0 else 0
        else:
            raise ValueError(f"Unknown position sizing method: {cfg.position_sizing_method.value}")
        return margin, None

    def _create_new_position_object(self):
        self.position = Position(
            leverage=self.config.leverage,
            mode=self.config.trading_mode.value,
            open_fee_rate=self.config.open_fee_rate,
            close_fee_rate=self.config.close_fee_rate
        )

    def _open_position(self, time: datetime, row: pd.Series, side: str):
        price, atr_value = row["close"], row.get("atr")
        initial_margin, _ = self._get_initial_margin(price, atr_value)

        if initial_margin <= 0 or self.balance < initial_margin:
            return

        sl_price, tp_price = None, None
        cfg = self.config

        if cfg.position_sizing_method == PositionSizingMethod.ATR_VOLATILITY:
            stop_loss_distance = atr_value * cfg.atr_volatility_multiplier
            if side == "long":
                sl_price = price - stop_loss_distance
            else:  # short
                sl_price = price + stop_loss_distance

        elif cfg.position_sizing_method == PositionSizingMethod.ATR_BANDS:
            stop_loss_distance = atr_value * cfg.atr_bands_multiplier
            risk_reward_ratio = cfg.atr_bands_risk_reward_ratio
            if side == "long":
                sl_price = price - stop_loss_distance
                tp_price = price + (stop_loss_distance * risk_reward_ratio)
            else:  # short
                sl_price = price + stop_loss_distance
                tp_price = price - (stop_loss_distance * risk_reward_ratio)

        if sl_price is None and cfg.stop_loss_pct is not None:
            sl_price = price * (1 - cfg.stop_loss_pct if side == "long" else 1 + cfg.stop_loss_pct)
        if tp_price is None and cfg.take_profit_pct is not None:
            tp_price = price * (1 + cfg.take_profit_pct if side == "long" else 1 - cfg.take_profit_pct)

        self.balance -= initial_margin
        cross_margin_balance = self.equity if self.config.trading_mode == TradingMode.CROSS else None
        self.position.open(time, side, price, initial_margin, sl_price, tp_price, f"Signal {side}", cross_margin_balance)

    def _close_position(self, time: datetime, price: float, reason: str):
        if not self.position or not self.position.is_open:
            return
        self.position.close(time, price, reason)
        self.balance += self.position.initial_margin + self.position.net_pnl
        self.equity = self.balance
        self.trades_info.append(self.position.get_trade_info())
        self._create_new_position_object()

    def _handle_exits(self, time: datetime, low_price: float, high_price: float) -> bool:
        if not self.position.is_open:
            return False
        exit_price, exit_reason = None, None
        if self.position.behavior.check_for_liquidation(self.position, low_price, high_price):
            exit_price, exit_reason = self.position.liquidation_price, "Liquidated"
        elif self.position.behavior.check_for_sl(self.position, low_price, high_price):
            exit_price, exit_reason = self.position.sl_price, "Stop Loss"
        elif self.position.behavior.check_for_tp(self.position, low_price, high_price):
            exit_price, exit_reason = self.position.tp_price, "Take Profit"
        if exit_reason:
            self._close_position(time, exit_price, exit_reason)
            return True
        return False

    def _handle_signal_changes(self, time: datetime, row: pd.Series):
        if not self.position.is_open:
            return
        signal, close_price, side = row["signal"], row["close"], self.position.side
        is_reversal = self.config.allow_reverse_trade and ((side == "long" and signal == -1) or (side == "short" and signal == 1))
        is_opposite_exit = self.config.exit_on_signal_opposite and ((side == "long" and signal == -1) or (side == "short" and signal == 1))
        is_neutral_exit = self.config.exit_on_signal_0 and signal == 0
        if is_reversal:
            self._close_position(time, close_price, "Reversed")
            self._open_position(time, row, "short" if signal == -1 else "long")
        elif is_opposite_exit or is_neutral_exit:
            self._close_position(time, close_price, "Signal Exit")

    def _handle_entries(self, time: datetime, row: pd.Series):
        if self.position.is_open:
            return
        signal = row["signal"]
        if signal == 1 and self.config.trading_mode != TradingMode.SPOT:
            self._open_position(time, row, "long")
        elif signal == -1 and self.config.trading_mode != TradingMode.SPOT:
            self._open_position(time, row, "short")

    def _update_and_record_state(self, time: datetime, row: pd.Series, timeline_cols: dict, position_was_open: bool, current_trade_duration: int) -> int:
        close_price = row["close"]
        
        if self.position.is_open:
            if not position_was_open:
                current_trade_duration = 1
            else:
                current_trade_duration += 1
        else:
            if position_was_open:
                current_trade_duration = 0

        unrealized_pnl, notional_value, margin_used, unrealized_pnl_pct = 0, 0, 0, 0
        if self.position and self.position.is_open:
            unrealized_pnl = self.position.behavior.calculate_pnl(self.position, close_price)
            self.equity = self.balance + self.position.initial_margin + unrealized_pnl
            notional_value = self.position.amount * close_price
            margin_used = self.position.initial_margin
            unrealized_pnl_pct = (unrealized_pnl / margin_used) * 100 if margin_used > 0 else 0
        else:
            self.equity = self.balance

        timeline_cols["equity"].append(self.equity)
        timeline_cols["position_side"].append(self.position.side if self.position.is_open else None)
        timeline_cols["unrealized_pnl"].append(unrealized_pnl)
        timeline_cols["position_amount"].append(self.position.amount if self.position.is_open else 0)
        timeline_cols["position_notional_value"].append(notional_value)
        timeline_cols["margin_used"].append(margin_used)
        timeline_cols["effective_leverage"].append((notional_value / self.equity) if self.equity > 0 else 0)
        timeline_cols["active_entry_price"].append(self.position.open_price if self.position.is_open else np.nan)
        timeline_cols["active_sl_price"].append(self.position.sl_price if self.position.is_open else None)
        timeline_cols["active_tp_price"].append(self.position.tp_price if self.position.is_open else None)
        timeline_cols["active_liq_price"].append(self.position.liquidation_price if self.position.is_open else None)
        timeline_cols["trade_duration_candles"].append(current_trade_duration)
        timeline_cols["unrealized_pnl_pct"].append(unrealized_pnl_pct)

        self.equity_record.append({"time": time, "equity": self.equity})
        return current_trade_duration

    def run(self) -> BacktestAnalysis:
        if self.is_running:
            return BacktestAnalysis(self)
        print("Starting backtest...")
        self.is_running = True
        self.balance = self.config.initial_balance
        self.equity = self.config.initial_balance
        self.trades_info, self.equity_record = [], []
        self._create_new_position_object()

        timeline_cols = {
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

        self.data["buy_and_hold_equity"] = self.config.initial_balance * (self.data["close"] / self.data["close"].iloc[0])
        current_trade_duration = 0

        for time, row in self.data.iterrows():
            position_was_open = self.position.is_open
            
            exited = self._handle_exits(time, row["low"], row["high"])
            if not exited:
                self._handle_signal_changes(time, row)

            self._handle_entries(time, row)

            current_trade_duration = self._update_and_record_state(time, row, timeline_cols, position_was_open, current_trade_duration)

        for col_name, data_list in timeline_cols.items():
            self.data[col_name] = data_list

        equity_ath = self.data["equity"].cummax()
        self.data["drawdown_pct"] = (equity_ath - self.data["equity"]) / equity_ath

        print("Backtest finished.")
        return BacktestAnalysis(self)
