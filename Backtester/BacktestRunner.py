# /TradingBacktester/Backtester/BacktestRunner.py

import pandas as pd
import talib
from typing import Dict, Any, List, Optional
import datetime
import numpy as np

from StrategyLab.Config.BacktestConfig import BacktestConfig
from StrategyLab.Utils.Position import Position
from StrategyLab.Analysis.BacktestAnalysis import BacktestAnalysis


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
        # Calculate ATR specifically for the chosen method to avoid unnecessary computation
        if self.config.position_sizing_method == "AtrVolatility":
            self.data["atr"] = talib.ATR(
                self.data["high"],
                self.data["low"],
                self.data["close"],
                timeperiod=self.config.atr_volatility_period,
            )
        elif self.config.position_sizing_method == "AtrBands":
            self.data["atr"] = talib.ATR(
                self.data["high"],
                self.data["low"],
                self.data["close"],
                timeperiod=self.config.atr_bands_period,
            )

        if "signal" not in self.data.columns:
            raise ValueError("Input data must contain a 'signal' column.")

    def _get_initial_margin(
        self, price: float, atr_value: Optional[float]
    ) -> tuple[float, Optional[float]]:
        sl_price, cfg = None, self.config
        if cfg.position_sizing_method == "PercentBalance":
            margin = self.balance * cfg.percent_balance_pct
        elif cfg.position_sizing_method == "FixedAmount":
            margin = cfg.fixed_amount_size
        elif (
            cfg.position_sizing_method == "AtrVolatility"
            or cfg.position_sizing_method
            == "AtrBands"  # Both methods use similar sizing
        ):
            if atr_value is None or atr_value == 0:
                return 0, None

            # Determine which set of parameters to use
            if cfg.position_sizing_method == "AtrVolatility":
                risk_pct = cfg.atr_volatility_risk_pct
                multiplier = cfg.atr_volatility_multiplier
            else:  # AtrBands
                risk_pct = cfg.atr_bands_risk_pct
                multiplier = cfg.atr_bands_multiplier

            risk_in_dollars, stop_loss_distance = (
                self.equity * risk_pct,
                atr_value * multiplier,
            )
            # The SL price itself is side-dependent, handled in _open_position
            # Here we only need the distance to calculate the size
            notional_value = (
                (risk_in_dollars / stop_loss_distance) * price
                if stop_loss_distance > 0
                else 0
            )
            margin = (
                notional_value / cfg.leverage if cfg.leverage > 0 else notional_value
            )
        elif cfg.position_sizing_method == "KellyCriterion":
            if len(self.trades_info) < cfg.kelly_criterion_lookback:
                margin = self.balance * 0.01
            else:
                recent_trades = pd.DataFrame(
                    self.trades_info[-cfg.kelly_criterion_lookback :]
                )
                wins, losses = (
                    recent_trades[recent_trades["net_pnl"] > 0],
                    recent_trades[recent_trades["net_pnl"] < 0],
                )
                if len(wins) == 0 or len(losses) == 0:
                    margin = self.balance * 0.01
                else:
                    W, R = len(wins) / len(recent_trades), wins["net_pnl"].mean() / abs(
                        losses["net_pnl"].mean()
                    )
                    kelly_pct = W - ((1 - W) / R)
                    margin = (
                        self.equity * kelly_pct * cfg.kelly_criterion_fraction
                        if kelly_pct > 0
                        else 0
                    )
        else:
            raise ValueError(
                f"Unknown position sizing method: {cfg.position_sizing_method}"
            )
        return margin, sl_price  # sl_price is no longer set here, will be None

    def _create_new_position_object(self):
        self.position = Position(
            leverage=self.config.leverage,
            mode=self.config.trading_mode,
            open_fee_rate=self.config.open_fee_rate,
            close_fee_rate=self.config.close_fee_rate,
        )

    def _open_position(self, time: datetime, row: pd.Series, side: str):
        price, atr_value = row["close"], row.get("atr")
        initial_margin, _ = self._get_initial_margin(price, atr_value)

        if initial_margin <= 0 or self.balance < initial_margin:
            return

        sl_price, tp_price = None, None
        cfg = self.config

        # SL/TP logic based on position sizing method
        if cfg.position_sizing_method == "AtrVolatility":
            stop_loss_distance = atr_value * cfg.atr_volatility_multiplier
            if side == "long":
                sl_price = price - stop_loss_distance
            else:  # short
                sl_price = price + stop_loss_distance

        elif cfg.position_sizing_method == "AtrBands":
            stop_loss_distance = atr_value * cfg.atr_bands_multiplier
            risk_reward_ratio = cfg.atr_bands_risk_reward_ratio
            if side == "long":
                sl_price = price - stop_loss_distance
                tp_price = price + (stop_loss_distance * risk_reward_ratio)
            else:  # short
                sl_price = price + stop_loss_distance
                tp_price = price - (stop_loss_distance * risk_reward_ratio)

        # Fallback to percentage-based SL/TP if not set by the sizing method
        if sl_price is None and cfg.stop_loss_pct is not None:
            sl_price = price * (
                1 - cfg.stop_loss_pct if side == "long" else 1 + cfg.stop_loss_pct
            )
        if tp_price is None and cfg.take_profit_pct is not None:
            tp_price = price * (
                1 + cfg.take_profit_pct if side == "long" else 1 - cfg.take_profit_pct
            )

        self.balance -= initial_margin
        cross_margin_balance = (
            self.equity if self.config.trading_mode == "Cross" else None
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

    def _close_position(self, time: datetime, price: float, reason: str):
        if not self.position or not self.position.is_open:
            return
        self.position.close(time, price, reason)
        self.balance += self.position.initial_margin + self.position.net_pnl
        self.equity = self.balance
        self.trades_info.append(self.position.get_trade_info())
        self._create_new_position_object()

    def run(self) -> BacktestAnalysis:
        if self.is_running:
            return BacktestAnalysis(self)
        print("Starting backtest...")
        self.is_running = True
        self.balance = self.config.initial_balance
        self.equity = self.config.initial_balance
        self.trades_info, self.equity_record = [], []
        self._create_new_position_object()

        # --- Initialize lists for all new columns ---
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

        # --- Pre-calculate Buy & Hold for performance ---
        self.data["buy_and_hold_equity"] = self.config.initial_balance * (
            self.data["close"] / self.data["close"].iloc[0]
        )

        # --- Initialize loop-dependent counters ---
        current_trade_duration = 0

        for time, row in self.data.iterrows():
            close_price, high_price, low_price, signal = (
                row["close"],
                row["high"],
                row["low"],
                row["signal"],
            )

            position_was_open = self.position.is_open

            # --- TRADING LOGIC ---
            if self.position.is_open:
                exit_price, exit_reason = None, None
                if self.position.behavior.check_for_liquidation(
                    self.position, low_price, high_price
                ):
                    exit_price, exit_reason = (
                        self.position.liquidation_price,
                        "Liquidated",
                    )
                elif self.position.behavior.check_for_sl(
                    self.position, low_price, high_price
                ):
                    exit_price, exit_reason = self.position.sl_price, "Stop Loss"
                elif self.position.behavior.check_for_tp(
                    self.position, low_price, high_price
                ):
                    exit_price, exit_reason = self.position.tp_price, "Take Profit"
                if exit_reason:
                    self._close_position(time, exit_price, exit_reason)

            if self.position.is_open:
                if self.config.allow_reverse_trade and (
                    (self.position.side == "long" and signal == -1)
                    or (self.position.side == "short" and signal == 1)
                ):
                    self._close_position(time, close_price, "Reversed")
                    self._open_position(time, row, "short" if signal == -1 else "long")
                elif (
                    self.config.exit_on_signal_opposite
                    and (
                        (self.position.side == "long" and signal == -1)
                        or (self.position.side == "short" and signal == 1)
                    )
                ) or (self.config.exit_on_signal_0 and signal == 0):
                    self._close_position(time, close_price, "Signal Exit")

            if not self.position.is_open:
                if signal == 1 and self.config.trading_mode != "ShortOnly":
                    self._open_position(time, row, "long")
                elif signal == -1 and self.config.trading_mode != "Spot":
                    self._open_position(time, row, "short")

            # --- DURATION COUNTER LOGIC ---
            if self.position.is_open:
                if not position_was_open:
                    current_trade_duration = 1  # Start of a new trade
                else:
                    current_trade_duration += 1  # Continuation of an existing trade
            else:
                if position_was_open:
                    current_trade_duration = 0  # Trade just closed

            # --- RECORD STATE FOR THIS CANDLE ---
            unrealized_pnl, notional_value, margin_used, unrealized_pnl_pct = 0, 0, 0, 0
            if self.position and self.position.is_open:
                unrealized_pnl = self.position.behavior.calculate_pnl(
                    self.position, close_price
                )
                self.equity = (
                    self.balance + self.position.initial_margin + unrealized_pnl
                )
                notional_value = self.position.amount * close_price
                margin_used = self.position.initial_margin
                unrealized_pnl_pct = (
                    (unrealized_pnl / margin_used) * 100 if margin_used > 0 else 0
                )
            else:
                self.equity = self.balance

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

        # --- Add all recorded timelines to the final DataFrame ---
        for col_name, data_list in timeline_cols.items():
            self.data[col_name] = data_list

        equity_ath = self.data["equity"].cummax()
        self.data["drawdown_pct"] = (equity_ath - self.data["equity"]) / equity_ath

        print("Backtest finished.")
        return BacktestAnalysis(self)
