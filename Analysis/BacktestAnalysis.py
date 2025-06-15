import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Import for the new interactive plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BacktestAnalysis:
    def __init__(self, runner):
        if not runner.is_running:
            raise ValueError("BacktestRunner has not been run yet.")

        self.results_df = runner.data.copy()
        self.trades = pd.DataFrame(runner.trades_info)
        self.wallet = self.results_df[["equity", "drawdown_pct"]].copy()

        if self.trades.empty:
            print("/!\\ No trades were executed during the backtest.")
            self.metrics = {}
            return

        self.metrics = self.compute_metrics()

    def compute_metrics(self) -> dict:
        """
        Computes a comprehensive set of performance metrics for the backtest.
        """
        metrics = {}
        # ... (This entire method remains unchanged from the previous version)
        # --- Trade Metrics ---
        self.trades["duration"] = pd.to_datetime(
            self.trades["close_time"]
        ) - pd.to_datetime(self.trades["open_time"])
        metrics["total_trades"] = len(self.trades)
        good_trades = self.trades.loc[self.trades["net_pnl"] > 0]
        bad_trades = self.trades.loc[self.trades["net_pnl"] < 0]
        metrics["total_good_trades"] = len(good_trades)
        metrics["total_bad_trades"] = len(bad_trades)
        metrics["avg_pnl_pct_good_trades"] = (
            good_trades["net_pnl_pct"].mean() if not good_trades.empty else 0
        )
        metrics["avg_pnl_pct_bad_trades"] = (
            bad_trades["net_pnl_pct"].mean() if not bad_trades.empty else 0
        )
        metrics["global_win_rate"] = (
            metrics["total_good_trades"] / metrics["total_trades"]
            if metrics["total_trades"] > 0
            else 0
        )
        metrics["total_profits"] = self.trades.loc[
            self.trades["net_pnl"] > 0, "net_pnl"
        ].sum()
        metrics["total_losses"] = abs(
            self.trades.loc[self.trades["net_pnl"] < 0, "net_pnl"].sum()
        )
        metrics["profit_factor"] = (
            metrics["total_profits"] / metrics["total_losses"]
            if metrics["total_losses"] != 0
            else float("inf")
        )
        metrics["avg_pnl_pct"] = self.trades["net_pnl_pct"].mean()
        # --- Equity and Return Metrics ---
        metrics["initial_balance"] = self.wallet.iloc[0]["equity"]
        metrics["final_balance"] = self.wallet.iloc[-1]["equity"]
        metrics["max_drawdown_equity"] = self.wallet["drawdown_pct"].max()
        total_return = (metrics["final_balance"] / metrics["initial_balance"]) - 1
        metrics["roi_pct"] = total_return * 100
        # --- Ratios ---
        daily_equity = self.wallet["equity"].resample("D").last()
        daily_returns = daily_equity.pct_change().dropna()
        if not daily_returns.empty:
            num_days = (self.wallet.index[-1] - self.wallet.index[0]).days
            num_years = num_days / 365.0
            annualized_return = (
                (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
            )
            daily_returns_mean = daily_returns.mean()
            daily_returns_std = daily_returns.std()
            metrics["sharpe_ratio"] = (
                (daily_returns_mean / daily_returns_std * np.sqrt(365))
                if daily_returns_std > 0
                else 0
            )
            downside_returns = daily_returns[daily_returns < 0]
            downside_deviation = downside_returns.std()
            metrics["sortino_ratio"] = (
                (daily_returns_mean / downside_deviation * np.sqrt(365))
                if downside_deviation > 0
                else 0
            )
            metrics["calmar_ratio"] = (
                annualized_return / metrics["max_drawdown_equity"]
                if metrics["max_drawdown_equity"] > 0
                else 0
            )
        else:
            (
                metrics["sharpe_ratio"],
                metrics["sortino_ratio"],
                metrics["calmar_ratio"],
            ) = (0, 0, 0)
        return metrics

    # === ▼▼▼ NEW INTERACTIVE PLOTTING METHOD ▼▼▼ ===

    def plot_interactive_trades(self):
        """
        Generates a high-performance, interactive candlestick chart with trade
        entry markers using Plotly.
        """
        if self.results_df.empty:
            print("No data to plot.")
            return

        df = self.results_df

        # This vectorized approach is highly efficient and avoids slow loops.
        # It finds the exact candle where a position *starts*.
        long_entries = df[
            (df["position_side"] == "long") & (df["position_side"].shift(1) != "long")
        ]
        short_entries = df[
            (df["position_side"] == "short") & (df["position_side"].shift(1) != "short")
        ]

        # Create the candlestick chart
        candlestick_trace = go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candlesticks",
        )

        # Create markers for long entries
        long_marker_trace = go.Scatter(
            x=long_entries.index,
            y=long_entries["low"] * 0.99,  # Place marker slightly below the low
            mode="markers",
            marker=dict(color="green", symbol="triangle-up", size=10),
            name="Long Entry",
        )

        # Create markers for short entries
        short_marker_trace = go.Scatter(
            x=short_entries.index,
            y=short_entries["high"] * 1.01,  # Place marker slightly above the high
            mode="markers",
            marker=dict(color="red", symbol="triangle-down", size=10),
            name="Short Entry",
        )

        # Combine traces and create the figure
        fig = go.Figure(data=[candlestick_trace, long_marker_trace, short_marker_trace])

        # Customize layout
        fig.update_layout(
            title="Interactive Trade Visualization",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",  # Use a dark theme
            xaxis_rangeslider_visible=False,  # Disable the range slider for a cleaner look
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Show the plot
        fig.show()

    # The rest of your plotting methods (print_metrics, plot_equity, etc.)
    def print_metrics(self):
        # ... (unchanged)
        if not self.metrics:
            print("No metrics to display.")
            return
        print("\n--- Backtest Results ---")
        print(
            f"Period: [{self.results_df.index[0].date()}] -> [{self.results_df.index[-1].date()}]"
        )
        print(f"Initial Balance:        {self.metrics.get('initial_balance', 0):,.2f}")
        print(f"Final Balance:          {self.metrics.get('final_balance', 0):,.2f}")
        print(f"ROI:                    {self.metrics.get('roi_pct', 0):.2f}%")
        print(f"Profit:                 {self.metrics.get('total_profits', 0):,.2f}")
        print(f"Loss:                   {self.metrics.get('total_losses', 0):.2f}")
        print(f"Sharpe Ratio:           {self.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio:          {self.metrics.get('sortino_ratio', 0):.2f}")
        print(f"Calmar Ratio:           {self.metrics.get('calmar_ratio', 0):.2f}")
        print(
            f"Max Drawdown:           {self.metrics.get('max_drawdown_equity', 0):.2%}"
        )
        print(f"Total Trades:           {self.metrics.get('total_trades', 0)}")
        print(f"Total Good Trades:      {self.metrics.get('total_good_trades', 0)}")
        print(f"Total Bad Trades:       {self.metrics.get('total_bad_trades', 0)}")
        print(
            f"Avg PnL Good Trades:    {self.metrics.get('avg_pnl_pct_good_trades', 0):.2f}%"
        )
        print(
            f"Avg PnL Bad Trades:     {self.metrics.get('avg_pnl_pct_bad_trades', 0):.2f}%"
        )
        print(f"Win Rate:               {self.metrics.get('global_win_rate', 0):.2%}")
        print(
            f"Loss Rate:              {1 - self.metrics.get('global_win_rate', 0):.2%}"
        )
        print(f"Profit Factor:          {self.metrics.get('profit_factor', 0):.2f}")

    def plot_equity(self):
        # ... (unchanged)
        if self.wallet.empty:
            return
            fig, ax1 = plt.subplots(figsize=(15, 7))
            ax1.set_title("Equity Curve vs. Asset Price")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Equity", color="tab:blue")
        ax1.plot(self.wallet.index, self.wallet["equity"], color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Asset Price", color="tab:grey")
        ax2.plot(
            self.results_df.index, self.results_df["close"], color="tab:grey", alpha=0.5
        )
        ax2.tick_params(axis="y", labelcolor="tab:grey")
        fig.tight_layout()
        plt.show()

    def plot_drawdown(self):
        # ... (unchanged)
        if self.wallet.empty:
            return
            plt.figure(figsize=(15, 7))
            plt.plot(
                self.wallet.index,
                self.wallet["drawdown_pct"] * 100,
                color="red",
                alpha=0.7,
            )
        plt.fill_between(
            self.wallet.index,
            self.wallet["drawdown_pct"] * 100,
            0,
            color="red",
            alpha=0.3,
        )
        plt.title("Drawdown Over Time")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_trade_distribution(self):
        # ... (unchanged)
        if self.trades.empty:
            return
            plt.figure(figsize=(15, 7))
            sns.histplot(data=self.trades["net_pnl_pct"], bins=50, kde=True)
        plt.title("Distribution of Trade Returns (%)")
        plt.xlabel("Return (%)")
        plt.ylabel("Frequency")
        plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_monthly_returns(self):
        # ... (unchanged)
        if self.wallet.empty:
            return
            monthly_returns = (
                self.wallet["equity"].resample("ME").last().pct_change() * 100
            )
        monthly_returns = monthly_returns.to_frame()
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns_pivot = pd.pivot_table(
            data=monthly_returns,
            values="equity",
            index=monthly_returns.index.year,
            columns=monthly_returns.index.month,
            aggfunc="first",
        )
        plt.figure(figsize=(15, 7))
        sns.heatmap(
            monthly_returns_pivot,
            annot=True,
            fmt=".1f",
            center=0,
            cmap="RdYlGn",
            cbar_kws={"label": "Returns (%)"},
        )
        plt.title("Monthly Returns Heatmap")
        plt.xlabel("Month")
        plt.ylabel("Year")
        plt.show()

    def plot_trade_analysis(self):
        # ... (unchanged)
        if self.trades.empty:
            return
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        ax1.plot(
            pd.to_datetime(self.trades["close_time"]),
            self.trades["net_pnl"].cumsum(),
            color="blue",
        )
        ax1.set_title("Cumulative PnL Over Time")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative PnL")
        ax1.grid(True, alpha=0.3)
        monthly_duration = (
            self.trades.set_index(pd.to_datetime(self.trades["close_time"]))["duration"]
            .resample("ME")
            .mean()
        )
        ax2.bar(
            monthly_duration.index,
            monthly_duration.dt.total_seconds() / 3600,
            color="green",
            alpha=0.7,
        )
        ax2.set_title("Average Trade Duration by Month")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Duration (hours)")
        ax2.grid(True, alpha=0.3)
        monthly_win_rate = (
            self.trades.set_index(pd.to_datetime(self.trades["close_time"]))["net_pnl"]
            > 0
        ).resample("ME").mean() * 100
        ax3.bar(monthly_win_rate.index, monthly_win_rate, color="purple", alpha=0.7)
        ax3.set_title("Monthly Win Rate")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Win Rate (%)")
        ax3.grid(True, alpha=0.3)
        ax4.hist(self.trades["net_pnl"], bins=50, color="orange", alpha=0.7)
        ax4.set_title("Trade PnL Distribution")
        ax4.set_xlabel("PnL")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_all(self):
        # As requested, the new interactive plot is NOT included here.
        # It must be called separately.
        self.plot_equity()
        self.plot_drawdown()
        self.plot_trade_distribution()
        self.plot_monthly_returns()
        self.plot_trade_analysis()
