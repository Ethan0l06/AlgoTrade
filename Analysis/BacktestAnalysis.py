import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from typing import Optional, Dict, List

# Import for the new interactive plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import quantstats_lumi as quantstats
from AlgoTrade.Config.Paths import QUANTSSTATS_DIR


class BacktestAnalysis:
    def __init__(self, runner):
        if not runner.is_running:
            raise ValueError("BacktestRunner has not been run yet.")

        self.results_df = runner.data.copy()
        self.trades = pd.DataFrame(runner.trades_info)
        # <- MODIFICATION: Added 'balance' to the wallet DataFrame
        self.wallet = self.results_df[["balance", "equity", "drawdown_pct"]].copy()

        if self.trades.empty:
            print("/!\\ No trades were executed during the backtest.")
            self.metrics = {}
            self.table_plotter = None
            return

        self.metrics = self.compute_metrics()

        # Initialize table plotter for enhanced analysis (lazy import to avoid circular dependency)
        self.table_plotter = None

    def compute_metrics(self) -> dict:
        """
        Computes a comprehensive set of performance metrics for the backtest.
        """
        metrics = {}
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

        # <- MODIFICATION: Updated this whole section for clarity
        # --- Equity and Return Metrics ---
        metrics["initial_balance"] = self.wallet.iloc[0]["balance"]
        metrics["final_balance"] = self.wallet.iloc[-1]["balance"]
        metrics["initial_equity"] = self.wallet.iloc[0]["equity"]
        metrics["final_equity"] = self.wallet.iloc[-1]["equity"]

        metrics["max_drawdown_equity"] = self.wallet["drawdown_pct"].max()
        # ROI is based on the growth of total Equity
        total_return = (
            (metrics["final_equity"] / metrics["initial_equity"]) - 1
            if metrics["initial_equity"] > 0
            else 0
        )
        metrics["roi_pct"] = total_return * 100

        # --- Ratios ---
        daily_equity = self.wallet["equity"].resample("D").last()
        daily_returns = daily_equity.pct_change().dropna()

        if not daily_returns.empty and not daily_returns.eq(0).all():
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

    def _get_table_plotter(self):
        """Lazy initialization of table plotter to avoid circular imports."""
        if self.table_plotter is None:
            from AlgoTrade.Analysis.TablePlots import AnalysisTablePlotter

            self.table_plotter = AnalysisTablePlotter(self, style="dark")
        return self.table_plotter

    # === ▼▼▼ NEW TABLE ANALYSIS METHODS ▼▼▼ ===

    def show_performance_summary_table(self, save_path: str = None):
        """Show interactive performance summary table."""
        if not self.metrics:
            print("No table plotter available. No trades were executed.")
            return

        plotter = self._get_table_plotter()
        fig = plotter.create_performance_summary_table(save_path)
        if fig:
            fig.show()

    def show_trades_analysis_table(self, top_n: int = 10, save_path: str = None):
        """Show detailed trades analysis table."""
        if not self.metrics:
            print("No table plotter available. No trades were executed.")
            return

        plotter = self._get_table_plotter()
        fig = plotter.create_trades_analysis_table(top_n)
        if fig:
            fig.show()

    def show_monthly_performance_table(self, save_path: str = None):
        """Show monthly performance breakdown table."""
        if not self.metrics:
            print("No table plotter available. No trades were executed.")
            return

        plotter = self._get_table_plotter()
        fig = plotter.create_monthly_performance_table()
        if fig:
            fig.show()

    def show_risk_metrics_table(self, save_path: str = None):
        """Show comprehensive risk metrics table."""
        if not self.metrics:
            print("No table plotter available. No trades were executed.")
            return

        plotter = self._get_table_plotter()
        fig = plotter.create_risk_metrics_table()
        if fig:
            fig.show()

    def show_trade_statistics_table(self, save_path: str = None):
        """Show detailed trade statistics table."""
        if not self.metrics:
            print("No table plotter available. No trades were executed.")
            return

        plotter = self._get_table_plotter()
        fig = plotter.create_trade_statistics_table()
        if fig:
            fig.show()

    def show_complete_analysis_dashboard(self, save_directory: str = None):
        """
        Show complete analysis dashboard with all tables.

        Args:
            save_directory: Optional directory to save all HTML files
        """
        if not self.metrics:
            print("No table plotter available. No trades were executed.")
            return

        print("\n" + "=" * 60)
        print("🚀 COMPREHENSIVE TRADING ANALYSIS DASHBOARD")
        print("=" * 60)

        plotter = self._get_table_plotter()
        plotter.show_dashboard()

        if save_directory:
            figures = plotter.create_dashboard(save_path=save_directory)
            print(f"\n💾 All analysis tables saved to: {save_directory}")
            
    def _get_graph_plotter(self):
        """Lazy initialization of graph plotter to avoid circular imports."""
        if not hasattr(self, 'graph_plotter') or self.graph_plotter is None:
            from AlgoTrade.Analysis.GraphPlots import TradingGraphPlotter
            self.graph_plotter = TradingGraphPlotter(self, style="dark")
        return self.graph_plotter

    # === ▼▼▼ NEW GRAPH ANALYSIS METHODS ▼▼▼ ===

    def show_equity_with_drawdown(self):
        """Show equity curve with drawdown analysis."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_equity_with_drawdown()
        if fig: fig.show()

    def show_rolling_performance(self, windows: List[int] = [30, 60, 90]):
        """Show rolling performance metrics."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_rolling_performance(windows)
        if fig: fig.show()

    def show_cumulative_returns_comparison(self):
        """Show cumulative returns vs benchmarks."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_cumulative_returns_comparison()
        if fig: fig.show()

    def show_streak_analysis(self):
        """Show win/loss streak analysis."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_streak_analysis()
        if fig: fig.show()

    def show_correlation_analysis(self, benchmark_data: Optional[pd.DataFrame] = None):
        """Show correlation analysis with market factors."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_correlation_analysis(benchmark_data)
        if fig: fig.show()

    def show_calendar_heatmap(self):
        """Show calendar-based performance heatmap."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_calendar_heatmap()
        if fig: fig.show()

    def show_market_regime_analysis(self):
        """Show performance across different market regimes."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_market_regime_analysis()
        if fig: fig.show()

    def show_interactive_trade_explorer(self):
        """Show interactive candlestick chart with trade markers."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_interactive_trade_explorer()
        if fig: fig.show()

    def show_parameter_sensitivity(self, param_results: Dict[str, Dict[str, float]]):
        """Show parameter sensitivity analysis."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_parameter_sensitivity(param_results)
        if fig: fig.show()

    def show_animated_equity_buildup(self, speed: int = 50):
        """Show animated equity curve buildup."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.create_animated_equity_buildup(speed)
        if fig: fig.show()

    def show_monte_carlo_analysis(self, n_simulations: int = 1000, forecast_days: int = 252):
        """Show Monte Carlo simulation analysis."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_monte_carlo_analysis(n_simulations, forecast_days)
        if fig: fig.show()

    def show_ml_insights(self):
        """Show machine learning insights and patterns."""
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        plotter = self._get_graph_plotter()
        fig = plotter.plot_ml_insights()
        if fig: fig.show()

    def show_complete_graph_dashboard(self, save_directory: str = None, param_results: dict = None):
        """
        Show complete graph analysis dashboard with all visualizations.
        
        Args:
            save_directory: Optional directory to save all HTML files
            param_results: Optional parameter sensitivity data
        """
        if not self.metrics:
            print("No data available for graph analysis.")
            return
        
        print("\n" + "="*60)
        print("🚀 COMPREHENSIVE GRAPH ANALYSIS DASHBOARD")
        print("="*60)
        
        plotter = self._get_graph_plotter()
        plotter.show_graph_dashboard(param_results=param_results)
        
        if save_directory:
            figures = plotter.create_complete_graph_dashboard(save_path=save_directory, param_results=param_results)
            print(f"\n💾 All graphs saved to: {save_directory}")

    def show_comparative_analysis_table(self, comparative_results: dict):
        """
        Show comparative analysis table for multiple strategies.

        Args:
            comparative_results: Dictionary of strategy_name -> BacktestAnalysis
        """
        if not self.metrics:
            print("No table plotter available. No trades were executed.")
            return

        plotter = self._get_table_plotter()
        fig = plotter.create_comparative_performance_table(comparative_results)
        if fig:
            fig.show()

    # The rest of your existing plotting methods
    def print_metrics(self):
        """Print metrics in a formatted way."""
        if not self.metrics:
            print("No metrics to display.")
            return
        print("\n--- Backtest Results ---")
        print(
            f"Period: [{self.results_df.index[0].date()}] -> [{self.results_df.index[-1].date()}]"
        )
        # <- MODIFICATION: Updated printout for clarity
        print(f"Initial Balance:        {self.metrics.get('initial_balance', 0):,.2f}")
        print(f"Final Balance:          {self.metrics.get('final_balance', 0):,.2f}")
        print(f"Initial Equity:         {self.metrics.get('initial_equity', 0):,.2f}")
        print(f"Final Equity:           {self.metrics.get('final_equity', 0):,.2f}")
        print(f"ROI (Equity):           {self.metrics.get('roi_pct', 0):.2f}%")
        print(f"Profit:                 {self.metrics.get('total_profits', 0):,.2f}")
        print(f"Loss:                   {self.metrics.get('total_losses', 0):,.2f}")
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

    def generate_quantstats_report(self, output_filename="strategy_report.html"):
        """Generates a comprehensive HTML report using the quantstats library."""
        if not quantstats:
            print("Cannot generate report: quantstats library is not installed.")
            return
        if self.results_df.empty:
            print("No data available to generate a report.")
            return

        output_path = QUANTSSTATS_DIR / output_filename
        print(f"Generating QuantStats report to {output_path}...")
        returns = self.results_df["equity"].resample("D").last().pct_change().fillna(0)
        quantstats.reports.html(
            returns, output=str(output_path), title="Strategy Performance Report"
        )
        print(f"Report saved to {output_path}")
