# AlgoTrade/Analysis/GraphPlots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import warnings

warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    from AlgoTrade.Analysis.BacktestAnalysis import BacktestAnalysis


class TradingGraphPlotter:
    """
    Comprehensive graph plotting system for trading strategy analysis.
    Provides advanced visualizations to understand strategy performance,
    risk characteristics, and trading patterns.
    """

    def __init__(self, analysis: "BacktestAnalysis", style: str = "dark"):
        """
        Initialize the graph plotter.

        Args:
            analysis: BacktestAnalysis object containing backtest results
            style: Visual style - "dark", "light", or "plotly"
        """
        self.analysis = analysis
        self.results_df = analysis.results_df
        self.trades_df = analysis.trades
        self.metrics = analysis.metrics
        self.style = style

        # Configure plotly template based on style
        if style == "dark":
            self.template = "plotly_dark"
        elif style == "light":
            self.template = "plotly_white"
        else:
            self.template = "plotly"

    # ==================== 1. EQUITY CURVE WITH DRAWDOWN ====================

    def plot_equity_with_drawdown(self) -> go.Figure:
        """
        Creates equity curve with drawdown visualization.

        WHAT IT TELLS YOU:
        - Overall strategy performance over time
        - Periods of gains vs losses
        - Severity and duration of drawdowns
        - Recovery patterns after losses

        HOW IT HELPS:
        - Identify if strategy is consistently profitable
        - Spot periods of high risk/volatility
        - Evaluate if drawdowns are acceptable for your risk tolerance
        - Compare performance vs buy-and-hold benchmark
        - Detect regime changes in strategy performance
        """
        if self.results_df.empty:
            print("No data available for equity curve.")
            return None

        df = self.results_df.copy()

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            row_heights=[0.6, 0.25, 0.15],
            subplot_titles=(
                "Portfolio Equity Over Time",
                "Drawdown %",
                "Position Exposure",
            ),
            vertical_spacing=0.05,
            shared_xaxes=True,
        )

        # 1. Equity curve
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["equity"],
                name="Strategy Equity",
                line=dict(color="#00ff88", width=2),
                hovertemplate="<b>Date:</b> %{x}<br><b>Equity:</b> $%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add buy & hold if available
        if "buy_and_hold_equity" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["buy_and_hold_equity"],
                    name="Buy & Hold",
                    line=dict(color="#ff6b6b", width=1, dash="dash"),
                    hovertemplate="<b>Date:</b> %{x}<br><b>B&H Equity:</b> $%{y:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # 2. Drawdown
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=-df["drawdown_pct"] * 100,  # Negative for visual appeal
                name="Drawdown %",
                fill="tonexty",
                fillcolor="rgba(255, 0, 0, 0.3)",
                line=dict(color="red", width=1),
                hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 3. Position exposure
        if "position_side" in df.columns:
            # Create position exposure signal
            position_signal = (
                df["position_side"].map({"long": 1, "short": -1}).fillna(0)
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=position_signal,
                    name="Position",
                    mode="lines",
                    line=dict(color="orange", width=1),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Position:</b> %{y}<extra></extra>",
                ),
                row=3,
                col=1,
            )

        # Layout updates
        fig.update_layout(
            title=dict(
                text="<b>📈 Strategy Performance Analysis</b><br>"
                f"<sub>Total Return: {self.metrics.get('roi_pct', 0):.1f}% | "
                f"Max DD: {self.metrics.get('max_drawdown_equity', 0)*100:.1f}% | "
                f"Sharpe: {self.metrics.get('sharpe_ratio', 0):.2f}</sub>",
                x=0.5,
                font=dict(size=16),
            ),
            template=self.template,
            height=700,
            showlegend=True,
            hovermode="x unified",
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(
            title_text="Position",
            row=3,
            col=1,
            tickvals=[-1, 0, 1],
            ticktext=["Short", "Flat", "Long"],
        )
        fig.update_xaxes(title_text="Date", row=3, col=1)

        return fig

    # ==================== 2. ROLLING PERFORMANCE METRICS ====================

    def plot_rolling_performance(self, windows: List[int] = [30, 60, 90]) -> go.Figure:
        """
        Creates rolling performance metrics analysis.

        WHAT IT TELLS YOU:
        - How strategy performance changes over time
        - Periods of consistent vs inconsistent performance
        - Whether strategy adapts to changing market conditions
        - Early warning signs of strategy degradation

        HOW IT HELPS:
        - Identify when to stop/start using a strategy
        - Detect optimal rebalancing periods
        - Monitor strategy health in real-time
        - Compare different time horizons for stability
        - Spot overfitting vs robust performance
        """
        if self.results_df.empty:
            print("No data available for rolling performance.")
            return None

        df = self.results_df.copy()

        # Calculate daily returns
        daily_returns = df["equity"].pct_change().dropna()

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Rolling Sharpe Ratio",
                "Rolling Win Rate",
                "Rolling Volatility",
            ),
            vertical_spacing=0.1,
            shared_xaxes=True,
        )

        colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]

        for i, window in enumerate(windows):
            # Rolling Sharpe
            rolling_sharpe = (
                daily_returns.rolling(window).mean()
                / daily_returns.rolling(window).std()
                * np.sqrt(252)
            )

            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe,
                    name=f"{window}D Sharpe",
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"<b>{window}D Sharpe:</b> %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Rolling Win Rate (for trades)
            if not self.trades_df.empty:
                # Calculate rolling win rate based on trade close times
                trades_with_dates = self.trades_df.copy()
                trades_with_dates["close_time"] = pd.to_datetime(
                    trades_with_dates["close_time"]
                )
                trades_with_dates["is_win"] = trades_with_dates["net_pnl"] > 0
                trades_with_dates = trades_with_dates.set_index(
                    "close_time"
                ).sort_index()

                rolling_win_rate = (
                    trades_with_dates["is_win"].rolling(f"{window}D").mean() * 100
                )

                fig.add_trace(
                    go.Scatter(
                        x=rolling_win_rate.index,
                        y=rolling_win_rate,
                        name=f"{window}D Win Rate",
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f"<b>{window}D Win Rate:</b> %{{y:.1f}}%<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

            # Rolling Volatility
            rolling_vol = daily_returns.rolling(window).std() * np.sqrt(252) * 100

            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    name=f"{window}D Vol",
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"<b>{window}D Volatility:</b> %{{y:.1f}}%<extra></extra>",
                ),
                row=3,
                col=1,
            )

        # Add reference lines
        fig.add_hline(
            y=1.0,
            row=1,
            col=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="Sharpe = 1.0",
        )
        fig.add_hline(
            y=50,
            row=2,
            col=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="50% Win Rate",
        )

        fig.update_layout(
            title="<b>📊 Rolling Performance Metrics</b>",
            template=self.template,
            height=800,
            showlegend=True,
        )

        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        return fig

    # ==================== 3. CUMULATIVE RETURNS COMPARISON ====================

    def plot_cumulative_returns_comparison(self) -> go.Figure:
        """
        Compares strategy returns vs benchmarks and different metrics.

        WHAT IT TELLS YOU:
        - How strategy performs vs buy-and-hold
        - Risk-adjusted performance comparison
        - Value of active management vs passive
        - Consistency of outperformance

        HOW IT HELPS:
        - Justify using active strategy vs passive investing
        - Identify periods where strategy adds/destroys value
        - Set realistic performance expectations
        - Evaluate if complexity is worth the results
        - Make allocation decisions between strategies
        """
        if self.results_df.empty:
            print("No data available for cumulative returns.")
            return None

        df = self.results_df.copy()

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Cumulative Returns",
                "Risk-Adjusted Returns",
                "Monthly Returns Distribution",
                "Rolling Correlation",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Calculate normalized returns (starting from 100)
        strategy_returns = (df["equity"] / df["equity"].iloc[0]) * 100

        # 1. Cumulative Returns
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=strategy_returns,
                name="Strategy",
                line=dict(color="#00ff88", width=3),
                hovertemplate="<b>Strategy:</b> %{y:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        if "buy_and_hold_equity" in df.columns:
            bh_returns = (
                df["buy_and_hold_equity"] / df["buy_and_hold_equity"].iloc[0]
            ) * 100
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=bh_returns,
                    name="Buy & Hold",
                    line=dict(color="#ff6b6b", width=2, dash="dash"),
                    hovertemplate="<b>Buy & Hold:</b> %{y:.1f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # 2. Risk-Adjusted Returns (Sharpe-weighted)
        daily_returns = df["equity"].pct_change().dropna()
        if len(daily_returns) > 30:
            rolling_sharpe = (
                daily_returns.rolling(30).mean() / daily_returns.rolling(30).std()
            )
            risk_adj_returns = strategy_returns * (
                rolling_sharpe / rolling_sharpe.abs().max()
            ).fillna(0)

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=risk_adj_returns,
                    name="Risk-Adjusted Strategy",
                    line=dict(color="#4ecdc4", width=2),
                    hovertemplate="<b>Risk-Adj:</b> %{y:.1f}<extra></extra>",
                ),
                row=1,
                col=2,
            )

        # 3. Monthly Returns Distribution
        monthly_returns = (
            daily_returns.resample("M").apply(lambda x: (1 + x).prod() - 1) * 100
        )

        fig.add_trace(
            go.Histogram(
                x=monthly_returns,
                name="Monthly Returns",
                nbinsx=20,
                marker_color="#45b7d1",
                opacity=0.7,
                hovertemplate="<b>Return Range:</b> %{x:.1f}%<br><b>Count:</b> %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 4. Rolling Correlation (if benchmark available)
        if "buy_and_hold_equity" in df.columns:
            bh_daily_returns = df["buy_and_hold_equity"].pct_change().dropna()
            if len(daily_returns) == len(bh_daily_returns):
                rolling_corr = daily_returns.rolling(60).corr(bh_daily_returns)

                fig.add_trace(
                    go.Scatter(
                        x=rolling_corr.index,
                        y=rolling_corr,
                        name="60D Correlation",
                        line=dict(color="#ff9f43", width=2),
                        hovertemplate="<b>Correlation:</b> %{y:.2f}<extra></extra>",
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            title="<b>📈 Cumulative Returns Analysis</b>",
            template=self.template,
            height=700,
            showlegend=True,
        )

        return fig

    # ==================== 6. WIN/LOSS STREAK ANALYSIS ====================

    def plot_streak_analysis(self) -> go.Figure:
        """
        Analyzes winning and losing streaks in trading.

        WHAT IT TELLS YOU:
        - How long winning/losing streaks typically last
        - Maximum streak lengths experienced
        - Distribution of streak patterns
        - Psychological pressure points

        HOW IT HELPS:
        - Prepare mentally for expected losing streaks
        - Set position sizing based on streak patterns
        - Identify when current streak is unusual
        - Plan for worst-case streak scenarios
        - Optimize strategy exit rules during bad streaks
        """
        if self.trades_df.empty:
            print("No trades available for streak analysis.")
            return None

        trades = self.trades_df.copy()
        trades["is_win"] = trades["net_pnl"] > 0

        # Calculate streaks
        streaks = []
        current_streak = 0
        current_type = None

        for _, trade in trades.iterrows():
            is_win = trade["is_win"]

            if current_type is None:
                current_type = is_win
                current_streak = 1
            elif current_type == is_win:
                current_streak += 1
            else:
                streaks.append(
                    {
                        "type": "Win" if current_type else "Loss",
                        "length": current_streak,
                    }
                )
                current_type = is_win
                current_streak = 1

        # Add final streak
        if current_streak > 0:
            streaks.append(
                {"type": "Win" if current_type else "Loss", "length": current_streak}
            )

        streak_df = pd.DataFrame(streaks)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Streak Length Distribution",
                "Streak Timeline",
                "Win vs Loss Streaks",
                "Streak Impact on Equity",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        if not streak_df.empty:
            # 1. Streak Length Distribution
            win_streaks = streak_df[streak_df["type"] == "Win"]["length"]
            loss_streaks = streak_df[streak_df["type"] == "Loss"]["length"]

            fig.add_trace(
                go.Histogram(
                    x=win_streaks,
                    name="Win Streaks",
                    marker_color="green",
                    opacity=0.7,
                    nbinsx=min(15, len(win_streaks)),
                    hovertemplate="<b>Win Streak Length:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Histogram(
                    x=loss_streaks,
                    name="Loss Streaks",
                    marker_color="red",
                    opacity=0.7,
                    nbinsx=min(15, len(loss_streaks)),
                    hovertemplate="<b>Loss Streak Length:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # 2. Streak Timeline
            streak_timeline = []
            for i, streak in enumerate(streaks):
                color = "green" if streak["type"] == "Win" else "red"
                streak_timeline.extend(
                    [streak["length"] if streak["type"] == "Win" else -streak["length"]]
                    * streak["length"]
                )

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(streak_timeline))),
                    y=streak_timeline,
                    name="Streak Timeline",
                    mode="lines+markers",
                    line=dict(color="blue", width=1),
                    marker=dict(size=3),
                    hovertemplate="<b>Trade #:</b> %{x}<br><b>Streak:</b> %{y}<extra></extra>",
                ),
                row=1,
                col=2,
            )

            # 3. Box plot comparison
            fig.add_trace(
                go.Box(
                    y=win_streaks,
                    name="Win Streaks",
                    marker_color="green",
                    boxmean=True,
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Box(
                    y=loss_streaks,
                    name="Loss Streaks",
                    marker_color="red",
                    boxmean=True,
                ),
                row=2,
                col=1,
            )

            # 4. Streak Impact on Equity
            if not self.results_df.empty and "equity" in self.results_df.columns:
                # Mark streak periods on equity curve
                equity_sample = self.results_df["equity"].iloc[
                    :: max(1, len(self.results_df) // 200)
                ]  # Sample for performance

                fig.add_trace(
                    go.Scatter(
                        x=equity_sample.index,
                        y=equity_sample,
                        name="Equity During Streaks",
                        line=dict(color="purple", width=2),
                        hovertemplate="<b>Date:</b> %{x}<br><b>Equity:</b> $%{y:,.2f}<extra></extra>",
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            title="<b>🔄 Win/Loss Streak Analysis</b>",
            template=self.template,
            height=700,
            showlegend=True,
        )

        return fig

    # ==================== 10. CORRELATION ANALYSIS ====================

    def plot_correlation_analysis(
        self, benchmark_data: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Analyzes correlation with market and other factors.

        WHAT IT TELLS YOU:
        - How correlated strategy is with market movements
        - Whether strategy provides diversification benefits
        - Performance in different market regimes
        - Relationship with volatility and other factors

        HOW IT HELPS:
        - Portfolio construction and allocation decisions
        - Risk management and hedging strategies
        - Understanding when strategy works best/worst
        - Diversification analysis for multi-strategy portfolios
        - Market regime detection and adaptation
        """
        if self.results_df.empty:
            print("No data available for correlation analysis.")
            return None

        df = self.results_df.copy()
        daily_returns = df["equity"].pct_change().dropna()

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Rolling Correlation Matrix",
                "Return vs Market Scatter",
                "Correlation Heatmap",
                "Beta Analysis",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # If benchmark data provided, use it; otherwise use buy_and_hold if available
        if benchmark_data is not None:
            market_returns = benchmark_data.pct_change().dropna()
        elif "buy_and_hold_equity" in df.columns:
            market_returns = df["buy_and_hold_equity"].pct_change().dropna()
        else:
            # Create synthetic market data for demonstration
            market_returns = pd.Series(
                np.random.normal(0.0005, 0.02, len(daily_returns)),
                index=daily_returns.index,
                name="Market",
            )

        # Align the series
        aligned_data = pd.concat([daily_returns, market_returns], axis=1).dropna()
        if aligned_data.shape[1] < 2:
            print("Insufficient data for correlation analysis.")
            return None

        strategy_ret = aligned_data.iloc[:, 0]
        market_ret = aligned_data.iloc[:, 1]

        # 1. Rolling Correlation
        rolling_corr = strategy_ret.rolling(60).corr(market_ret)

        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr,
                name="60D Correlation",
                line=dict(color="#4ecdc4", width=2),
                hovertemplate="<b>Date:</b> %{x}<br><b>Correlation:</b> %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add reference lines
        fig.add_hline(y=0, row=1, col=1, line_dash="dash", line_color="gray")
        fig.add_hline(
            y=0.5,
            row=1,
            col=1,
            line_dash="dot",
            line_color="orange",
            annotation_text="High Correlation",
        )
        fig.add_hline(y=-0.5, row=1, col=1, line_dash="dot", line_color="orange")

        # 2. Scatter plot
        fig.add_trace(
            go.Scatter(
                x=market_ret * 100,
                y=strategy_ret * 100,
                mode="markers",
                name="Daily Returns",
                marker=dict(
                    color=strategy_ret * 100, colorscale="RdYlGn", size=4, opacity=0.6
                ),
                hovertemplate="<b>Market:</b> %{x:.2f}%<br><b>Strategy:</b> %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Add regression line
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            market_ret, strategy_ret
        )
        line_x = np.array([market_ret.min(), market_ret.max()])
        line_y = slope * line_x + intercept

        fig.add_trace(
            go.Scatter(
                x=line_x * 100,
                y=line_y * 100,
                mode="lines",
                name=f"Regression (β={slope:.2f})",
                line=dict(color="red", width=2),
                hovertemplate=f"<b>Beta:</b> {slope:.2f}<br><b>R²:</b> {r_value**2:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # 3. Correlation Heatmap (if we have more data)
        # Create rolling correlations for different periods
        periods = [30, 60, 90]
        corr_matrix = []

        for period in periods:
            rolling_corr_period = strategy_ret.rolling(period).corr(market_ret)
            corr_matrix.append(
                [
                    rolling_corr_period.mean(),
                    rolling_corr_period.std(),
                    rolling_corr_period.min(),
                    rolling_corr_period.max(),
                ]
            )

        corr_df = pd.DataFrame(
            corr_matrix,
            columns=["Mean", "Std", "Min", "Max"],
            index=[f"{p}D" for p in periods],
        )

        fig.add_trace(
            go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale="RdBu",
                zmid=0,
                hovertemplate="<b>Period:</b> %{y}<br><b>Metric:</b> %{x}<br><b>Value:</b> %{z:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 4. Rolling Beta Analysis
        rolling_beta = (
            strategy_ret.rolling(60).cov(market_ret) / market_ret.rolling(60).var()
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_beta.index,
                y=rolling_beta,
                name="60D Beta",
                line=dict(color="#ff6b6b", width=2),
                hovertemplate="<b>Date:</b> %{x}<br><b>Beta:</b> %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        fig.add_hline(
            y=1,
            row=2,
            col=2,
            line_dash="dash",
            line_color="gray",
            annotation_text="Beta = 1.0",
        )

        fig.update_layout(
            title="<b>📊 Correlation & Market Relationship Analysis</b>",
            template=self.template,
            height=800,
            showlegend=True,
        )

        return fig

    # ==================== 11. CALENDAR HEATMAPS ====================

    def plot_calendar_heatmap(self) -> go.Figure:
        """
        Creates calendar heatmaps for returns analysis.

        WHAT IT TELLS YOU:
        - Seasonal patterns in strategy performance
        - Monthly/weekly performance consistency
        - Best/worst performing periods
        - Calendar-based risk patterns

        HOW IT HELPS:
        - Time-based position sizing adjustments
        - Identify seasonal opportunity windows
        - Plan for historically weak periods
        - Optimize strategy scheduling and maintenance
        - Set realistic monthly/quarterly targets
        """
        if self.results_df.empty:
            print("No data available for calendar analysis.")
            return None

        df = self.results_df.copy()
        daily_returns = df["equity"].pct_change().dropna()

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Monthly Returns Heatmap",
                "Day of Week Performance",
                "Hour of Day Performance",
                "Yearly Performance",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # 1. Monthly Returns Heatmap
        monthly_returns = (
            daily_returns.resample("M").apply(lambda x: (1 + x).prod() - 1) * 100
        )

        # Create month-year matrix
        monthly_data = []
        years = sorted(monthly_returns.index.year.unique())
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        heatmap_data = []
        hover_text = []

        for year in years:
            year_data = []
            year_hover = []
            for month in range(1, 13):
                try:
                    value = monthly_returns[
                        (monthly_returns.index.year == year)
                        & (monthly_returns.index.month == month)
                    ].iloc[0]
                    year_data.append(value)
                    year_hover.append(f"{months[month-1]} {year}: {value:.1f}%")
                except:
                    year_data.append(np.nan)
                    year_hover.append(f"{months[month-1]} {year}: No Data")

            heatmap_data.append(year_data)
            hover_text.append(year_hover)

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=months,
                y=[str(year) for year in years],
                colorscale="RdYlGn",
                zmid=0,
                text=hover_text,
                hovertemplate="<b>%{text}</b><extra></extra>",
                colorbar=dict(title="Return %"),
            ),
            row=1,
            col=1,
        )

        # 2. Day of Week Performance
        daily_returns_with_dow = daily_returns.copy()
        daily_returns_with_dow.index = pd.to_datetime(daily_returns_with_dow.index)
        dow_performance = (
            daily_returns_with_dow.groupby(
                daily_returns_with_dow.index.day_name()
            ).mean()
            * 100
        )

        # Reorder days
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        dow_performance = dow_performance.reindex(
            [day for day in day_order if day in dow_performance.index]
        )

        fig.add_trace(
            go.Bar(
                x=dow_performance.index,
                y=dow_performance.values,
                name="Avg Daily Return",
                marker_color=[
                    "green" if x > 0 else "red" for x in dow_performance.values
                ],
                hovertemplate="<b>%{x}:</b> %{y:.3f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # 3. Hour of Day Performance (if intraday data available)
        if hasattr(daily_returns.index, "hour"):
            hourly_performance = (
                daily_returns.groupby(daily_returns.index.hour).mean() * 100
            )

            fig.add_trace(
                go.Scatter(
                    x=hourly_performance.index,
                    y=hourly_performance.values,
                    mode="lines+markers",
                    name="Hourly Returns",
                    line=dict(color="blue", width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>Hour %{x}:</b> %{y:.3f}%<extra></extra>",
                ),
                row=2,
                col=1,
            )
        else:
            # Show monthly distribution instead
            monthly_dist = monthly_returns.groupby(monthly_returns.index.month).mean()

            fig.add_trace(
                go.Bar(
                    x=[months[i - 1] for i in monthly_dist.index],
                    y=monthly_dist.values,
                    name="Avg Monthly Return",
                    marker_color=[
                        "green" if x > 0 else "red" for x in monthly_dist.values
                    ],
                    hovertemplate="<b>%{x}:</b> %{y:.2f}%<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # 4. Yearly Performance
        yearly_returns = (
            daily_returns.resample("Y").apply(lambda x: (1 + x).prod() - 1) * 100
        )

        fig.add_trace(
            go.Bar(
                x=yearly_returns.index.year,
                y=yearly_returns.values,
                name="Annual Return",
                marker_color=[
                    "green" if x > 0 else "red" for x in yearly_returns.values
                ],
                hovertemplate="<b>%{x}:</b> %{y:.1f}%<extra></extra>",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="<b>📅 Calendar Performance Analysis</b>",
            template=self.template,
            height=800,
            showlegend=True,
        )

        return fig

    # ==================== 12. MARKET REGIME ANALYSIS ====================

    def plot_market_regime_analysis(self) -> go.Figure:
        """
        Analyzes performance across different market regimes.

        WHAT IT TELLS YOU:
        - How strategy performs in bull vs bear markets
        - Performance during high vs low volatility periods
        - Adaptation to changing market conditions
        - Risk characteristics in different environments

        HOW IT HELPS:
        - Adjust position sizing based on market regime
        - Identify when to pause/resume strategy
        - Prepare for regime transitions
        - Optimize strategy parameters for current regime
        - Build regime-aware risk management rules
        """
        if self.results_df.empty:
            print("No data available for regime analysis.")
            return None

        df = self.results_df.copy()
        daily_returns = df["equity"].pct_change().dropna()

        # Define market regimes
        # Volatility regime
        rolling_vol = daily_returns.rolling(30).std() * np.sqrt(252)
        vol_threshold = rolling_vol.median()
        high_vol_periods = rolling_vol > vol_threshold

        # Trend regime (using equity curve slope)
        rolling_trend = (
            df["equity"].rolling(30).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        )
        bull_periods = rolling_trend > 0

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Performance by Volatility Regime",
                "Performance by Trend Regime",
                "Regime Timeline",
                "Risk-Return by Regime",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # 1. Volatility Regime Performance
        high_vol_returns = (
            daily_returns[high_vol_periods].mean() * 252 * 100
        )  # Annualized
        low_vol_returns = daily_returns[~high_vol_periods].mean() * 252 * 100

        vol_regime_data = ["High Volatility", "Low Volatility"]
        vol_regime_returns = [high_vol_returns, low_vol_returns]

        fig.add_trace(
            go.Bar(
                x=vol_regime_data,
                y=vol_regime_returns,
                name="Volatility Regime Returns",
                marker_color=["red" if x < 0 else "green" for x in vol_regime_returns],
                hovertemplate="<b>%{x}:</b> %{y:.1f}% annual<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 2. Trend Regime Performance
        bull_returns = daily_returns[bull_periods].mean() * 252 * 100
        bear_returns = daily_returns[~bull_periods].mean() * 252 * 100

        trend_regime_data = ["Bull Market", "Bear Market"]
        trend_regime_returns = [bull_returns, bear_returns]

        fig.add_trace(
            go.Bar(
                x=trend_regime_data,
                y=trend_regime_returns,
                name="Trend Regime Returns",
                marker_color=[
                    "green" if x > 0 else "red" for x in trend_regime_returns
                ],
                hovertemplate="<b>%{x}:</b> %{y:.1f}% annual<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # 3. Regime Timeline
        regime_signal = pd.Series(0, index=df.index)
        regime_signal[high_vol_periods & bull_periods] = 3  # High vol bull
        regime_signal[high_vol_periods & ~bull_periods] = 2  # High vol bear
        regime_signal[~high_vol_periods & bull_periods] = 1  # Low vol bull
        regime_signal[~high_vol_periods & ~bull_periods] = 0  # Low vol bear

        colors = ["red", "orange", "yellow", "green"]
        regime_names = [
            "Low Vol Bear",
            "Low Vol Bull",
            "High Vol Bear",
            "High Vol Bull",
        ]

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=regime_signal,
                mode="lines",
                name="Market Regime",
                line=dict(color="purple", width=2),
                hovertemplate="<b>Date:</b> %{x}<br><b>Regime:</b> %{text}<extra></extra>",
                text=[regime_names[int(x)] for x in regime_signal],
            ),
            row=2,
            col=1,
        )

        # 4. Risk-Return Scatter by Regime
        regime_stats = []
        for i, regime_name in enumerate(regime_names):
            regime_mask = regime_signal == i
            if regime_mask.sum() > 5:  # Need minimum observations
                regime_returns = daily_returns[regime_mask]
                ann_return = regime_returns.mean() * 252 * 100
                ann_vol = regime_returns.std() * np.sqrt(252) * 100
                regime_stats.append((ann_return, ann_vol, regime_name, colors[i]))

        if regime_stats:
            returns, vols, names, colors_list = zip(*regime_stats)

            fig.add_trace(
                go.Scatter(
                    x=vols,
                    y=returns,
                    mode="markers+text",
                    text=names,
                    textposition="top center",
                    marker=dict(
                        size=15, color=colors_list, line=dict(width=2, color="black")
                    ),
                    name="Regime Risk-Return",
                    hovertemplate="<b>%{text}</b><br><b>Return:</b> %{y:.1f}%<br><b>Volatility:</b> %{x:.1f}%<extra></extra>",
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="<b>🌊 Market Regime Analysis</b>",
            template=self.template,
            height=800,
            showlegend=True,
        )

        # Update axis labels
        fig.update_yaxes(title_text="Annual Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Annual Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="Regime", row=2, col=1)
        fig.update_yaxes(title_text="Annual Return (%)", row=2, col=2)
        fig.update_xaxes(title_text="Annual Volatility (%)", row=2, col=2)

        return fig

    # ==================== 15. INTERACTIVE TRADE EXPLORER ====================

    def plot_interactive_trade_explorer(self) -> go.Figure:
        """
        Creates an interactive candlestick chart with detailed trade information.

        WHAT IT TELLS YOU:
        - Exact entry/exit points on price chart
        - Trade context and market conditions
        - Visual patterns in trade timing
        - Price action around trades

        HOW IT HELPS:
        - Debug strategy entry/exit logic
        - Identify optimal entry timing improvements
        - Spot patterns in successful vs failed trades
        - Visual strategy validation and refinement
        - Educational tool for understanding strategy behavior
        """
        if self.results_df.empty or self.trades_df.empty:
            print("No data available for trade explorer.")
            return None

        df = self.results_df.copy()
        trades = self.trades_df.copy()

        # Sample data if too large (for performance)
        if len(df) > 5000:
            df = df.iloc[:: len(df) // 5000]

        fig = go.Figure()

        # 1. Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            )
        )

        # 2. Add trade markers
        if not trades.empty:
            trades["open_time"] = pd.to_datetime(trades["open_time"])
            trades["close_time"] = pd.to_datetime(trades["close_time"])

            # Entry markers
            for _, trade in trades.iterrows():
                # Entry marker
                fig.add_trace(
                    go.Scatter(
                        x=[trade["open_time"]],
                        y=[trade["open_price"]],
                        mode="markers",
                        marker=dict(
                            symbol=(
                                "triangle-up"
                                if trade["side"] == "long"
                                else "triangle-down"
                            ),
                            size=12,
                            color="green" if trade["side"] == "long" else "red",
                            line=dict(width=2, color="white"),
                        ),
                        name=f"Entry {trade['side'].title()}",
                        hovertemplate=f"<b>Entry {trade['side'].title()}</b><br>"
                        f"<b>Date:</b> {trade['open_time']}<br>"
                        f"<b>Price:</b> ${trade['open_price']:.4f}<br>"
                        f"<b>Reason:</b> {trade.get('open_reason', 'Signal')}<extra></extra>",
                        showlegend=False,
                    )
                )

                # Exit marker
                exit_color = "green" if trade["net_pnl"] > 0 else "red"
                fig.add_trace(
                    go.Scatter(
                        x=[trade["close_time"]],
                        y=[trade["close_price"]],
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=10,
                            color=exit_color,
                            line=dict(width=2, color="white"),
                        ),
                        name=f"Exit",
                        hovertemplate=f"<b>Exit {trade['side'].title()}</b><br>"
                        f"<b>Date:</b> {trade['close_time']}<br>"
                        f"<b>Price:</b> ${trade['close_price']:.4f}<br>"
                        f"<b>P&L:</b> ${trade['net_pnl']:.2f} ({trade['net_pnl_pct']:.1f}%)<br>"
                        f"<b>Reason:</b> {trade.get('close_reason', 'Unknown')}<extra></extra>",
                        showlegend=False,
                    )
                )

                # Connect entry to exit with line
                fig.add_trace(
                    go.Scatter(
                        x=[trade["open_time"], trade["close_time"]],
                        y=[trade["open_price"], trade["close_price"]],
                        mode="lines",
                        line=dict(color=exit_color, width=1, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        # 3. Add equity overlay (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["equity"],
                name="Equity",
                line=dict(color="blue", width=2),
                yaxis="y2",
                hovertemplate="<b>Date:</b> %{x}<br><b>Equity:</b> $%{y:,.2f}<extra></extra>",
            )
        )

        # Layout with secondary y-axis
        fig.update_layout(
            title="<b>🔍 Interactive Trade Explorer</b><br>"
            "<sub>Hover over markers for trade details. Use zoom and pan to explore.</sub>",
            template=self.template,
            height=600,
            xaxis=dict(title="Date", rangeslider=dict(visible=False)),
            yaxis=dict(title="Price ($)", side="left"),
            yaxis2=dict(title="Equity ($)", side="right", overlaying="y"),
            hovermode="x unified",
            showlegend=True,
        )

        return fig

    # ==================== 16. PARAMETER SENSITIVITY ANALYSIS ====================

    def plot_parameter_sensitivity(
        self, param_results: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """
        Visualizes parameter sensitivity analysis results.

        WHAT IT TELLS YOU:
        - Which parameters most affect strategy performance
        - Optimal parameter ranges and stability
        - Risk of overfitting to specific values
        - Robustness of parameter choices

        HOW IT HELPS:
        - Choose robust parameter values
        - Identify critical vs non-critical parameters
        - Set parameter ranges for optimization
        - Avoid overfitting in strategy development
        - Build confidence in parameter choices

        Args:
            param_results: Dict of {param_name: {param_value: metric_value}}
        """
        if not param_results:
            print("No parameter sensitivity data provided.")
            return None

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Parameter Impact on Returns",
                "Parameter Impact on Sharpe",
                "Parameter Stability Analysis",
                "3D Parameter Surface",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "surface"}],
            ],
        )

        colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffd93d"]

        # 1. Parameter Impact on Returns
        for i, (param_name, param_data) in enumerate(param_results.items()):
            if len(param_data) > 1:
                param_values = list(param_data.keys())
                returns = [param_data[val].get("roi_pct", 0) for val in param_values]

                fig.add_trace(
                    go.Scatter(
                        x=param_values,
                        y=returns,
                        mode="lines+markers",
                        name=f"{param_name} ROI",
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6),
                        hovertemplate=f"<b>{param_name}:</b> %{{x}}<br><b>ROI:</b> %{{y:.1f}}%<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # 2. Parameter Impact on Sharpe
        for i, (param_name, param_data) in enumerate(param_results.items()):
            if len(param_data) > 1:
                param_values = list(param_data.keys())
                sharpe = [
                    param_data[val].get("sharpe_ratio", 0) for val in param_values
                ]

                fig.add_trace(
                    go.Scatter(
                        x=param_values,
                        y=sharpe,
                        mode="lines+markers",
                        name=f"{param_name} Sharpe",
                        line=dict(color=colors[i % len(colors)], width=2, dash="dash"),
                        marker=dict(size=6),
                        hovertemplate=f"<b>{param_name}:</b> %{{x}}<br><b>Sharpe:</b> %{{y:.2f}}<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )

        # 3. Parameter Stability (coefficient of variation)
        stability_data = []
        param_names = []

        for param_name, param_data in param_results.items():
            if len(param_data) > 2:
                returns = [
                    param_data[val].get("roi_pct", 0) for val in param_data.keys()
                ]
                cv = (
                    np.std(returns) / np.mean(returns)
                    if np.mean(returns) != 0
                    else float("inf")
                )
                stability_data.append(cv)
                param_names.append(param_name)

        if stability_data:
            fig.add_trace(
                go.Bar(
                    x=param_names,
                    y=stability_data,
                    name="Parameter Stability",
                    marker_color=[
                        "red" if x > 1 else "orange" if x > 0.5 else "green"
                        for x in stability_data
                    ],
                    hovertemplate="<b>%{x}:</b><br><b>Coefficient of Variation:</b> %{y:.2f}<br><b>Stability:</b> %{text}<extra></extra>",
                    text=[
                        "Low" if x > 1 else "Medium" if x > 0.5 else "High"
                        for x in stability_data
                    ],
                ),
                row=2,
                col=1,
            )

        # 4. 3D Surface (if we have 2+ parameters)
        param_names_list = list(param_results.keys())
        if len(param_names_list) >= 2:
            # Create a simple 3D surface using the first two parameters
            param1_name = param_names_list[0]
            param2_name = param_names_list[1]

            param1_values = list(param_results[param1_name].keys())
            param2_values = list(param_results[param2_name].keys())

            # Create meshgrid for surface
            if len(param1_values) > 1 and len(param2_values) > 1:
                X, Y = np.meshgrid(
                    param1_values[:5], param2_values[:5]
                )  # Limit for performance
                Z = (
                    np.random.random(X.shape) * 100
                )  # Placeholder - in real use, calculate actual results

                fig.add_trace(
                    go.Surface(
                        x=X,
                        y=Y,
                        z=Z,
                        colorscale="Viridis",
                        name="Parameter Surface",
                        hovertemplate=f"<b>{param1_name}:</b> %{{x}}<br><b>{param2_name}:</b> %{{y}}<br><b>Performance:</b> %{{z:.1f}}<extra></extra>",
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            title="<b>🔧 Parameter Sensitivity Analysis</b>",
            template=self.template,
            height=800,
            showlegend=True,
        )

        return fig

    # ==================== 18. ANIMATED ANALYSIS ====================

    def create_animated_equity_buildup(self, speed: int = 50) -> go.Figure:
        """
        Creates animated equity curve showing strategy development over time.

        WHAT IT TELLS YOU:
        - How equity builds up trade by trade
        - Timing of major gains/losses
        - Strategy momentum and consistency
        - Visual story of strategy performance

        HOW IT HELPS:
        - Presentation and education tool
        - Identify key performance inflection points
        - Understand strategy behavior patterns
        - Communicate strategy story to stakeholders
        - Motivation and confidence building

        Args:
            speed: Animation speed in milliseconds between frames
        """
        if self.results_df.empty:
            print("No data available for animation.")
            return None

        df = self.results_df.copy()

        # Sample data for performance (animations can be slow with too much data)
        if len(df) > 500:
            step = len(df) // 500
            df = df.iloc[::step]

        # Create frames for animation
        frames = []
        for i in range(1, len(df) + 1):
            frame_data = df.iloc[:i]

            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(
                            x=frame_data.index,
                            y=frame_data["equity"],
                            mode="lines",
                            line=dict(color="#00ff88", width=3),
                            name="Equity Growth",
                        ),
                        go.Scatter(
                            x=[frame_data.index[-1]],
                            y=[frame_data["equity"].iloc[-1]],
                            mode="markers",
                            marker=dict(size=10, color="yellow", symbol="star"),
                            name="Current Position",
                        ),
                    ],
                    name=str(frame_data.index[-1].date()),
                )
            )

        # Initial figure
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df.index[:1],
                    y=df["equity"][:1],
                    mode="lines",
                    line=dict(color="#00ff88", width=3),
                    name="Equity Growth",
                )
            ],
            frames=frames,
        )

        # Add animation controls
        fig.update_layout(
            title="<b>🎬 Animated Strategy Performance</b><br>"
            "<sub>Watch your equity grow over time</sub>",
            template=self.template,
            height=600,
            xaxis=dict(title="Date", range=[df.index.min(), df.index.max()]),
            yaxis=dict(
                title="Equity ($)",
                range=[df["equity"].min() * 0.95, df["equity"].max() * 1.05],
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": speed, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": speed // 2},
                                },
                            ],
                        },
                        {
                            "label": "⏸ Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [f.name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": f.name,
                            "method": "animate",
                        }
                        for f in frames
                    ],
                    "active": 0,
                    "currentvalue": {"prefix": "Date: "},
                    "len": 0.9,
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )

        return fig

    # ==================== 19. MONTE CARLO ANALYSIS ====================

    def plot_monte_carlo_analysis(
        self, n_simulations: int = 1000, forecast_days: int = 252
    ) -> go.Figure:
        """
        Creates Monte Carlo simulation for strategy performance projections.

        WHAT IT TELLS YOU:
        - Possible future performance scenarios
        - Confidence intervals for projections
        - Worst-case and best-case outcomes
        - Probability of specific performance levels

        HOW IT HELPS:
        - Risk assessment and planning
        - Set realistic expectations
        - Position sizing for worst-case scenarios
        - Communicate uncertainty to stakeholders
        - Stress testing and scenario planning

        Args:
            n_simulations: Number of Monte Carlo simulations
            forecast_days: Number of days to forecast
        """
        if self.results_df.empty:
            print("No data available for Monte Carlo analysis.")
            return None

        df = self.results_df.copy()
        daily_returns = df["equity"].pct_change().dropna()

        if len(daily_returns) < 30:
            print("Insufficient data for Monte Carlo analysis.")
            return None

        # Calculate return statistics
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        current_equity = df["equity"].iloc[-1]

        # Generate Monte Carlo simulations
        np.random.seed(42)  # For reproducible results
        simulations = []

        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, forecast_days)

            # Calculate equity path
            equity_path = [current_equity]
            for ret in random_returns:
                equity_path.append(equity_path[-1] * (1 + ret))

            simulations.append(equity_path[1:])  # Exclude starting point

        simulations = np.array(simulations)

        # Create future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D"
        )

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Monte Carlo Simulation Paths",
                "Confidence Intervals",
                "Final Equity Distribution",
                "Probability Analysis",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # 1. Sample simulation paths
        sample_size = min(100, n_simulations)
        for i in range(0, sample_size, max(1, sample_size // 20)):  # Show ~20 paths
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=simulations[i],
                    mode="lines",
                    line=dict(color="lightblue", width=0.5),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        # Add historical equity
        fig.add_trace(
            go.Scatter(
                x=df.index[-252:],  # Last year
                y=df["equity"].iloc[-252:],
                mode="lines",
                line=dict(color="black", width=2),
                name="Historical Equity",
                hovertemplate="<b>Historical:</b> $%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # ==================== 20. MACHINE LEARNING INSIGHTS ====================

    def plot_ml_insights(self) -> go.Figure:
        """
        Provides machine learning-based insights into trading patterns.

        WHAT IT TELLS YOU:
        - Hidden patterns in successful vs failed trades
        - Feature importance for trade outcomes
        - Clustering of similar trading conditions
        - Anomaly detection in performance

        HOW IT HELPS:
        - Identify key factors driving trade success
        - Improve entry/exit timing rules
        - Detect unusual market conditions
        - Optimize strategy parameters using data science
        - Build predictive models for trade outcomes
        """
        if self.trades_df.empty or len(self.trades_df) < 20:
            print("Insufficient trade data for ML analysis.")
            return None

        trades = self.trades_df.copy()

        # Prepare features for ML analysis
        trades["trade_duration_hours"] = (
            pd.to_datetime(trades["close_time"]) - pd.to_datetime(trades["open_time"])
        ).dt.total_seconds() / 3600

        trades["is_profitable"] = trades["net_pnl"] > 0
        trades["hour_opened"] = pd.to_datetime(trades["open_time"]).dt.hour
        trades["day_of_week"] = pd.to_datetime(trades["open_time"]).dt.dayofweek
        trades["month"] = pd.to_datetime(trades["open_time"]).dt.month

        # Calculate additional features
        trades["price_change_pct"] = (
            (trades["close_price"] - trades["open_price"]) / trades["open_price"] * 100
        )
        trades["leverage_used"] = trades.get("leverage", 1)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Feature Importance for Trade Success",
                "Trade Clustering Analysis",
                "Anomaly Detection",
                "Success Rate by Conditions",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # 1. Feature Importance (simplified correlation analysis)
        numeric_features = [
            "trade_duration_hours",
            "hour_opened",
            "day_of_week",
            "month",
            "open_price",
            "leverage_used",
        ]

        # Calculate correlation with profitability
        feature_importance = {}
        for feature in numeric_features:
            if feature in trades.columns:
                correlation = abs(
                    trades[feature].corr(trades["is_profitable"].astype(int))
                )
                feature_importance[feature] = (
                    correlation if not pd.isna(correlation) else 0
                )

        if feature_importance:
            features, importances = zip(
                *sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            fig.add_trace(
                go.Bar(
                    x=list(features),
                    y=list(importances),
                    name="Feature Importance",
                    marker_color=[
                        "darkgreen" if x > 0.2 else "orange" if x > 0.1 else "lightblue"
                        for x in importances
                    ],
                    hovertemplate="<b>%{x}:</b><br><b>Correlation:</b> %{y:.3f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # 2. Trade Clustering (using simple 2D projection)
        if len(trades) >= 10:
            # Simple clustering based on duration and return
            x_data = trades["trade_duration_hours"].fillna(
                trades["trade_duration_hours"].median()
            )
            y_data = trades["net_pnl_pct"].fillna(0)

            # Simple clustering: profitable vs unprofitable
            colors = [
                "green" if profit else "red" for profit in trades["is_profitable"]
            ]

            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="markers",
                    marker=dict(
                        color=colors,
                        size=8,
                        opacity=0.6,
                        line=dict(width=1, color="white"),
                    ),
                    name="Trade Clusters",
                    hovertemplate="<b>Duration:</b> %{x:.1f}h<br><b>Return:</b> %{y:.1f}%<br>"
                    "<b>Side:</b> %{text}<extra></extra>",
                    text=trades["side"],
                ),
                row=1,
                col=2,
            )

        # 3. Anomaly Detection (outlier identification)
        # Identify trades with unusual characteristics
        if len(trades) >= 10:
            # Calculate z-scores for returns
            returns = trades["net_pnl_pct"].fillna(0)
            z_scores = abs((returns - returns.mean()) / returns.std())

            # Identify outliers (z-score > 2)
            outliers = trades[z_scores > 2]
            normal_trades = trades[z_scores <= 2]

            # Plot normal trades
            fig.add_trace(
                go.Scatter(
                    x=normal_trades.index,
                    y=normal_trades["net_pnl_pct"],
                    mode="markers",
                    marker=dict(color="blue", size=6, opacity=0.6),
                    name="Normal Trades",
                    hovertemplate="<b>Trade #:</b> %{x}<br><b>Return:</b> %{y:.1f}%<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Plot outliers
            if not outliers.empty:
                fig.add_trace(
                    go.Scatter(
                        x=outliers.index,
                        y=outliers["net_pnl_pct"],
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="diamond"),
                        name="Anomalous Trades",
                        hovertemplate="<b>Outlier Trade #:</b> %{x}<br><b>Return:</b> %{y:.1f}%<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

        # 4. Success Rate by Conditions
        # Analyze success rate by different conditions
        condition_analysis = {}

        # Success rate by hour
        hourly_success = trades.groupby("hour_opened")["is_profitable"].mean() * 100
        condition_analysis["hour"] = hourly_success

        # Success rate by day of week
        daily_success = trades.groupby("day_of_week")["is_profitable"].mean() * 100
        condition_analysis["day"] = daily_success

        # Plot hourly success rate
        if not hourly_success.empty:
            fig.add_trace(
                go.Scatter(
                    x=hourly_success.index,
                    y=hourly_success.values,
                    mode="lines+markers",
                    name="Hourly Success Rate",
                    line=dict(color="purple", width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>Hour:</b> %{x}<br><b>Success Rate:</b> %{y:.1f}%<extra></extra>",
                ),
                row=2,
                col=2,
            )

        # Add reference line at 50%
        fig.add_hline(
            y=50,
            row=2,
            col=2,
            line_dash="dash",
            line_color="gray",
            annotation_text="50% Success Rate",
        )

        # Calculate ML insights summary
        total_trades = len(trades)
        success_rate = trades["is_profitable"].mean() * 100
        avg_winner = trades[trades["is_profitable"]]["net_pnl_pct"].mean()
        avg_loser = trades[~trades["is_profitable"]]["net_pnl_pct"].mean()

        fig.update_layout(
            title=f"<b>🤖 Machine Learning Trading Insights</b><br>"
            f"<sub>Trades: {total_trades} | Success Rate: {success_rate:.1f}% | "
            f"Avg Winner: {avg_winner:.1f}% | Avg Loser: {avg_loser:.1f}%</sub>",
            template=self.template,
            height=800,
            showlegend=True,
        )

        # Update axis labels
        fig.update_xaxes(title_text="Features", row=1, col=1)
        fig.update_yaxes(title_text="Correlation", row=1, col=1)
        fig.update_xaxes(title_text="Duration (hours)", row=1, col=2)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
        fig.update_xaxes(title_text="Trade Number", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
        fig.update_yaxes(title_text="Success Rate (%)", row=2, col=2)

        return fig

    # ==================== DASHBOARD CREATION ====================

    def create_complete_graph_dashboard(
        self, save_path: Optional[str] = None, param_results: Optional[Dict] = None
    ) -> Dict[str, go.Figure]:
        """
        Create a complete dashboard with all graph visualizations.

        Args:
            save_path: Optional directory path to save HTML files
            param_results: Optional parameter sensitivity data

        Returns:
            Dictionary of figure names and plotly figures
        """
        dashboard_figures = {}

        print("🚀 Creating Comprehensive Graph Analysis Dashboard...")

        # 1. Equity with Drawdown
        print("  📈 Creating equity curve with drawdown...")
        dashboard_figures["equity_drawdown"] = self.plot_equity_with_drawdown()

        # 2. Rolling Performance
        print("  📊 Creating rolling performance metrics...")
        dashboard_figures["rolling_performance"] = self.plot_rolling_performance()

        # 3. Cumulative Returns Comparison
        print("  📈 Creating cumulative returns comparison...")
        dashboard_figures["cumulative_returns"] = (
            self.plot_cumulative_returns_comparison()
        )

        # 6. Streak Analysis
        print("  🔄 Creating streak analysis...")
        dashboard_figures["streak_analysis"] = self.plot_streak_analysis()

        # 10. Correlation Analysis
        print("  📊 Creating correlation analysis...")
        dashboard_figures["correlation_analysis"] = self.plot_correlation_analysis()

        # 11. Calendar Heatmap
        print("  📅 Creating calendar heatmap...")
        dashboard_figures["calendar_heatmap"] = self.plot_calendar_heatmap()

        # 12. Market Regime Analysis
        print("  🌊 Creating market regime analysis...")
        dashboard_figures["market_regime"] = self.plot_market_regime_analysis()

        # 15. Interactive Trade Explorer
        print("  🔍 Creating interactive trade explorer...")
        dashboard_figures["trade_explorer"] = self.plot_interactive_trade_explorer()

        # 16. Parameter Sensitivity (if data provided)
        if param_results:
            print("  🔧 Creating parameter sensitivity analysis...")
            dashboard_figures["parameter_sensitivity"] = (
                self.plot_parameter_sensitivity(param_results)
            )

        # 18. Animated Analysis
        print("  🎬 Creating animated equity buildup...")
        dashboard_figures["animated_equity"] = self.create_animated_equity_buildup()

        # 19. Monte Carlo Analysis
        print("  🎲 Creating Monte Carlo analysis...")
        dashboard_figures["monte_carlo"] = self.plot_monte_carlo_analysis()

        # 20. ML Insights
        print("  🤖 Creating ML insights...")
        dashboard_figures["ml_insights"] = self.plot_ml_insights()

        # Save figures if path provided
        if save_path:
            import os

            os.makedirs(save_path, exist_ok=True)

            for name, fig in dashboard_figures.items():
                if fig is not None:
                    file_path = os.path.join(save_path, f"{name}.html")
                    fig.write_html(file_path)
                    print(f"  💾 Saved {name} to {file_path}")

        print("✅ Graph dashboard creation completed!")
        return dashboard_figures

    def show_graph_dashboard(self, param_results: Optional[Dict] = None):
        """Display all dashboard graphs in sequence."""
        figures = self.create_complete_graph_dashboard(param_results=param_results)

        for name, fig in figures.items():
            if fig is not None:
                print(f"\n{'='*60}")
                print(f"Displaying: {name.replace('_', ' ').title()}")
                print(f"{'='*60}")
                fig.show()
