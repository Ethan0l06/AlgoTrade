# AlgoTrade/Analysis/TablePlots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from AlgoTrade.Analysis.BacktestAnalysis import BacktestAnalysis


class AnalysisTablePlotter:
    """
    Enhanced table plotting class for comprehensive trading strategy analysis.
    Provides both static (matplotlib/seaborn) and interactive (plotly) visualizations.
    """
    
    def __init__(self, analysis: 'BacktestAnalysis', style: str = "dark"):
        """
        Initialize the table plotter.
        
        Args:
            analysis: BacktestAnalysis object containing backtest results
            style: Visual style - "dark", "light", or "colorful"
        """
        self.analysis = analysis
        self.results_df = analysis.results_df
        self.trades_df = analysis.trades
        self.metrics = analysis.metrics
        self.style = style
        
        # Set style configurations
        self._configure_style()
    
    def _configure_style(self):
        """Configure visual styling based on selected theme."""
        if self.style == "dark":
            plt.style.use('dark_background')
            self.bg_color = '#1e1e1e'
            self.text_color = 'white'
            self.accent_color = '#00ff88'
            self.loss_color = '#ff4444'
            self.profit_color = '#44ff44'
        elif self.style == "light":
            plt.style.use('default')
            self.bg_color = 'white'
            self.text_color = 'black'
            self.accent_color = '#2E86C1'
            self.loss_color = '#E74C3C'
            self.profit_color = '#27AE60'
        else:  # colorful
            self.bg_color = '#f8f9fa'
            self.text_color = '#2c3e50'
            self.accent_color = '#3498db'
            self.loss_color = '#e74c3c'
            self.profit_color = '#2ecc71'

    # ==================== PERFORMANCE SUMMARY TABLES ====================
    
    def create_performance_summary_table(self, save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive performance summary table with key metrics.
        """
        # Prepare data for the table
        metrics_data = []
        
        # Financial Performance
        metrics_data.extend([
            ["📊 FINANCIAL PERFORMANCE", "", ""],
            ["Initial Balance", f"${self.metrics.get('initial_balance', 0):,.2f}", "Starting capital"],
            ["Final Balance", f"${self.metrics.get('final_balance', 0):,.2f}", "Ending capital"],
            ["Final Equity", f"${self.metrics.get('final_equity', 0):,.2f}", "Total equity including unrealized P&L"],
            ["Total ROI", f"{self.metrics.get('roi_pct', 0):.2f}%", "Return on investment"],
            ["Total Profit", f"${self.metrics.get('total_profits', 0):,.2f}", "Sum of all profitable trades"],
            ["Total Loss", f"${self.metrics.get('total_losses', 0):,.2f}", "Sum of all losing trades"],
            ["Net P&L", f"${self.metrics.get('total_profits', 0) - self.metrics.get('total_losses', 0):,.2f}", "Total profit minus total loss"],
        ])
        
        # Risk Metrics
        metrics_data.extend([
            ["", "", ""],
            ["⚠️ RISK METRICS", "", ""],
            ["Max Drawdown", f"{self.metrics.get('max_drawdown_equity', 0):.2%}", "Maximum equity decline from peak"],
            ["Sharpe Ratio", f"{self.metrics.get('sharpe_ratio', 0):.3f}", "Risk-adjusted return measure"],
            ["Sortino Ratio", f"{self.metrics.get('sortino_ratio', 0):.3f}", "Downside risk-adjusted return"],
            ["Calmar Ratio", f"{self.metrics.get('calmar_ratio', 0):.3f}", "Annual return / max drawdown"],
            ["Profit Factor", f"{self.metrics.get('profit_factor', 0):.2f}", "Gross profit / gross loss"],
        ])
        
        # Trading Activity
        metrics_data.extend([
            ["", "", ""],
            ["📈 TRADING ACTIVITY", "", ""],
            ["Total Trades", f"{self.metrics.get('total_trades', 0):,}", "Total number of completed trades"],
            ["Winning Trades", f"{self.metrics.get('total_good_trades', 0):,}", "Number of profitable trades"],
            ["Losing Trades", f"{self.metrics.get('total_bad_trades', 0):,}", "Number of losing trades"],
            ["Win Rate", f"{self.metrics.get('global_win_rate', 0):.2%}", "Percentage of winning trades"],
            ["Avg Win", f"{self.metrics.get('avg_pnl_pct_good_trades', 0):.2f}%", "Average return of winning trades"],
            ["Avg Loss", f"{self.metrics.get('avg_pnl_pct_bad_trades', 0):.2f}%", "Average return of losing trades"],
        ])
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data, columns=["Metric", "Value", "Description"])
        
        # Create color mapping for values
        colors = []
        for _, row in df.iterrows():
            if row['Metric'] in ["📊 FINANCIAL PERFORMANCE", "⚠️ RISK METRICS", "📈 TRADING ACTIVITY"]:
                colors.append(['lightblue', 'lightblue', 'lightblue'])
            elif row['Metric'] == "":
                colors.append(['white', 'white', 'white'])
            else:
                # Color code based on value performance
                value_str = row['Value']
                if '%' in value_str and value_str.replace('%', '').replace('-', '').replace('.', '').isdigit():
                    val = float(value_str.replace('%', ''))
                    if val > 0:
                        colors.append(['white', 'lightgreen', 'white'])
                    elif val < 0:
                        colors.append(['white', 'lightcoral', 'white'])
                    else:
                        colors.append(['white', 'white', 'white'])
                else:
                    colors.append(['white', 'white', 'white'])
        
        # Create Plotly table
        fig = go.Figure(data=[go.Table(
            columnwidth=[200, 150, 300],
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>', '<b>Description</b>'],
                fill_color='darkblue',
                font=dict(color='white', size=14),
                align='left',
                height=40
            ),
            cells=dict(
                values=[df['Metric'], df['Value'], df['Description']],
                fill_color=colors,
                font=dict(color='black', size=12),
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="<b>📊 Strategy Performance Summary</b>",
                x=0.5,
                font=dict(size=20)
            ),
            width=800,
            height=600,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

    def create_trades_analysis_table(self, top_n: int = 10) -> go.Figure:
        """
        Create a detailed trades analysis table showing best/worst trades.
        """
        if self.trades_df.empty:
            print("No trades data available for analysis.")
            return None
        
        # Get top winning and losing trades
        top_winners = self.trades_df.nlargest(top_n, 'net_pnl_pct')
        top_losers = self.trades_df.nsmallest(top_n, 'net_pnl_pct')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("🏆 Top Winning Trades", "💔 Top Losing Trades"),
            specs=[[{"type": "table"}], [{"type": "table"}]],
            vertical_spacing=0.1
        )
        
        # Prepare data for top winners
        winners_data = []
        for _, trade in top_winners.iterrows():
            duration = pd.to_datetime(trade['close_time']) - pd.to_datetime(trade['open_time'])
            winners_data.append([
                trade['side'].upper(),
                f"${trade['open_price']:.4f}",
                f"${trade['close_price']:.4f}",
                f"{trade['net_pnl_pct']:.2f}%",
                f"${trade['net_pnl']:.2f}",
                f"{duration.total_seconds() / 3600:.1f}h",
                trade['close_reason']
            ])
        
        # Prepare data for top losers
        losers_data = []
        for _, trade in top_losers.iterrows():
            duration = pd.to_datetime(trade['close_time']) - pd.to_datetime(trade['open_time'])
            losers_data.append([
                trade['side'].upper(),
                f"${trade['open_price']:.4f}",
                f"${trade['close_price']:.4f}",
                f"{trade['net_pnl_pct']:.2f}%",
                f"${trade['net_pnl']:.2f}",
                f"{duration.total_seconds() / 3600:.1f}h",
                trade['close_reason']
            ])
        
        headers = ['Side', 'Entry Price', 'Exit Price', 'Return %', 'P&L $', 'Duration', 'Exit Reason']
        
        # Add winners table
        fig.add_trace(
            go.Table(
                columnwidth=[60, 80, 80, 80, 80, 70, 120],
                header=dict(
                    values=[f'<b>{h}</b>' for h in headers],
                    fill_color='darkgreen',
                    font=dict(color='white', size=11),
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*winners_data)) if winners_data else [[] for _ in headers],
                    fill_color='lightgreen',
                    font=dict(color='black', size=10),
                    align='center'
                )
            ),
            row=1, col=1
        )
        
        # Add losers table
        fig.add_trace(
            go.Table(
                columnwidth=[60, 80, 80, 80, 80, 70, 120],
                header=dict(
                    values=[f'<b>{h}</b>' for h in headers],
                    fill_color='darkred',
                    font=dict(color='white', size=11),
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*losers_data)) if losers_data else [[] for _ in headers],
                    fill_color='lightcoral',
                    font=dict(color='black', size=10),
                    align='center'
                )
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=dict(
                text="<b>📋 Detailed Trades Analysis</b>",
                x=0.5,
                font=dict(size=18)
            ),
            height=700,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        return fig

    def create_monthly_performance_table(self) -> go.Figure:
        """
        Create a monthly performance breakdown table.
        """
        if self.results_df.empty:
            print("No results data available for monthly analysis.")
            return None
        
        # Prepare monthly data
        monthly_data = self.results_df.copy()
        monthly_data['year_month'] = monthly_data.index.to_period('M')
        
        # Calculate monthly returns
        monthly_returns = monthly_data.groupby('year_month').agg({
            'equity': ['first', 'last', 'max', 'min'],
            'balance': ['first', 'last'],
        }).round(2)
        
        monthly_returns.columns = ['equity_start', 'equity_end', 'equity_max', 'equity_min', 
                                 'balance_start', 'balance_end']
        
        # Calculate monthly metrics
        monthly_returns['monthly_return_pct'] = (
            (monthly_returns['equity_end'] - monthly_returns['equity_start']) / 
            monthly_returns['equity_start'] * 100
        ).round(2)
        
        monthly_returns['monthly_drawdown_pct'] = (
            (monthly_returns['equity_max'] - monthly_returns['equity_min']) / 
            monthly_returns['equity_max'] * 100
        ).round(2)
        
        # Prepare table data
        table_data = []
        for period, row in monthly_returns.iterrows():
            return_pct = row['monthly_return_pct']
            
            # Color coding for returns
            if return_pct > 5:
                return_color = 'darkgreen'
            elif return_pct > 0:
                return_color = 'lightgreen'
            elif return_pct > -5:
                return_color = 'yellow'
            else:
                return_color = 'lightcoral'
            
            table_data.append([
                str(period),
                f"${row['equity_start']:,.0f}",
                f"${row['equity_end']:,.0f}",
                f"{return_pct:+.2f}%",
                f"{row['monthly_drawdown_pct']:.2f}%",
                return_color
            ])
        
        if not table_data:
            print("No monthly data available.")
            return None
        
        # Separate data and colors
        data_without_colors = [row[:-1] for row in table_data]
        colors = [row[-1] for row in table_data]
        
        # Create DataFrame for easier handling
        df_monthly = pd.DataFrame(data_without_colors, 
                                columns=['Month', 'Start Equity', 'End Equity', 'Return %', 'Max DD %'])
        
        # Create cell colors matrix
        cell_colors = []
        for color in colors:
            cell_colors.append(['white', 'white', 'white', color, 'white'])
        
        fig = go.Figure(data=[go.Table(
            columnwidth=[100, 120, 120, 100, 100],
            header=dict(
                values=['<b>Month</b>', '<b>Start Equity</b>', '<b>End Equity</b>', 
                       '<b>Monthly Return</b>', '<b>Max Drawdown</b>'],
                fill_color='darkblue',
                font=dict(color='white', size=12),
                align='center',
                height=40
            ),
            cells=dict(
                values=[df_monthly[col] for col in df_monthly.columns],
                fill_color=list(zip(*cell_colors)),
                font=dict(color='black', size=11),
                align='center',
                height=35
            )
        )])
        
        # Add summary statistics
        total_return = monthly_returns['monthly_return_pct'].sum()
        avg_monthly_return = monthly_returns['monthly_return_pct'].mean()
        win_rate = (monthly_returns['monthly_return_pct'] > 0).mean() * 100
        
        fig.update_layout(
            title=dict(
                text=f"<b>📅 Monthly Performance Breakdown</b><br>"
                     f"<sub>Total Return: {total_return:+.1f}% | "
                     f"Avg Monthly: {avg_monthly_return:+.1f}% | "
                     f"Win Rate: {win_rate:.0f}%</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            height=min(600, len(table_data) * 40 + 200),
            margin=dict(l=20, r=20, t=100, b=20)
        )
        
        return fig

    def create_risk_metrics_table(self) -> go.Figure:
        """
        Create a comprehensive risk metrics analysis table.
        """
        # Calculate additional risk metrics
        if self.results_df.empty or 'equity' not in self.results_df.columns:
            print("No equity data available for risk analysis.")
            return None
        
        equity_series = self.results_df['equity']
        returns = equity_series.pct_change().dropna()
        
        # Risk calculations
        risk_metrics = {}
        
        # Basic metrics
        risk_metrics['Total Return'] = f"{(equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100:.2f}%"
        risk_metrics['Annualized Return'] = f"{self._calculate_annualized_return():.2f}%"
        risk_metrics['Volatility (Daily)'] = f"{returns.std() * 100:.2f}%"
        risk_metrics['Volatility (Annualized)'] = f"{returns.std() * np.sqrt(252) * 100:.2f}%"
        
        # Drawdown metrics
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        risk_metrics['Maximum Drawdown'] = f"{drawdown.min() * 100:.2f}%"
        risk_metrics['Average Drawdown'] = f"{drawdown[drawdown < 0].mean() * 100:.2f}%"
        
        # Recovery metrics
        underwater_periods = self._calculate_underwater_periods(drawdown)
        risk_metrics['Longest Drawdown Period'] = f"{underwater_periods.max()} days" if len(underwater_periods) > 0 else "0 days"
        
        # Calculate recovery factor properly
        total_return_pct = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
        max_dd_pct = abs(drawdown.min() * 100)
        recovery_factor = abs(total_return_pct) / max_dd_pct if max_dd_pct != 0 else 0
        risk_metrics['Recovery Factor'] = f"{recovery_factor:.2f}"
        
        # Risk ratios (from existing metrics)
        risk_metrics['Sharpe Ratio'] = f"{self.metrics.get('sharpe_ratio', 0):.3f}"
        risk_metrics['Sortino Ratio'] = f"{self.metrics.get('sortino_ratio', 0):.3f}"
        risk_metrics['Calmar Ratio'] = f"{self.metrics.get('calmar_ratio', 0):.3f}"
        
        # VaR calculations
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        risk_metrics['Value at Risk (95%)'] = f"{var_95:.2f}%"
        risk_metrics['Value at Risk (99%)'] = f"{var_99:.2f}%"
        
        # Expected Shortfall (CVaR)
        es_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        risk_metrics['Expected Shortfall (95%)'] = f"{es_95:.2f}%"
        
        # Prepare data for table
        table_data = []
        categories = {
            '📈 RETURN METRICS': ['Total Return', 'Annualized Return'],
            '📊 VOLATILITY METRICS': ['Volatility (Daily)', 'Volatility (Annualized)'],
            '📉 DRAWDOWN METRICS': ['Maximum Drawdown', 'Average Drawdown', 'Longest Drawdown Period', 'Recovery Factor'],
            '⚖️ RISK-ADJUSTED RATIOS': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
            '⚠️ TAIL RISK METRICS': ['Value at Risk (95%)', 'Value at Risk (99%)', 'Expected Shortfall (95%)']
        }
        
        for category, metrics in categories.items():
            table_data.append([category, '', ''])  # Category header
            for metric in metrics:
                if metric in risk_metrics:
                    table_data.append([metric, risk_metrics[metric], self._get_metric_description(metric)])
            table_data.append(['', '', ''])  # Spacing
        
        # Remove last empty row
        if table_data and table_data[-1] == ['', '', '']:
            table_data.pop()
        
        # Create DataFrame
        df = pd.DataFrame(table_data, columns=['Metric', 'Value', 'Description'])
        
        # Color coding
        colors = []
        for _, row in df.iterrows():
            if any(cat in row['Metric'] for cat in ['📈', '📊', '📉', '⚖️', '⚠️']):
                colors.append(['lightblue', 'lightblue', 'lightblue'])
            elif row['Metric'] == '':
                colors.append(['white', 'white', 'white'])
            else:
                colors.append(['white', 'white', 'white'])
        
        fig = go.Figure(data=[go.Table(
            columnwidth=[200, 150, 350],
            header=dict(
                values=['<b>Risk Metric</b>', '<b>Value</b>', '<b>Description</b>'],
                fill_color='darkred',
                font=dict(color='white', size=14),
                align='left',
                height=40
            ),
            cells=dict(
                values=[df['Metric'], df['Value'], df['Description']],
                fill_color=colors,
                font=dict(color='black', size=12),
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="<b>⚠️ Comprehensive Risk Analysis</b>",
                x=0.5,
                font=dict(size=18)
            ),
            width=900,
            height=700,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

    def create_trade_statistics_table(self) -> go.Figure:
        """
        Create detailed trade statistics and patterns table.
        """
        if self.trades_df.empty:
            print("No trades data available for statistics.")
            return None
        
        trades = self.trades_df.copy()
        
        # Calculate trade statistics
        stats = {}
        
        # Basic trade stats
        stats['Total Trades'] = len(trades)
        stats['Winning Trades'] = len(trades[trades['net_pnl'] > 0])
        stats['Losing Trades'] = len(trades[trades['net_pnl'] < 0])
        stats['Break-even Trades'] = len(trades[trades['net_pnl'] == 0])
        
        # Win/Loss rates
        stats['Win Rate'] = f"{(stats['Winning Trades'] / stats['Total Trades'] * 100):.1f}%"
        stats['Loss Rate'] = f"{(stats['Losing Trades'] / stats['Total Trades'] * 100):.1f}%"
        
        # P&L statistics
        winning_trades = trades[trades['net_pnl'] > 0]
        losing_trades = trades[trades['net_pnl'] < 0]
        
        stats['Average Win'] = f"${winning_trades['net_pnl'].mean():.2f}" if not winning_trades.empty else "$0.00"
        stats['Average Loss'] = f"${losing_trades['net_pnl'].mean():.2f}" if not losing_trades.empty else "$0.00"
        stats['Largest Win'] = f"${trades['net_pnl'].max():.2f}"
        stats['Largest Loss'] = f"${trades['net_pnl'].min():.2f}"
        
        # Percentage returns
        stats['Average Win %'] = f"{winning_trades['net_pnl_pct'].mean():.2f}%" if not winning_trades.empty else "0.00%"
        stats['Average Loss %'] = f"{losing_trades['net_pnl_pct'].mean():.2f}%" if not losing_trades.empty else "0.00%"
        stats['Best Trade %'] = f"{trades['net_pnl_pct'].max():.2f}%"
        stats['Worst Trade %'] = f"{trades['net_pnl_pct'].min():.2f}%"
        
        # Trade duration analysis
        trades['duration'] = pd.to_datetime(trades['close_time']) - pd.to_datetime(trades['open_time'])
        trades['duration_hours'] = trades['duration'].dt.total_seconds() / 3600
        
        stats['Average Trade Duration'] = f"{trades['duration_hours'].mean():.1f} hours"
        stats['Shortest Trade'] = f"{trades['duration_hours'].min():.1f} hours"
        stats['Longest Trade'] = f"{trades['duration_hours'].max():.1f} hours"
        
        # Long vs Short analysis
        long_trades = trades[trades['side'] == 'long']
        short_trades = trades[trades['side'] == 'short']
        
        stats['Long Trades'] = len(long_trades)
        stats['Short Trades'] = len(short_trades)
        stats['Long Win Rate'] = f"{(len(long_trades[long_trades['net_pnl'] > 0]) / len(long_trades) * 100):.1f}%" if not long_trades.empty else "0.0%"
        stats['Short Win Rate'] = f"{(len(short_trades[short_trades['net_pnl'] > 0]) / len(short_trades) * 100):.1f}%" if not short_trades.empty else "0.0%"
        
        # Consecutive wins/losses
        trades_sorted = trades.sort_values('open_time')
        wins_losses = (trades_sorted['net_pnl'] > 0).astype(int)
        consecutive_stats = self._calculate_consecutive_stats(wins_losses)
        
        stats['Max Consecutive Wins'] = consecutive_stats['max_wins']
        stats['Max Consecutive Losses'] = consecutive_stats['max_losses']
        stats['Current Streak'] = consecutive_stats['current_streak']
        
        # Prepare table data with categories
        table_data = []
        categories = {
            '📊 TRADE OVERVIEW': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Break-even Trades', 'Win Rate', 'Loss Rate'],
            '💰 PROFIT & LOSS': ['Average Win', 'Average Loss', 'Largest Win', 'Largest Loss', 'Average Win %', 'Average Loss %', 'Best Trade %', 'Worst Trade %'],
            '⏱️ TRADE DURATION': ['Average Trade Duration', 'Shortest Trade', 'Longest Trade'],
            '📈📉 LONG vs SHORT': ['Long Trades', 'Short Trades', 'Long Win Rate', 'Short Win Rate'],
            '🔄 STREAK ANALYSIS': ['Max Consecutive Wins', 'Max Consecutive Losses', 'Current Streak']
        }
        
        for category, metrics in categories.items():
            table_data.append([category, '', ''])  # Category header
            for metric in metrics:
                if metric in stats:
                    table_data.append([metric, str(stats[metric]), self._get_trade_stat_description(metric)])
            table_data.append(['', '', ''])  # Spacing
        
        # Remove last empty row
        if table_data and table_data[-1] == ['', '', '']:
            table_data.pop()
        
        # Create DataFrame
        df = pd.DataFrame(table_data, columns=['Statistic', 'Value', 'Description'])
        
        # Color coding
        colors = []
        for _, row in df.iterrows():
            if any(emoji in row['Statistic'] for emoji in ['📊', '💰', '⏱️', '📈📉', '🔄']):
                colors.append(['lightblue', 'lightblue', 'lightblue'])
            elif row['Statistic'] == '':
                colors.append(['white', 'white', 'white'])
            else:
                # Color based on performance
                value = row['Value']
                if 'Win Rate' in row['Statistic'] or 'Win %' in row['Statistic']:
                    if '%' in value:
                        pct = float(value.replace('%', ''))
                        if pct >= 60:
                            colors.append(['white', 'lightgreen', 'white'])
                        elif pct >= 40:
                            colors.append(['white', 'lightyellow', 'white'])
                        else:
                            colors.append(['white', 'lightcoral', 'white'])
                    else:
                        colors.append(['white', 'white', 'white'])
                else:
                    colors.append(['white', 'white', 'white'])
        
        fig = go.Figure(data=[go.Table(
            columnwidth=[200, 150, 350],
            header=dict(
                values=['<b>Trade Statistic</b>', '<b>Value</b>', '<b>Description</b>'],
                fill_color='darkgreen',
                font=dict(color='white', size=14),
                align='left',
                height=40
            ),
            cells=dict(
                values=[df['Statistic'], df['Value'], df['Description']],
                fill_color=colors,
                font=dict(color='black', size=12),
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="<b>📋 Detailed Trade Statistics</b>",
                x=0.5,
                font=dict(size=18)
            ),
            width=900,
            height=800,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

    # ==================== HELPER METHODS ====================
    
    def _calculate_annualized_return(self) -> float:
        """Calculate annualized return."""
        if self.results_df.empty:
            return 0.0
        
        start_equity = self.results_df['equity'].iloc[0]
        end_equity = self.results_df['equity'].iloc[-1]
        
        # Calculate time period in years
        start_date = self.results_df.index[0]
        end_date = self.results_df.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        
        if years <= 0 or start_equity <= 0:
            return 0.0
        
        # Calculate annualized return
        total_return = (end_equity / start_equity) - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        return annualized_return * 100

    def _calculate_underwater_periods(self, drawdown_series: pd.Series) -> pd.Series:
        """Calculate periods spent underwater (in drawdown)."""
        underwater = drawdown_series < 0
        periods = []
        current_period = 0
        
        for is_underwater in underwater:
            if is_underwater:
                current_period += 1
            else:
                if current_period > 0:
                    periods.append(current_period)
                current_period = 0
        
        # Add final period if still underwater
        if current_period > 0:
            periods.append(current_period)
        
        return pd.Series(periods)

    def _calculate_consecutive_stats(self, wins_losses: pd.Series) -> Dict[str, str]:
        """Calculate consecutive wins/losses statistics."""
        if wins_losses.empty:
            return {'max_wins': '0', 'max_losses': '0', 'current_streak': 'None'}
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for win in wins_losses:
            if win == 1:  # Win
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:  # Loss
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Determine current streak
        if wins_losses.iloc[-1] == 1:
            current_streak = f"{current_wins} Wins"
        else:
            current_streak = f"{current_losses} Losses"
        
        return {
            'max_wins': str(max_wins),
            'max_losses': str(max_losses),
            'current_streak': current_streak
        }

    def _get_metric_description(self, metric: str) -> str:
        """Get description for risk metrics."""
        descriptions = {
            'Total Return': 'Overall percentage return from start to end',
            'Annualized Return': 'Return normalized to annual basis',
            'Volatility (Daily)': 'Daily standard deviation of returns',
            'Volatility (Annualized)': 'Annualized standard deviation of returns',
            'Maximum Drawdown': 'Largest peak-to-trough decline',
            'Average Drawdown': 'Average of all drawdown periods',
            'Longest Drawdown Period': 'Maximum time spent in drawdown',
            'Recovery Factor': 'Total return divided by maximum drawdown',
            'Sharpe Ratio': 'Risk-adjusted return (excess return per unit of volatility)',
            'Sortino Ratio': 'Return per unit of downside deviation',
            'Calmar Ratio': 'Annual return divided by maximum drawdown',
            'Value at Risk (95%)': 'Maximum expected loss 95% of the time',
            'Value at Risk (99%)': 'Maximum expected loss 99% of the time',
            'Expected Shortfall (95%)': 'Average loss when VaR is exceeded'
        }
        return descriptions.get(metric, 'Custom metric')

    def _get_trade_stat_description(self, stat: str) -> str:
        """Get description for trade statistics."""
        descriptions = {
            'Total Trades': 'Total number of completed trades',
            'Winning Trades': 'Number of profitable trades',
            'Losing Trades': 'Number of unprofitable trades',
            'Break-even Trades': 'Number of trades with zero P&L',
            'Win Rate': 'Percentage of winning trades',
            'Loss Rate': 'Percentage of losing trades',
            'Average Win': 'Average profit per winning trade',
            'Average Loss': 'Average loss per losing trade',
            'Largest Win': 'Single best trade in dollar terms',
            'Largest Loss': 'Single worst trade in dollar terms',
            'Average Win %': 'Average percentage return of winning trades',
            'Average Loss %': 'Average percentage return of losing trades',
            'Best Trade %': 'Best single trade percentage return',
            'Worst Trade %': 'Worst single trade percentage return',
            'Average Trade Duration': 'Mean time trades are held',
            'Shortest Trade': 'Minimum trade duration',
            'Longest Trade': 'Maximum trade duration',
            'Long Trades': 'Number of long (buy) positions',
            'Short Trades': 'Number of short (sell) positions',
            'Long Win Rate': 'Win rate for long positions only',
            'Short Win Rate': 'Win rate for short positions only',
            'Max Consecutive Wins': 'Longest winning streak',
            'Max Consecutive Losses': 'Longest losing streak',
            'Current Streak': 'Current consecutive wins or losses'
        }
        return descriptions.get(stat, 'Custom statistic')

    # ==================== COMPARATIVE ANALYSIS ====================
    
    def create_comparative_performance_table(self, 
                                           comparative_results: Dict[str, 'BacktestAnalysis']) -> go.Figure:
        """
        Create a comparison table for multiple strategy results.
        
        Args:
            comparative_results: Dictionary of strategy name -> BacktestAnalysis
        """
        if not comparative_results:
            print("No comparative results provided.")
            return None
        
        # Define metrics to compare
        comparison_metrics = [
            'initial_balance', 'final_equity', 'roi_pct', 'total_trades',
            'global_win_rate', 'profit_factor', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown_equity', 'total_profits', 'total_losses'
        ]
        
        # Prepare data
        table_data = []
        strategy_names = list(comparative_results.keys())
        
        # Metric labels
        metric_labels = {
            'initial_balance': 'Initial Balance ($)',
            'final_equity': 'Final Equity ($)',
            'roi_pct': 'ROI (%)',
            'total_trades': 'Total Trades',
            'global_win_rate': 'Win Rate (%)',
            'profit_factor': 'Profit Factor',
            'sharpe_ratio': 'Sharpe Ratio',
            'sortino_ratio': 'Sortino Ratio',
            'max_drawdown_equity': 'Max Drawdown (%)',
            'total_profits': 'Total Profits ($)',
            'total_losses': 'Total Losses ($)'
        }
        
        for metric in comparison_metrics:
            row = [metric_labels.get(metric, metric)]
            
            for strategy_name in strategy_names:
                analysis = comparative_results[strategy_name]
                value = analysis.metrics.get(metric, 0)
                
                # Format values appropriately
                if metric in ['initial_balance', 'final_equity', 'total_profits', 'total_losses']:
                    formatted_value = f"${value:,.2f}"
                elif metric in ['roi_pct', 'global_win_rate', 'max_drawdown_equity']:
                    formatted_value = f"{value:.2f}%" if metric != 'max_drawdown_equity' else f"{value*100:.2f}%"
                elif metric in ['sharpe_ratio', 'sortino_ratio', 'profit_factor']:
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:,.0f}"
                
                row.append(formatted_value)
            
            table_data.append(row)
        
        # Create DataFrame
        columns = ['Metric'] + strategy_names
        df_comp = pd.DataFrame(table_data, columns=columns)
        
        # Color coding for performance comparison
        colors = []
        for i, row in df_comp.iterrows():
            row_colors = ['lightblue']  # Metric column
            
            # Get numeric values for comparison (excluding metric name)
            values = []
            for j in range(1, len(row)):
                try:
                    # Extract numeric value
                    val_str = str(row.iloc[j]).replace(',', '').replace('%', '')
                    val = float(val_str)
                    values.append(val)
                except:
                    values.append(0)
            
            # Color based on relative performance
            if len(values) > 1:
                max_val = max(values)
                min_val = min(values)
                
                for val in values:
                    if max_val == min_val:
                        row_colors.append('white')
                    elif val == max_val and row.iloc[0] not in ['Max Drawdown (%)', 'Total Losses ($)']:
                        row_colors.append('lightgreen')  # Best performance (except for bad metrics)
                    elif val == min_val and row.iloc[0] in ['Max Drawdown (%)', 'Total Losses ($)']:
                        row_colors.append('lightgreen')  # Best performance for "lower is better" metrics
                    elif val == min_val:
                        row_colors.append('lightcoral')  # Worst performance
                    else:
                        row_colors.append('white')
            else:
                row_colors.extend(['white'] * len(values))
            
            colors.append(row_colors)
        
        fig = go.Figure(data=[go.Table(
            columnwidth=[150] + [120] * len(strategy_names),
            header=dict(
                values=[f'<b>{col}</b>' for col in df_comp.columns],
                fill_color='darkblue',
                font=dict(color='white', size=12),
                align='center',
                height=40
            ),
            cells=dict(
                values=[df_comp[col] for col in df_comp.columns],
                fill_color=list(zip(*colors)),
                font=dict(color='black', size=11),
                align='center',
                height=35
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="<b>🏆 Strategy Performance Comparison</b>",
                x=0.5,
                font=dict(size=18)
            ),
            width=min(1200, 200 + len(strategy_names) * 150),
            height=600,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

    # ==================== MAIN DASHBOARD METHOD ====================
    
    def create_dashboard(self, 
                        save_path: Optional[str] = None,
                        include_comparative: Optional[Dict[str, 'BacktestAnalysis']] = None) -> Dict[str, go.Figure]:
        """
        Create a complete dashboard with all analysis tables.
        
        Args:
            save_path: Optional directory path to save HTML files
            include_comparative: Optional comparative analysis results
            
        Returns:
            Dictionary of figure names and plotly figures
        """
        dashboard_figures = {}
        
        print("🚀 Creating Trading Analysis Dashboard...")
        
        # 1. Performance Summary
        print("  📊 Creating performance summary table...")
        dashboard_figures['performance_summary'] = self.create_performance_summary_table()
        
        # 2. Trades Analysis
        print("  📋 Creating trades analysis table...")
        dashboard_figures['trades_analysis'] = self.create_trades_analysis_table()
        
        # 3. Monthly Performance
        print("  📅 Creating monthly performance table...")
        dashboard_figures['monthly_performance'] = self.create_monthly_performance_table()
        
        # 4. Risk Metrics
        print("  ⚠️ Creating risk metrics table...")
        dashboard_figures['risk_metrics'] = self.create_risk_metrics_table()
        
        # 5. Trade Statistics
        print("  📈 Creating trade statistics table...")
        dashboard_figures['trade_statistics'] = self.create_trade_statistics_table()
        
        # 6. Comparative Analysis (if provided)
        if include_comparative:
            print("  🏆 Creating comparative analysis table...")
            dashboard_figures['comparative_analysis'] = self.create_comparative_performance_table(include_comparative)
        
        # Save figures if path provided
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            for name, fig in dashboard_figures.items():
                if fig is not None:
                    file_path = os.path.join(save_path, f"{name}.html")
                    fig.write_html(file_path)
                    print(f"  💾 Saved {name} to {file_path}")
        
        print("✅ Dashboard creation completed!")
        return dashboard_figures

    def show_dashboard(self, 
                      include_comparative: Optional[Dict[str, 'BacktestAnalysis']] = None):
        """
        Display all dashboard tables in sequence.
        """
        figures = self.create_dashboard(include_comparative=include_comparative)
        
        for name, fig in figures.items():
            if fig is not None:
                print(f"\n{'='*60}")
                print(f"Displaying: {name.replace('_', ' ').title()}")
                print(f"{'='*60}")
                fig.show()

