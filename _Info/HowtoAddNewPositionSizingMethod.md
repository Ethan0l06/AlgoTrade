# **How to Add a New Position Sizing Method**

This guide provides a step-by-step tutorial for adding a new, custom position sizing method to the backtesting framework. The system is designed to be modular, so adding a new method only requires changes in three specific files.

Let's use a hypothetical new method called **RiskParity** as an example throughout this tutorial.

### **Step 1: Add New Parameters to the Configuration**

First, you need to add any parameters your new method requires to the main configuration class. This makes them accessible throughout the system and ensures they are properly configured for each run.

**File to Edit:** Config/BacktestConfig.py

1. Find the \--- Position Sizing \--- section.  
2. Add a new comment block and the required parameter(s) for your method.

**Example:** Let's say our RiskParity method needs a risk\_parity\_lookback parameter.

\# /TradingBacktester/Config/BacktestConfig.py

\# ... (other parameters) ...

    \# Parameters for 'KellyCriterion'  
    kelly\_lookback\_period: int \= 50  
    kelly\_fraction: float \= 0.5  \# e.g., use 50% of the calculated Kelly fraction

    \# \=== ▼▼▼ ADD YOUR NEW PARAMETERS HERE ▼▼▼ \===

    \# Parameters for 'RiskParity'  
    risk\_parity\_lookback: int \= 20 \# Lookback for calculating portfolio volatility

\# ... (rest of the file) ...

### **Step 2: Implement the Sizing Logic in the Backtest Runner**

Next, you need to teach the BacktestRunner how to calculate the initial\_margin using your new method. This logic is centralized in the \_get\_initial\_margin private method.

**File** to **Edit:** Backtester/BacktestRunner.py

1. Navigate to the \_get\_initial\_margin method.  
2. Add a new elif condition for your new method's name.  
3. Inside this block, write the Python code to calculate the margin. This code will have access to self.equity, self.balance, and any parameters you added to self.config in Step 1\.

**Example:** Adding the logic for RiskParity.

\# /TradingBacktester/Backtester/BacktestRunner.py

    def \_get\_initial\_margin(self, price: float, atr\_value: Optional\[float\]) \-\> tuple\[float, Optional\[float\]\]:  
        \# ... (other code in the method) ...  
        elif cfg.position\_sizing\_method \== 'KellyCriterion':  
            \# ... (logic for Kelly Criterion) ...

        \# \=== ▼▼▼ ADD YOUR NEW LOGIC HERE ▼▼▼ \===

        elif cfg.position\_sizing\_method \== 'RiskParity':  
            \# This is a placeholder for your actual risk parity logic.  
            \# You can access the new parameter via \`cfg.risk\_parity\_lookback\`.  
            \# For this example, we'll just implement a simple logic.  
            if len(self.trades\_info) \< cfg.risk\_parity\_lookback:  
                \# Fallback if there isn't enough trade history  
                margin \= self.balance \* 0.01 \# Fallback to 1% of balance  
            else:  
                \# Your actual calculation would go here.  
                \# Example: size inversely to recent return volatility.  
                recent\_returns \= pd.Series(\[t\['net\_pnl\_pct'\] for t in self.trades\_info\[-cfg.risk\_parity\_lookback:\]\])  
                volatility \= recent\_returns.std()  
                if volatility \> 0:  
                    \# Simplified example: risk less when volatility is high  
                    margin \= (self.equity \* 0.01) / volatility  
                else:  
                    margin \= self.balance \* 0.01  
          
        \# \=== ▲▲▲ END OF NEW LOGIC ▲▲▲ \===

        else:  
            raise ValueError(f"Unknown position sizing method: {cfg.position\_sizing\_method}")  
              
        return margin, sl\_price

### **Step 3: Add the Method to the Comparative Run**

Finally, to ensure your new method is included in the side-by-side performance report, you need to add its name to the list of methods to be tested.

**File to Edit:** comparative\_run.py

1. Find the run\_comparative\_analysis function.  
2. Add the exact string name of your new method to the methods\_to\_compare list. This name must match the one you used in the elif block in Step 2\.

**Example:** Adding RiskParity to the comparison.

\# /TradingBacktester/comparative\_run.py

def run\_comparative\_analysis(base\_config: BacktestConfig, data: pd.DataFrame) \-\> Dict\[str, BacktestAnalysis\]:  
    methods\_to\_compare \= \[  
        'PercentBalance',  
        'FixedAmount',  
        'AtrVolatility',  
        'KellyCriterion',  
        'RiskParity'  \# \<-- ADD YOUR NEW METHOD'S NAME HERE  
    \]  
      
    \# ... (rest of the function is unchanged) ...

That's it\! Once you have completed these three steps, your new position sizing method will be fully integrated into the backtesting framework. When you next run the comparative\_run.py script, it will automatically be included in the test and will appear as a new row in the final performance report.