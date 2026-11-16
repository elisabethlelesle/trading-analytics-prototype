# Trading Analytics Prototype (inspired by JP Athena's conceptual pipeline)

Thank you Carlos for your feedback!

Local Shiny dashboard for fictional European options on XOM and NEM. All positions are tested (long/short call/put).

Features: Black-Scholes pricing, Greeks, scenario analysis and P&L, yfinance daily snapshots (Yahoo finance API), and numerical validation with standard vollib library.

Steps:

1. In Terminal, install: python3 -m pip install -r requirements.txt

2. Run: python3 -m shiny run --launch-browser trader_dashboard.py

3. Use Refresh daily prices on first launch, then Run analysis.
The prices are taken from the latest stock market data snapshots from yfinance.

Assumptions: zero dividends, historical volatility, and an arbitrary interest rate (can be manually modified). Numerical checks establish computational consistency, not market-price accuracy. This is just a simple prototype to better help me understand how a platform like this might work.

I also want to specify that this project was done with the help of AI (OpenAI Codex).
