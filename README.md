# Trading Analytics Prototype

Local Shiny dashboard for fictional European options on XOM and NEM.

Features: Black-Scholes pricing, Greeks, scenario analysis, yfinance daily snapshots, and numerical validation.

Install: python3 -m pip install -r requirements.txt

Run: python3 -m shiny run --launch-browser trader_dashboard.py

Use Refresh daily prices on first launch, then Run analysis.

Assumptions: zero dividends, historical volatility, and a user-assumed interest rate. Numerical checks establish computational consistency, not market-price accuracy. Educational local prototype; no authentication.
