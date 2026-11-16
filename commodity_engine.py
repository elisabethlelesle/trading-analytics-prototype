"""Fictional European options on real equity market snapshots."""
from datetime import datetime, timezone, date, timedelta
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import math
import re

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "commodity_data"

SYMBOLS = {
    "XOM": "Exxon Mobil — oil and gas equity",
    "NEM": "Newmont — gold-mining equity",
}

FIELDS = [
    "P&L ($)", "Delta", "Gamma",
    "Vega ($/pp)", "Theta ($/day)", "Rho ($/bp)",
]


# Black-Scholes prices and analytic Greeks, per underlying unit.
@lru_cache(maxsize=4096)
def model(kind, s, k, t, r, v):
    numbers = (s, k, t, r, v)
    if (
        kind not in ("call", "put")
        or not all(math.isfinite(x) for x in numbers)
        or min(s, k, t, v) <= 0
    ):
        raise ValueError(
            "Need call/put, finite inputs, and positive stock price, "
            "strike, time to expiry and volatility."
        )

    N = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2

    d1 = (
        math.log(s / k) + (r + v * v / 2) * t
    ) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    density = math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    sign = 1 if kind == "call" else -1
    discount = math.exp(-r * t)

    return {
        "price": sign * (
            s * N(sign * d1) - k * discount * N(sign * d2)
        ),
        "Delta": sign * N(sign * d1),
        "Gamma": density / (s * v * math.sqrt(t)),
        "Vega ($/pp)": s * density * math.sqrt(t) / 100,
        "Theta ($/day)": (
            -s * density * v / (2 * math.sqrt(t))
            - sign * r * k * discount * N(sign * d2)
        ) / 365,
        "Rho ($/bp)": (
            sign * k * t * discount * N(sign * d2)
        ) / 10000,
    }


# Fetch both stocks before replacing the saved snapshot.
def refresh():
    import yfinance as yf
    import numpy as np

    STORE.mkdir(exist_ok=True)
    assets = {}

    for symbol, description in SYMBOLS.items():
        history = yf.Ticker(symbol).history(
            period="6mo",
            interval="1d",
            auto_adjust=False,
        )

        required = {"Close", "Adj Close"}
        if history.empty or not required.issubset(history.columns):
            raise ValueError(
                f"{symbol}: missing price history. Try again later."
            )

        # Exclude today's potentially unfinished daily bar.
        today = datetime.now(history.index.tz).date()
        history = history[
            [stamp.date() < today for stamp in history.index]
        ].dropna(subset=["Close", "Adj Close"])

        if (
            len(history) < 61
            or (history[["Close", "Adj Close"]] <= 0).any().any()
        ):
            raise ValueError(
                f"{symbol}: need 61 valid completed daily observations."
            )

        # Adjusted prices reduce dividend/split distortions in returns.
        returns = np.log(
            history["Adj Close"] / history["Adj Close"].shift(1)
        ).dropna().tail(60)

        volatility = float(
            returns.std(ddof=1) * np.sqrt(252)
        )
        spot = float(history["Close"].iloc[-1])

        if not math.isfinite(volatility) or volatility <= 0:
            raise ValueError(f"{symbol}: invalid volatility estimate.")

        assets[symbol] = {
            "description": description,
            "spot": spot,
            "vol": volatility,
            "date": str(history.index[-1].date()),
            "observations": len(returns),
            "vol_start": str(history.index[-61].date()),
            "vol_end": str(history.index[-1].date()),
            "vol_source": (
                "Historical: sample standard deviation of 60 adjusted "
                "daily log returns multiplied by sqrt(252)"
            ),
            "spot_source": (
                "Yahoo Finance via yfinance: latest retrieved "
                "completed daily Close"
            ),
        }

    dates = {asset["date"] for asset in assets.values()}
    if len(dates) != 1:
        raise ValueError(
            "The stock histories end on different dates. "
            "Snapshot not saved; try again later."
        )

    snapshot = {
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "as_of": next(iter(dates)),
        "assets": assets,
    }

    temporary = STORE / "snapshot.tmp"
    temporary.write_text(json.dumps(snapshot, indent=2))
    temporary.replace(STORE / "snapshot.json")

    # Refreshes preserve existing strikes, quantities and expiries.
    if not (STORE / "trades.json").exists():
        reset_trades(snapshot)

    return snapshot


def load():
    path = STORE / "snapshot.json"
    if not path.exists():
        raise ValueError(
            "No saved snapshot. Click Fetch market snapshot first."
        )
    return json.loads(path.read_text())


# Create eight fictional contracts around the snapshot prices.
def reset_trades(snapshot=None):
    snapshot = snapshot or load()
    STORE.mkdir(exist_ok=True)
    trades = []

    examples = [
        ("LC", "call",  5, 1.00, 365),
        ("SC", "call", -3, 1.10, 365),
        ("LP", "put",   4, 0.95, 180),
        ("SP", "put",  -2, 0.90, 180),
    ]

    for symbol, asset in snapshot["assets"].items():
        for label, kind, quantity, strike_factor, days in examples:
            trades.append({
                "id": f"{symbol}_{label}",
                "symbol": symbol,
                "kind": kind,
                "contracts": quantity,
                "multiplier": 100,
                "strike": round(asset["spot"] * strike_factor, 2),
                "expiry": str(
                    date.fromisoformat(snapshot["as_of"])
                    + timedelta(days=days)
                ),
            })

    (STORE / "trades.json").write_text(
        json.dumps(trades, indent=2)
    )


def parse(text):
    parts = text.strip().split()
    if (
        len(parts) < 2
        or parts[0].lower() not in ("price", "risk", "scenario")
    ):
        raise ValueError(
            "Use risk portfolio OR "
            "scenario portfolio spot=-10% vol=+5pp rate=+25bp"
        )

    action, target = parts[0].lower(), parts[1].upper()
    shocks = {"spot": 0.0, "vol": 0.0, "rate": 0.0}
    seen = set()

    for token in parts[2:]:
        match = re.fullmatch(
            r"(spot|vol|rate)=([+-]?(?:\d+(?:\.\d*)?|\.\d+))(%|pp|bp)",
            token,
        )
        if not match or action != "scenario":
            raise ValueError(
                "Shocks require scenario: spot %, vol pp, rate bp."
            )

        key, value, unit = match.groups()
        expected = {"spot": "%", "vol": "pp", "rate": "bp"}
        if key in seen or unit != expected[key]:
            raise ValueError("Duplicate shock or incorrect units.")

        shocks[key] = float(value) / (
            10000 if key == "rate" else 100
        )
        seen.add(key)

    if action == "scenario" and not seen:
        raise ValueError("A scenario needs at least one shock.")

    return action, target, shocks


def evaluate(text, assumed_rate):
    _, target, shocks = parse(text)
    snapshot = load()

    trades = json.loads((STORE / "trades.json").read_text())
    trades = [
        trade for trade in trades
        if target == "PORTFOLIO"
        or target in (trade["id"], trade["symbol"])
    ]

    if not trades:
        raise ValueError(
            "Unknown target. Use portfolio, XOM, NEM, "
            "or a trade ID such as XOM_LC."
        )

    def calculate(trade):
        asset = snapshot["assets"][trade["symbol"]]
        t = (
            date.fromisoformat(trade["expiry"])
            - date.fromisoformat(snapshot["as_of"])
        ).days / 365

        kind, strike = trade["kind"], trade["strike"]
        spot = asset["spot"] * (1 + shocks["spot"])
        vol = asset["vol"] + shocks["vol"]
        rate = assumed_rate + shocks["rate"]

        base = model(
            kind, asset["spot"], strike, t,
            assumed_rate, asset["vol"],
        )
        scenario = model(kind, spot, strike, t, rate, vol)
        units = trade["contracts"] * trade["multiplier"]

        row = {
            "Trade": trade["id"],
            "Stock": trade["symbol"],
            "Type": kind,
            "Contracts": trade["contracts"],
            "Strike ($)": strike,
            "Expiry": trade["expiry"],
            "Base value ($)": base["price"] * units,
            "Scenario value ($)": scenario["price"] * units,
            "P&L ($)": (scenario["price"] - base["price"]) * units,
            **{
                greek: scenario[greek] * units
                for greek in FIELDS[1:]
            },
        }

        # Approximate derivatives by small up/down input changes.
        def price(ss=spot, tt=t, rr=rate, vv=vol):
            return model(kind, ss, strike, tt, rr, vv)["price"]

        hs = spot * 0.0001
        hv = min(0.0001, vol / 10)
        hr = 0.00001
        ht = min(0.0001, t / 10)

        numerical = {
            "Delta": (
                price(ss=spot + hs) - price(ss=spot - hs)
            ) / (2 * hs),
            "Gamma": (
                price(ss=spot + hs) - 2 * price()
                + price(ss=spot - hs)
            ) / hs**2,
            "Vega ($/pp)": (
                price(vv=vol + hv) - price(vv=vol - hv)
            ) / (2 * hv) * 0.01,
            "Theta ($/day)": (
                price(tt=t - ht) - price(tt=t + ht)
            ) / (2 * ht) / 365,
            "Rho ($/bp)": (
                price(rr=rate + hr) - price(rr=rate - hr)
            ) / (2 * hr) * 0.0001,
        }

        checks = []
        for greek, approximation in numerical.items():
            analytic = row[greek]
            estimate = approximation * units
            tolerance = (
                0.0001 + 0.001 * max(abs(analytic), abs(estimate))
            )
            error = abs(analytic - estimate)
            checks.append({
                "Trade": trade["id"],
                "Check": greek,
                "Reported": analytic,
                "Numerical / expected": estimate,
                "Absolute error": error,
                "Tolerance": tolerance,
                "Pass": error <= tolerance,
            })

        expected_pnl = (
            row["Scenario value ($)"] - row["Base value ($)"]
        )
        error = abs(row["P&L ($)"] - expected_pnl)
        checks.append({
            "Trade": trade["id"],
            "Check": "P&L reconciliation",
            "Reported": row["P&L ($)"],
            "Numerical / expected": expected_pnl,
            "Absolute error": error,
            "Tolerance": 1e-7,
            "Pass": error <= 1e-7,
        })

        # European put-call parity is an additional pricing identity.
        parity = (
            model("call", spot, strike, t, rate, vol)["price"]
            - model("put", spot, strike, t, rate, vol)["price"]
        )
        expected_parity = spot - strike * math.exp(-rate * t)
        error = abs(parity - expected_parity)
        checks.append({
            "Trade": trade["id"],
            "Check": "Put-call parity (per unit)",
            "Reported": parity,
            "Numerical / expected": expected_parity,
            "Absolute error": error,
            "Tolerance": 1e-7,
            "Pass": error <= 1e-7,
        })

        return row, checks

    with ThreadPoolExecutor(max_workers=4) as pool:
        calculated = list(pool.map(calculate, trades))

    return {
        "rows": [item[0] for item in calculated],
        "checks": [
            check for item in calculated for check in item[1]
        ],
        "snapshot": snapshot,
        "request": text,
        "assumed_base_rate": assumed_rate,
        "scenario_rate": assumed_rate + shocks["rate"],
    }
