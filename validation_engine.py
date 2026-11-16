"""Validation and reproducibility wrapper for commodity_engine."""
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import platform
import uuid

import numpy as np
import pandas as pd
import exchange_calendars as xcals
import yfinance as yf

from vollib.black_scholes import black_scholes
from vollib.black_scholes.greeks import analytical as ref_greeks

import commodity_engine as core

STORE = core.STORE
FIELDS = core.FIELDS
VERSION = "validation-v1"


def record(section, check, status, detail):
    return {
        "Section": section,
        "Check": check,
        "Status": status,
        "Detail": str(detail),
    }


# Reference outputs use the same conventions as our engine.
def reference(kind, s, k, t, r, v):
    flag = "c" if kind == "call" else "p"
    args = (flag, s, k, t, r, v)

    return {
        "price": float(black_scholes(*args)),
        "Delta": float(ref_greeks.delta(*args)),
        "Gamma": float(ref_greeks.gamma(*args)),
        "Vega ($/pp)": float(ref_greeks.vega(*args)),
        "Theta ($/day)": float(ref_greeks.theta(*args)),
        # vollib rho is per percentage point; ours is per basis point.
        "Rho ($/bp)": float(ref_greeks.rho(*args)) / 100,
    }


def comparisons(label, args):
    own = core.model(*args)
    ref = reference(*args)
    rows = []

    for metric, value in own.items():
        expected = ref[metric]
        tolerance = 1e-7 + 1e-5 * max(abs(value), abs(expected))
        error = abs(value - expected)

        rows.append({
            "Case": label,
            "Metric": metric,
            "Our per-unit result": value,
            "vollib per-unit result": expected,
            "Absolute error": error,
            "Tolerance": tolerance,
            "Status": (
                "PASS"
                if math.isfinite(error) and error <= tolerance
                else "FAIL"
            ),
        })

    return rows


# Test different market conditions, not just the example portfolio.
def edge_tests():
    rows = []

    cases = [
        ("At money", 100, 100, 1, .04, .2),
        ("Deep in/out money", 250, 100, 1, .04, .2),
        ("Low stock", 40, 100, 1, .04, .2),
        ("One day remaining", 100, 100, 1/365, .04, .2),
        ("Low volatility", 100, 100, 1, .04, .01),
        ("High volatility", 100, 100, 1, .04, 1.5),
        ("Zero rate", 100, 100, 1, 0, .2),
        ("Negative rate", 100, 100, 1, -.02, .2),
    ]

    for label, s, k, t, r, v in cases:
        for kind in ("call", "put"):
            checks = comparisons(label, (kind, s, k, t, r, v))
            failures = [
                x["Metric"] for x in checks
                if x["Status"] == "FAIL"
            ]
            rows.append(record(
                "Edge cases",
                label + " " + kind,
                "FAIL" if failures else "PASS",
                "Reference agreement: " + (
                    ", ".join(failures)
                    if failures else "all six outputs"
                ),
            ))

    invalid_cases = [
        ("Expired contract", ("call", 100, 100, 0, .04, .2)),
        ("Negative spot", ("call", -1, 100, 1, .04, .2)),
        ("Zero volatility", ("call", 100, 100, 1, .04, 0)),
        ("Nonfinite input", ("call", float("nan"), 100, 1, .04, .2)),
    ]

    for label, args in invalid_cases:
        try:
            core.model(*args)
            rows.append(record(
                "Edge cases", label, "FAIL",
                "Invalid input was accepted",
            ))
        except ValueError:
            rows.append(record(
                "Edge cases", label, "PASS",
                "Invalid input rejected as intended",
            ))

    for text in (
        "scenario portfolio spot=-100%",
        "scenario portfolio vol=-200pp",
    ):
        try:
            _, _, shocks = core.parse(text)
            core.model(
                "call",
                100 * (1 + shocks["spot"]),
                100, 1, .04,
                .2 + shocks["vol"],
            )
            rows.append(record(
                "Edge cases", text, "FAIL",
                "Invalid shocked input accepted",
            ))
        except ValueError:
            rows.append(record(
                "Edge cases", text, "PASS",
                "Invalid shocked input rejected",
            ))

    return rows


def quality(snapshot):
    rows = []
    calendar = xcals.get_calendar("XNYS")
    today = pd.Timestamp.now(tz="America/New_York").date()

    # Match our policy: always exclude today's daily observation.
    expected = calendar.date_to_session(
        pd.Timestamp(today - timedelta(days=1)),
        direction="previous",
    ).date()

    dates = []

    for symbol, asset in snapshot["assets"].items():
        observed_date = date.fromisoformat(asset["date"])
        dates.append(observed_date)

        valid = all(
            math.isfinite(asset[key]) and asset[key] > 0
            for key in ("spot", "vol")
        )
        rows.append(record(
            "Data", symbol + " valid inputs",
            "PASS" if valid else "FAIL",
            "Positive finite price and volatility",
        ))

        count = asset.get("observations", 0)
        rows.append(record(
            "Data", symbol + " return count",
            "PASS" if count >= 60 else "FAIL", count,
        ))

        freshness = (
            "PASS" if observed_date == expected
            else "FAIL" if observed_date > expected
            else "WARNING"
        )
        rows.append(record(
            "Data", symbol + " freshness", freshness,
            f"Observed {observed_date}; expected latest "
            f"session before today {expected}",
        ))

        history = asset.get("history")
        if not history:
            rows.append(record(
                "Data", symbol + " saved history", "WARNING",
                "Older snapshot: fetch again to save observations "
                "and verify missing sessions",
            ))
            continue

        frame = pd.DataFrame(history)
        duplicate = frame["date"].duplicated().any()
        values = frame[["close", "adjusted"]].to_numpy()
        valid_history = (
            np.isfinite(values).all() and (values > 0).all()
        )

        rows.append(record(
            "Data", symbol + " history values",
            "PASS" if valid_history and not duplicate else "FAIL",
            "Finite positive closes; no duplicate dates",
        ))

        actual = set(frame["date"])
        sessions = calendar.sessions_in_range(
            frame["date"].min(), frame["date"].max()
        )
        missing = [
            str(session.date()) for session in sessions
            if str(session.date()) not in actual
        ]
        rows.append(record(
            "Data", symbol + " missing sessions",
            "WARNING" if missing else "PASS",
            missing or "No missing NYSE sessions",
        ))

        returns = np.log(
            frame["adjusted"] / frame["adjusted"].shift(1)
        ).dropna()

        large = int((abs(returns) > .20).sum())
        rows.append(record(
            "Data", symbol + " large returns",
            "WARNING" if large else "PASS",
            f"{large} adjusted daily log returns exceed 20%; "
            "review if present",
        ))

        estimated = float(
            returns.tail(60).std(ddof=1) * np.sqrt(252)
        )
        rows.append(record(
            "Data", symbol + " volatility reconstruction",
            "PASS" if abs(estimated - asset["vol"]) < 1e-10 else "FAIL",
            "Recomputed from saved adjusted closes",
        ))

    rows.append(record(
        "Data", "Common valuation date",
        "PASS" if len(set(dates)) == 1 else "FAIL", dates,
    ))

    return rows


# Fetch and validate before replacing a previously saved snapshot.
def refresh():
    STORE.mkdir(exist_ok=True)
    assets = {}

    for symbol, description in core.SYMBOLS.items():
        history = yf.Ticker(symbol).history(
            period="6mo", interval="1d", auto_adjust=False
        )

        if (
            history.empty
            or not {"Close", "Adj Close"}.issubset(history.columns)
        ):
            raise ValueError(
                f"{symbol}: missing history; previous snapshot retained"
            )

        today = datetime.now(history.index.tz).date()
        history = history[
            [stamp.date() < today for stamp in history.index]
        ]

        if history.index.duplicated().any():
            raise ValueError(
                f"{symbol}: duplicate dates; previous snapshot retained"
            )

        values = history[["Close", "Adj Close"]].to_numpy()
        if (
            len(history) < 61
            or not np.isfinite(values).all()
            or (values <= 0).any()
        ):
            raise ValueError(
                f"{symbol}: insufficient or invalid data; "
                "previous snapshot retained"
            )

        observations = [
            {
                "date": str(stamp.date()),
                "close": float(row["Close"]),
                "adjusted": float(row["Adj Close"]),
            }
            for stamp, row in history.iterrows()
        ]

        returns = np.log(
            history["Adj Close"]
            / history["Adj Close"].shift(1)
        ).dropna().tail(60)

        assets[symbol] = {
            "description": description,
            "spot": float(history["Close"].iloc[-1]),
            "vol": float(returns.std(ddof=1) * np.sqrt(252)),
            "date": str(history.index[-1].date()),
            "observations": len(returns),
            "vol_start": str(history.index[-61].date()),
            "vol_end": str(history.index[-1].date()),
            "vol_source": (
                "Historical: sample std of 60 adjusted "
                "log returns multiplied by sqrt(252)"
            ),
            "history": observations,
        }

    snapshot = {
        "as_of": next(iter(assets.values()))["date"],
        "assets": assets,
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
    }

    failures = [
        check for check in quality(snapshot)
        if check["Status"] == "FAIL"
    ]
    if failures:
        raise ValueError("Snapshot rejected: " + str(failures))

    temporary = STORE / "snapshot.tmp"
    temporary.write_text(json.dumps(snapshot, indent=2))
    temporary.replace(STORE / "snapshot.json")

    if not (STORE / "trades.json").exists():
        core.reset_trades(snapshot)

    return snapshot


load = core.load
reset_trades = core.reset_trades


def assumptions():
    entries = [
        (
            "Exercise",
            "Fictional European contracts. Listed US equity "
            "options generally permit early exercise.",
        ),
        (
            "Dividends",
            "Zero in this model. XOM and NEM pay dividends. "
            "Adjusting historical stock prices does not fix "
            "this option-pricing assumption.",
        ),
        (
            "Volatility",
            "Backward-looking 60-return estimate, not option-chain "
            "implied volatility. Constant within each valuation.",
        ),
        (
            "Interest rate",
            "User assumption: flat, continuously compounded annual "
            "rate; not an observed yield curve.",
        ),
        (
            "Market data",
            "Completed daily closes; not synchronized live quotes.",
        ),
        (
            "Meaning of PASS",
            "Numerical agreement does not prove market-price "
            "accuracy or profitable trading.",
        ),
        (
            "Position risk",
            "Short calls can have unlimited loss. Selected scenarios "
            "do not establish maximum possible loss.",
        ),
        (
            "Access",
            "Local prototype without authentication "
            "or multi-user isolation.",
        ),
    ]

    return [
        record("Assumptions", name, "LIMITATION", detail)
        for name, detail in entries
    ]


def evaluate(text, rate):
    snapshot = load()
    data_checks = quality(snapshot)

    failures = [
        check for check in data_checks
        if check["Status"] == "FAIL"
    ]
    if failures:
        raise ValueError("Data quality failure: " + str(failures))

    trade_bytes = (STORE / "trades.json").read_bytes()
    data = core.evaluate(text, rate)

    if (
        (STORE / "trades.json").read_bytes() != trade_bytes
        or load() != snapshot
    ):
        raise ValueError("Inputs changed during analysis. Run again.")

    trades = json.loads(trade_bytes)
    _, _, shocks = core.parse(text)
    selected = {row["Trade"] for row in data["rows"]}

    references = []
    pricing_inputs = []

    for trade in trades:
        if trade["id"] not in selected:
            continue

        asset = snapshot["assets"][trade["symbol"]]
        maturity = (
            date.fromisoformat(trade["expiry"])
            - date.fromisoformat(snapshot["as_of"])
        ).days / 365

        markets = [
            ("base", asset["spot"], asset["vol"], rate),
            (
                "scenario",
                asset["spot"] * (1 + shocks["spot"]),
                asset["vol"] + shocks["vol"],
                rate + shocks["rate"],
            ),
        ]

        for label, spot, vol, current_rate in markets:
            args = (
                trade["kind"], spot, trade["strike"],
                maturity, current_rate, vol,
            )
            references.extend(
                comparisons(trade["id"] + " " + label, args)
            )
            pricing_inputs.append({
                "trade": trade["id"],
                "valuation": label,
                "args": list(args),
                "signed_units": (
                    trade["contracts"] * trade["multiplier"]
                ),
            })

    scaling = []
    for row in data["rows"]:
        trade = next(
            trade for trade in trades
            if trade["id"] == row["Trade"]
        )
        units = trade["contracts"] * trade["multiplier"]
        args = next(
            item["args"] for item in pricing_inputs
            if item["trade"] == row["Trade"]
            and item["valuation"] == "scenario"
        )
        ref = reference(*args)

        for metric in ["Scenario value ($)"] + FIELDS[1:]:
            key = "price" if metric == "Scenario value ($)" else metric
            expected = ref[key] * units
            error = abs(row[metric] - expected)
            tolerance = 1e-5 + 1e-5 * abs(expected)

            scaling.append(record(
                "Position scaling",
                row["Trade"] + " " + metric,
                "PASS" if error <= tolerance else "FAIL",
                f"Absolute difference from signed reference: {error:.8g}",
            ))

    data["reference_checks"] = references
    data["validation_status"] = (
        data_checks + scaling + edge_tests() + assumptions()
    )
    data["pricing_inputs"] = pricing_inputs
    data["trades"] = trades
    data["model_version"] = VERSION

    packages = (
        "vollib", "yfinance", "numpy", "pandas",
        "shiny", "exchange-calendars",
    )
    data["environment"] = {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in packages
        },
    }

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-" + uuid.uuid4().hex[:10]
    )
    run_directory = STORE / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    data["run_id"] = run_id
    data["archive_path"] = str(run_directory)

    (run_directory / "run.json").write_text(
        json.dumps(data, indent=2, allow_nan=False)
    )

    sources = [
        ("commodity_engine.py", Path(core.__file__).read_bytes()),
        ("validation_engine.py", Path(__file__).read_bytes()),
    ]
    for filename, content in sources:
        (run_directory / filename).write_bytes(content)

    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run_directory.iterdir()
        if path.is_file()
    }
    (run_directory / "sha256.json").write_text(
        json.dumps(manifest, indent=2)
    )

    return data


# Recalculate saved position values without downloading new data.
def replay(run_path):
    path = Path(run_path)
    manifest = json.loads((path / "sha256.json").read_text())

    for name, expected_hash in manifest.items():
        actual_hash = hashlib.sha256(
            (path / name).read_bytes()
        ).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("Archive integrity mismatch: " + name)

    data = json.loads((path / "run.json").read_text())

    for saved in data["pricing_inputs"]:
        row = next(
            row for row in data["rows"]
            if row["Trade"] == saved["trade"]
        )
        column = (
            "Base value ($)"
            if saved["valuation"] == "base"
            else "Scenario value ($)"
        )
        expected = row[column]
        current = (
            core.model(*saved["args"])["price"]
            * saved["signed_units"]
        )
        if not math.isclose(
            current, expected, rel_tol=1e-10, abs_tol=1e-7
        ):
            raise ValueError("Replay mismatch: " + saved["trade"])

    return (
        "Archive hashes verified; base/scenario position values "
        "reproduced with the current engine."
    )
