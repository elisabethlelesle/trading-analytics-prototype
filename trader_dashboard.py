from datetime import datetime, timezone
import json

import pandas as pd
from matplotlib.figure import Figure
from shiny import App, reactive, render, ui

import validation_engine as engine




def position_choices():
    choices = {
        "portfolio": "All positions",
        "XOM": "All XOM positions",
        "NEM": "All NEM positions",
    }

    import json
    trades_path = engine.STORE / "trades.json"
    if trades_path.exists():
        for trade in json.loads(trades_path.read_text()):
            side = "Long" if trade["contracts"] > 0 else "Short"
            label = (
                f"{trade['id']} — {side} {trade['kind']} | "
                f"{abs(trade['contracts'])} contracts | "
                f"strike ${trade['strike']:.2f} | "
                f"expiry {trade['expiry']}"
            )
            choices[trade["id"]] = label

    return choices

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("Market"),
        ui.output_text("market_stamp"),
        ui.input_action_button("fetch", "Refresh daily prices"),
        ui.hr(),
        ui.input_numeric(
            "rate", "Base interest rate (%)",
            value=4, min=-5, max=30, step=0.25,
        ),
        ui.input_radio_buttons(
            "mode", "Request",
            {"controls": "Controls", "text": "Text"},
            selected="controls",
        ),
        ui.panel_conditional(
            "input.mode === 'controls'",
            ui.input_select(
                "target",
                "Positions",
                position_choices(),
            ),
            ui.input_checkbox(
                "scenario", "Apply scenario", value=True
            ),
            ui.panel_conditional(
                "input.scenario",
                ui.input_slider(
                    "spot", "Stock price change (%)",
                    min=-50, max=50, value=-10,
                ),
                ui.input_slider(
                    "vol", "Volatility change (pp)",
                    min=-10, max=30, value=5,
                ),
                ui.input_slider(
                    "rateshock", "Rate change (bp)",
                    min=-200, max=200, value=25, step=5,
                ),
            ),
        ),
        ui.panel_conditional(
            "input.mode === 'text'",
            ui.input_text_area(
                "text", "Request",
                value="risk portfolio", rows=3,
            ),
        ),
        ui.input_action_button(
            "run", "Run analysis",
            class_="btn-primary w-100",
        ),
        width=300,
    ),
    ui.tags.style("""
        body { background: #f6f8fb; }
        .card { border-radius: 10px; }
        .run-label {
            color: #526173;
            margin: 12px 0;
            font-size: .9rem;
        }
        .shiny-table-output { overflow-x: auto; }
        table { font-size: .88rem; white-space: nowrap; }
    """),
    ui.output_ui("operation_status"),
    ui.output_ui("summary"),
    ui.div(
        ui.output_text("run_label"),
        class_="run-label",
    ),
    ui.navset_card_tab(
        ui.nav_panel(
            "Positions",
            ui.output_table("positions"),
            ui.download_button("download", "Download CSV"),
        ),
        ui.nav_panel(
            "Risk by stock",
            ui.output_table("riskgroups"),
        ),
        ui.nav_panel(
            "P&L",
            ui.output_plot("chart", height="350px"),
        ),
        ui.nav_panel(
            "Market data",
            ui.output_table("compact_market"),
        ),
        ui.nav_panel(
            "Validation",
            ui.output_ui("validation_status_compact"),
            ui.accordion(
                ui.accordion_panel(
                    "Data quality",
                    ui.output_table("data_validation"),
                ),
                ui.accordion_panel(
                    "Reference library",
                    ui.output_table("reference_validation"),
                ),
                ui.accordion_panel(
                    "Position scaling",
                    ui.output_table("scale_validation"),
                ),
                ui.accordion_panel(
                    "Numerical checks",
                    ui.output_table("validation"),
                ),
                ui.accordion_panel(
                    "Edge cases",
                    ui.output_table("edge_validation"),
                ),
                ui.accordion_panel(
                    "Model assumptions",
                    ui.output_table("assumption_validation"),
                ),
                ui.accordion_panel(
                    "Run archive",
                    ui.output_text("archive_info"),
                    ui.download_button(
                        "validation_download",
                        "Download run JSON",
                    ),
                ),
                open=False,
            ),
        ),
        ui.nav_panel(
            "Settings",
            ui.h5("Example portfolio"),
            ui.p(
                "Reset creates eight example positions using "
                "the saved market snapshot. It replaces existing "
                "strikes, quantities and expiries. "
                "Refreshing prices does not reset positions."
            ),
            ui.input_action_button(
                "reset", "Reset example positions",
                class_="btn-outline-danger",
            ),
            ui.hr(),
            ui.h5("Data method"),
            ui.p(
                "XOM: Exxon Mobil. NEM: Newmont. "
                "Prices are completed daily closes. "
                "Volatility is estimated from 60 adjusted daily "
                "returns. The base interest rate is editable."
            ),
        ),
        id="workspace_view",
        selected="Positions",
    ),
    title="Trading analytics",
    fillable=False,
)



def server(input, output, session):
    snapshot = reactive.Value(None)
    result = reactive.Value(None)
    notice = reactive.Value(
        "Fetch a snapshot, then click Run analysis. "
        "A saved snapshot can also be used offline."
    )

    try:
        snapshot.set(engine.load())
    except ValueError:
        pass

    @reactive.effect
    @reactive.event(input.fetch)
    def fetch():
        try:
            snapshot.set(engine.refresh())
            result.set(None)
            notice.set(
                "Snapshot saved. Existing trades retained. "
                "Click Run analysis."
            )
        except Exception as exc:
            notice.set(
                f"Fetch failed: {exc}. Any previous snapshot "
                "is retained; no synthetic fallback is used."
            )

    @reactive.effect
    @reactive.event(input.reset)
    def reset():
        try:
            engine.reset_trades()
            result.set(None)
            notice.set(
                "Eight new fictional positions created around "
                "the saved snapshot. Click Run analysis."
            )
        except Exception as exc:
            notice.set(str(exc))

    @reactive.effect
    @reactive.event(input.run)
    def run():
        event = {
            "time_utc": datetime.now(timezone.utc).isoformat()
        }
        try:
            rate = float(input.rate()) / 100

            if input.mode() == "text":
                text = input.text()
            elif input.scenario():
                text = (
                    f"scenario {input.target()} "
                    f"spot={input.spot():+g}% "
                    f"vol={input.vol():+g}pp "
                    f"rate={input.rateshock():+g}bp"
                )
            else:
                text = f"risk {input.target()}"

            event.update(request=text, assumed_rate=rate)
            data = engine.evaluate(text, rate)
            result.set(data)
            snapshot.set(data["snapshot"])

            passed = sum(check["Pass"] for check in data["checks"])
            notice.set(
                f"Completed. {passed}/{len(data['checks'])} "
                "numerical checks passed."
            )
            event.update(
                status="ok",
                snapshot=data["snapshot"],
                rows=data["rows"],
                checks=data["checks"],
            )
        except Exception as exc:
            result.set(None)
            notice.set(f"Analysis failed: {exc}")
            event.update(status="error", error=str(exc))

        engine.STORE.mkdir(exist_ok=True)
        with (engine.STORE / "audit.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")

    @render.text
    def message():
        return notice()

    @render.table
    def market():
        data = snapshot()
        if data is None:
            return pd.DataFrame({"Status": ["No snapshot yet"]})

        return pd.DataFrame([
            {
                "Stock": symbol,
                "Close ($)": round(asset["spot"], 2),
                "Price date": asset["date"],
                "Historical vol (%)": round(asset["vol"] * 100, 2),
                "Returns": asset["observations"],
                "Vol window": (
                    asset["vol_start"] + " to " + asset["vol_end"]
                ),
                "Vol source": (
                    "Sample std of 60 adjusted log returns × sqrt(252)"
                ),
            }
            for symbol, asset in data["assets"].items()
        ])

    @render.ui
    def summary():
        data = result()
        if data is None:
            return None

        frame = pd.DataFrame(data["rows"])
        base = frame["Base value ($)"].sum()
        value = frame["Scenario value ($)"].sum()
        pnl = frame["P&L ($)"].sum()

        return ui.layout_columns(
            ui.value_box(
                "Base value", f"${base:,.2f}", theme="primary"
            ),
            ui.value_box(
                "Scenario value", f"${value:,.2f}", theme="primary"
            ),
            ui.value_box(
                "Scenario P&L", f"${pnl:,.2f}",
                theme="danger" if pnl < 0 else "success",
            ),
            col_widths=(4, 4, 4),
        )

    @render.text
    def provenance():
        data = result()
        if data is None:
            return "No analysis yet."

        return (
            f"Request: {data['request']} | "
            f"Snapshot fetched UTC: {data['snapshot']['fetched_utc']} | "
            f"Assumed base rate: {data['assumed_base_rate']:.2%} | "
            f"Scenario rate: {data['scenario_rate']:.2%}. "
            "Volatility is historical, not option-chain implied."
        )

    @render.table
    def positions():
        data = result()
        return (
            pd.DataFrame(data["rows"]).round(4)
            if data else pd.DataFrame()
        )

    @render.table
    def riskgroups():
        data = result()
        if not data:
            return pd.DataFrame()

        return (
            pd.DataFrame(data["rows"])
            .groupby("Stock")[engine.FIELDS]
            .sum()
            .reset_index()
            .round(4)
        )

    @render.table
    def validation():
        data = result()
        return (
            pd.DataFrame(data["checks"])
            if data else pd.DataFrame()
        )

    @render.plot
    def chart():
        figure = Figure(figsize=(9, 3.5))
        axis = figure.subplots()
        data = result()

        if data:
            rows = data["rows"]
            axis.bar(
                [row["Trade"] for row in rows],
                [row["P&L ($)"] for row in rows],
                color=[
                    "#dc2626" if row["P&L ($)"] < 0 else "#059669"
                    for row in rows
                ],
            )
            axis.axhline(0, color="gray", linewidth=0.8)
            axis.set_ylabel("Scenario P&L (USD)")
            axis.tick_params(axis="x", labelrotation=30)
        else:
            axis.text(
                0.5, 0.5, "Run an analysis",
                ha="center", transform=axis.transAxes,
            )
            axis.set_axis_off()

        figure.tight_layout()
        return figure

    @render.download(filename="commodity-positions.csv")
    def download():
        data = result()
        yield (
            pd.DataFrame(data["rows"]).to_csv(index=False)
            if data else "Status\nRun analysis first\n"
        )

    @render.download(filename="numerical-checks.csv")
    def download_checks():
        data = result()
        yield (
            pd.DataFrame(data["checks"]).to_csv(index=False)
            if data else "Status\nRun analysis first\n"
        )



    @render.ui
    def validation_overview():
        data = result()
        if not data:
            return ui.p(
                "Run an analysis to populate validation results."
            )

        rows = (
            data["validation_status"] + data["reference_checks"]
        )
        failures = sum(
            row["Status"] == "FAIL" for row in rows
        )
        failures += sum(
            not row["Pass"] for row in data["checks"]
        )
        warnings = sum(
            row["Status"] == "WARNING" for row in rows
        )

        return ui.div(
            ui.strong(
                f"{failures} failures; {warnings} warnings."
            ),
            ui.p(
                "PASS means numerical agreement or a valid input "
                "check. It does not establish market-price accuracy. "
                "LIMITATION identifies a modelling assumption, "
                "not a passed test."
            ),
            class_=(
                "alert alert-danger" if failures
                else "alert alert-warning" if warnings
                else "alert alert-success"
            ),
        )

    def section_table(section):
        data = result()
        if not data:
            return pd.DataFrame()
        return pd.DataFrame([
            row for row in data["validation_status"]
            if row["Section"] == section
        ])

    @render.table
    def data_validation():
        return section_table("Data")

    @render.table
    def scale_validation():
        return section_table("Position scaling")

    @render.table
    def edge_validation():
        return section_table("Edge cases")

    @render.table
    def assumption_validation():
        return section_table("Assumptions")

    @render.table
    def reference_validation():
        data = result()
        return (
            pd.DataFrame(data["reference_checks"])
            if data else pd.DataFrame()
        )

    @render.text
    def archive_info():
        data = result()
        if not data:
            return "No archived run yet."

        return (
            f"Run: {data['run_id']} | "
            f"Saved to: {data['archive_path']} | "
            "Includes market observations, exact trades, assumptions, "
            "results, source files, package versions and SHA-256 hashes. "
            "Hashes detect accidental changes; this is not "
            "a tamper-proof audit system."
        )

    @render.download(filename="validated-run.json")
    def validation_download():
        data = result()
        yield (
            json.dumps(data, indent=2)
            if data else '{"status":"Run analysis first"}'
        )

    @render.text
    def market_stamp():
        data = snapshot()
        if not data:
            return "No saved prices. Refresh to load."

        from datetime import datetime
        fetched = datetime.fromisoformat(
            data["fetched_utc"]
        ).astimezone()

        return (
            f"Price date: {data['as_of']} | "
            f"Retrieved: {fetched.strftime('%d %b %H:%M:%S %Z')}"
        )

    @render.ui
    def operation_status():
        text = notice()
        lowered = text.lower()

        if "failed" in lowered or "error" in lowered:
            return ui.div(
                text, class_="alert alert-danger"
            )

        if result() is None:
            # Avoid showing the original startup instructions
            # when a saved snapshot is already available.
            if snapshot() and "fetch a snapshot" in lowered:
                return None
            return ui.div(
                text, class_="alert alert-info"
            )

        return None

    @render.text
    def run_label():
        data = result()
        if not data:
            return "Run an analysis to display results."

        return (
            f"Displayed: {data['request']} | "
            f"Price date: {data['snapshot']['as_of']} | "
            f"Base rate: {data['assumed_base_rate']:.2%} | "
            f"Scenario rate: {data['scenario_rate']:.2%}"
        )

    @render.table
    def compact_market():
        data = snapshot()
        if not data:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "Stock": symbol,
                "Close ($)": round(asset["spot"], 2),
                "Date": asset["date"],
                "Historical vol (%)": round(
                    asset["vol"] * 100, 2
                ),
            }
            for symbol, asset in data["assets"].items()
        ])

    @render.ui
    def validation_status_compact():
        data = result()
        if not data:
            return ui.p("Run an analysis first.")

        rows = (
            data["validation_status"]
            + data["reference_checks"]
        )
        failures = sum(
            row["Status"] == "FAIL" for row in rows
        )
        failures += sum(
            not row["Pass"] for row in data["checks"]
        )
        warnings = sum(
            row["Status"] == "WARNING" for row in rows
        )

        return ui.div(
            f"{failures} failures · {warnings} warnings",
            class_=(
                "alert alert-danger" if failures
                else "alert alert-warning" if warnings
                else "alert alert-success"
            ),
        )


app = App(app_ui, server)
