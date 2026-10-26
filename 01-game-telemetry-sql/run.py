"""
Project 01 — Game telemetry & SQL metrics layer.

1. Generates a synthetic F2P telemetry dataset (60k players, 160 days) with gamesim.
2. Creates a second copy with deliberately injected data defects.
3. Runs the SQL data-quality suite on both copies (it must pass on clean data and catch every defect).
4. Runs the KPI SQL layer in DuckDB and renders the report figures / tables.

Usage:  python 01-game-telemetry-sql/run.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamesim.plotting import GREY, PALETTE, save  # noqa: E402
from gamesim.simulator import SimConfig, simulate  # noqa: E402

HERE = Path(__file__).resolve().parent
SQL = HERE / "sql"
FIG = HERE / "reports" / "figures"
TAB = HERE / "reports" / "tables"
DATA = ROOT / "data" / "telemetry"
TABLES = ["players", "activity", "purchases", "tutorial"]


# ----------------------------------------------------------------------------------------------
def generate() -> None:
    clean = DATA / "clean"
    if not (clean / "players.parquet").exists():
        print("simulating 60k players ...")
        simulate(SimConfig(n_players=60_000, seed=42)).save(clean)
    dirty = DATA / "dirty"
    if (dirty / "players.parquet").exists():
        return
    rng = np.random.default_rng(7)
    dirty.mkdir(parents=True, exist_ok=True)
    players = pd.read_parquet(clean / "players.parquet")
    activity = pd.read_parquet(clean / "activity.parquet")
    purchases = pd.read_parquet(clean / "purchases.parquet")
    tutorial = pd.read_parquet(clean / "tutorial.parquet")

    # (a) client retries -> exact duplicate transactions (0.5%)
    dup = purchases.sample(frac=0.005, random_state=1)
    # (b) device clock skew -> purchases timestamped 1-3 days before install
    skew_idx = purchases.sample(200, random_state=2).index
    purchases.loc[skew_idx, "event_date"] = purchases.loc[skew_idx, "player_id"].map(
        players.set_index("player_id")["install_date"]) - pd.to_timedelta(rng.integers(1, 4, 200), unit="D")
    # (c) events from test devices that never sent an install event
    orphans = activity.sample(150, random_state=3).assign(player_id=lambda d: d["player_id"] + 10_000_000)
    # (d) zero-price purchases from a broken promo
    purchases.loc[purchases.sample(50, random_state=4).index, "price_usd"] = 0.0
    # (e) lost tutorial_start events for 0.3% of installs
    lost = players.sample(frac=0.003, random_state=5)["player_id"]
    tutorial = tutorial[~((tutorial["step_no"] == 1) & tutorial["player_id"].isin(lost))]

    pd.concat([purchases, dup]).to_parquet(dirty / "purchases.parquet", index=False)
    pd.concat([activity, orphans]).to_parquet(dirty / "activity.parquet", index=False)
    players.to_parquet(dirty / "players.parquet", index=False)
    tutorial.to_parquet(dirty / "tutorial.parquet", index=False)


def connect(folder: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in TABLES:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{folder / (t + '.parquet')}')")
    # cast timestamps to DATE once, so SQL files can compare dates directly
    return con


def q(con, name: str) -> pd.DataFrame:
    return con.execute((SQL / f"{name}.sql").read_text()).df()


def run_dq(con) -> list[tuple[str, int]]:
    text = (SQL / "dq_checks.sql").read_text()
    blocks = re.split(r"^-- check: ", text, flags=re.M)[1:]
    out = []
    for b in blocks:
        name, body = b.split("\n", 1)
        out.append((name.strip(), int(con.execute(body).fetchone()[0])))
    return out


# ----------------------------------------------------------------------------------------------
def main() -> None:
    generate()

    # ---- data quality ------------------------------------------------------------------------
    dq_clean = run_dq(connect(DATA / "clean"))
    dq_dirty = run_dq(connect(DATA / "dirty"))
    dq = pd.DataFrame({"check": [c for c, _ in dq_clean],
                       "clean_data": [v for _, v in dq_clean],
                       "defect_injected_data": [v for _, v in dq_dirty]})
    dq.to_csv(TAB / "dq_checks.csv", index=False)
    print(dq.to_string(index=False))

    con = connect(DATA / "clean")
    res: dict = {}

    # ---- daily KPIs --------------------------------------------------------------------------
    kpi = q(con, "daily_kpis")
    kpi.to_csv(TAB / "daily_kpis.csv", index=False)
    stick = q(con, "stickiness")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True)
    axes[0].fill_between(kpi.event_date, kpi.dau, color=PALETTE[0], alpha=0.18)
    axes[0].plot(kpi.event_date, kpi.dau, color=PALETTE[0], lw=1.6)
    axes[0].set_title("DAU — installs stop on day 60, so the base starts to decay")
    axes[1].bar(kpi.event_date, kpi.revenue, color=PALETTE[1], width=0.9)
    axes[1].plot(kpi.event_date, kpi.revenue.rolling(7, center=True).mean(), color=INK_DARK, lw=1.4,
                 label="7-day mean")
    axes[1].set_title("Daily gross bookings, $")
    axes[1].legend(loc="upper left")
    axes[2].plot(stick.event_date, stick.stickiness, color=PALETTE[2], lw=1.6)
    axes[2].set_title("Stickiness: DAU / MAU(28d)")
    axes[2].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    save(fig, FIG / "01_daily_kpis.png")

    # ---- retention -------------------------------------------------------------------------
    curve = q(con, "retention_curve_by_channel")
    curve.to_csv(TAB / "retention_curve_by_channel.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (ch, g) in enumerate(curve[curve.age >= 1].groupby("channel")):
        ax.plot(g.age, g.retention, label=ch, color=PALETTE[i], lw=1.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 7, 14, 30, 60, 90], ["D1", "D3", "D7", "D14", "D30", "D60", "D90"])
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Retention curve by acquisition channel (censoring-aware)")
    ax.set_ylabel("share of installs active on day N")
    ax.legend()
    save(fig, FIG / "02_retention_by_channel.png")

    coh = q(con, "retention_cohorts")
    coh.to_csv(TAB / "retention_cohorts.csv", index=False)
    pivot = coh.pivot(index="cohort_week", columns="day_n", values="retention")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=0, vmax=0.4)
    ax.set_xticks(range(pivot.shape[1]), [f"D{c}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [d.strftime("%d %b") for d in pivot.index])
    for (r, c), v in np.ndenumerate(pivot.values):
        if not np.isnan(v):
            ax.text(c, r, f"{v:.1%}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.25 else "#1F2430")
    ax.grid(False)
    ax.set_title("Weekly install cohorts × N-day retention (blank = not observable yet)")
    fig.colorbar(im, ax=ax, format=plt.matplotlib.ticker.PercentFormatter(1), shrink=0.8)
    save(fig, FIG / "03_cohort_heatmap.png")

    overall = curve.groupby("age").apply(lambda g: (g.retention * g.eligible).sum() / g.eligible.sum(),
                                         include_groups=False)
    res["retention"] = {f"D{d}": round(float(overall[d]), 4) for d in [1, 7, 30, 90]}

    # ---- funnel ----------------------------------------------------------------------------
    fun = q(con, "tutorial_funnel")
    fun.to_csv(TAB / "tutorial_funnel.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    steps = fun.step.unique()
    w = 0.2
    for i, (ch, g) in enumerate(fun.groupby("channel")):
        ax.bar(np.arange(len(steps)) + (i - 1.5) * w, g.conv_from_start, w, label=ch, color=PALETTE[i])
    ax.set_xticks(range(len(steps)), steps)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Onboarding funnel — share of installs reaching each step")
    ax.legend(ncol=4, loc="lower left")
    save(fig, FIG / "04_tutorial_funnel.png")
    res["tutorial_completion"] = round(float(
        fun[fun.step == "tutorial_complete"].players.sum() / fun[fun.step_no == 1].players.sum()), 4)

    # ---- monetisation ----------------------------------------------------------------------
    ltv = q(con, "ltv_roas_by_channel")
    ltv.to_csv(TAB / "ltv_roas_by_channel.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    for i, (ch, g) in enumerate(ltv.groupby("channel")):
        axes[0].plot(g.age, g.cum_arpu, marker="o", ms=3.5, label=ch, color=PALETTE[i])
    axes[0].set_title("Cumulative gross ARPU by days since install, $")
    axes[0].set_xlabel("days since install")
    axes[0].legend()
    d90 = ltv[(ltv.age == 90) & (ltv.cpi > 0)].copy()
    d90["net_roas"] = d90.gross_roas * 0.7
    colors = [PALETTE[list(ltv.channel.unique()).index(c)] for c in d90.channel]
    axes[1].bar(d90.channel, d90.net_roas, color=colors)
    axes[1].axhline(1, color=GREY, ls="--", lw=1)
    for x, v in enumerate(d90.net_roas):
        axes[1].text(x, v + 0.02, f"{v:.0%}", ha="center")
    axes[1].set_title("Net D90 ROAS (after 30% store fee)")
    axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    save(fig, FIG / "05_ltv_roas.png")
    res["net_d90_roas"] = {r.channel: round(float(r.net_roas), 3) for r in d90.itertuples()}
    res["gross_arpu_d90"] = round(float(
        (ltv[ltv.age == 90].cum_arpu * ltv[ltv.age == 90].installs).sum() / ltv[ltv.age == 90].installs.sum()), 3)

    tiers = q(con, "payer_concentration")
    tiers.to_csv(TAB / "payer_concentration.csv", index=False)
    par = q(con, "pareto_curve")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(par.payer_share, par.revenue_share, color=PALETTE[3], lw=2)
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
    top10 = float(np.interp(0.10, par.payer_share, par.revenue_share))
    ax.scatter([0.10], [top10], color=PALETTE[3], zorder=3)
    ax.annotate(f"top 10% of payers → {top10:.0%} of revenue", (0.10, top10), (0.25, top10 - 0.12),
                arrowprops=dict(arrowstyle="-", color=GREY))
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel("share of payers (sorted by spend)")
    ax.set_ylabel("share of revenue")
    ax.set_title("Revenue concentration (Pareto curve)")
    save(fig, FIG / "06_pareto.png")
    res["top10pct_payers_revenue_share"] = round(top10, 3)
    res["payer_tiers"] = tiers.round(4).to_dict(orient="records")
    res["players"] = int(con.execute("SELECT COUNT(*) FROM players").fetchone()[0])
    res["payer_conversion_lifetime"] = round(float(con.execute(
        "SELECT COUNT(DISTINCT player_id) FROM purchases").fetchone()[0]) / res["players"], 4)
    res["dq"] = dq.to_dict(orient="records")

    (HERE / "reports" / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("dq", "payer_tiers")}, indent=2))


INK_DARK = "#1F2430"

if __name__ == "__main__":
    main()
