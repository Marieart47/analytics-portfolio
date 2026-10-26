"""
Project 02 — A/B test deep dive: "Daily rewards 2.0" for existing players.

The experiment is simulated with gamesim, so the TRUE effect is known and every method can be
checked against it.

Usage:  python 02-ab-test-deep-dive/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import abstats as ab  # noqa: E402
from gamesim.plotting import GREY, INK, PALETTE, save  # noqa: E402
from gamesim.simulator import Effects, SimConfig, make_players, simulate  # noqa: E402

FIG = HERE / "reports" / "figures"
OUT = HERE / "reports"
OUT.mkdir(parents=True, exist_ok=True)
START, END, PRE = 120, 134, 106  # test window [120, 134), pre-period [106, 120)
CFG = SimConfig(n_players=220_000, install_days=START, horizon_days=END, seed=2026)
C_COL, T_COL = PALETTE[0], PALETTE[1]

# ---------------------------------------------------------------------------------------------
# True treatment effect (unknown to the analyst):
#   * activity: +0.10 logit, plus a NOVELTY bump of +0.25 that decays with a 2.5-day time constant
#   * purchases: +0.15 logit for players who never paid (new "first-purchase" reward),
#                -0.05 for existing payers (the free daily gems slightly cannibalise purchases)
#   * sessions: +3% per active day
# ---------------------------------------------------------------------------------------------


def assign(player_id: int, salt: str = "daily_rewards_v2") -> int:
    """Deterministic 50/50 bucket from a salted hash (stable across sessions and devices)."""
    import hashlib

    return int(hashlib.md5(f"{salt}:{player_id}".encode()).hexdigest(), 16) % 100 < 50


def make_effects(treat: np.ndarray) -> Effects:
    def active_shift(d, ctx):
        if d < START:
            return 0.0
        return treat[ctx["idx"]] * (0.10 + 0.25 * np.exp(-(d - START) / 2.5))

    def buy_shift(d, ctx):
        if d < START:
            return 0.0
        return treat[ctx["idx"]] * np.where(ctx["n_buys"] > 0, -0.05, 0.15)

    def sessions_mult(d, ctx):
        if d < START:
            return 1.0
        return 1.0 + 0.03 * treat[ctx["idx"]]

    return Effects(active_shift=active_shift, buy_shift=buy_shift, sessions_mult=sessions_mult)


def user_table(res, players) -> pd.DataFrame:
    """One row per eligible player (active in the pre-period) with pre- and test-period metrics."""
    act, pur = res.activity, res.purchases
    pre_a = act[(act.day >= PRE) & (act.day < START)]
    eligible = pre_a.player_id.unique()
    u = players.set_index("player_id").loc[eligible, ["platform", "channel", "country_tier"]].copy()

    def agg(a, p, prefix):
        g = a.groupby("player_id").agg(active_days=("day", "nunique"), sessions=("sessions", "sum"))
        r = p.groupby("player_id").agg(revenue=("price_usd", "sum"), transactions=("price_usd", "size"))
        out = g.join(r, how="outer").reindex(eligible).fillna(0)
        return out.add_prefix(prefix)

    u = u.join(agg(pre_a, pur[(pur.day >= PRE) & (pur.day < START)], "pre_"))
    test_a = act[(act.day >= START) & (act.day < END)]
    u = u.join(agg(test_a, pur[(pur.day >= START) & (pur.day < END)], "test_"))
    u["test_paid"] = (u.test_revenue > 0).astype(int)
    u["test_week2_active"] = u.index.isin(test_a[test_a.day >= START + 7].player_id).astype(int)
    u["pre_payer"] = u.index.isin(pur[pur.day < START].player_id).astype(int)
    u["engagement"] = pd.cut(u.pre_active_days, [0, 3, 9, 14], labels=["low (1-3d)", "mid (4-9d)", "high (10-14d)"])
    return u


def daily_dau(res, ids) -> pd.Series:
    a = res.activity
    a = a[(a.day >= START) & (a.day < END) & a.player_id.isin(ids)]
    return a.groupby("day").player_id.nunique()


# ---------------------------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(CFG.seed)
    players, truth = make_players(CFG, rng)
    # randomisation: salted hash of player_id, as production split systems do
    treat = np.array([assign(pid) for pid in players.player_id], dtype=float)
    players["group"] = np.where(treat == 1, "treatment", "control")

    print("simulating experiment ...")
    res = simulate(CFG, make_effects(treat), players, truth)
    u = user_table(res, players).join(players.set_index("player_id")["group"])
    c, t = u[u.group == "control"], u[u.group == "treatment"]
    R: dict = {"n_control": len(c), "n_treatment": len(t)}

    # ---- oracle: the same population fully treated vs fully untreated, 4 seed pairs ----------
    print("simulating counterfactual oracle ...")
    truth_rows = []
    for s in range(4):
        cfg_s = SimConfig(**{**CFG.__dict__, "seed": 500 + s})
        r0 = simulate(cfg_s, make_effects(np.zeros(len(players))), players, truth)
        r1 = simulate(cfg_s, make_effects(np.ones(len(players))), players, truth)
        u0, u1 = user_table(r0, players), user_table(r1, players)
        row = {}
        for m in ["test_active_days", "test_revenue", "test_paid", "test_week2_active"]:
            row[m] = u1[m].mean() / u0[m].mean() - 1
        row["sessions_per_active_day"] = (u1.test_sessions.sum() / u1.test_active_days.sum()) / (
            u0.test_sessions.sum() / u0.test_active_days.sum()) - 1
        for seg_col in ["pre_payer", "engagement"]:
            for seg, g0 in u0.groupby(seg_col, observed=True):
                g1 = u1[u1[seg_col] == seg]
                row[f"{seg_col}={seg}|test_revenue"] = g1.test_revenue.mean() / g0.test_revenue.mean() - 1
                row[f"{seg_col}={seg}|test_active_days"] = g1.test_active_days.mean() / g0.test_active_days.mean() - 1
        d0, d1 = daily_dau(r0, u0.index), daily_dau(r1, u1.index)
        for d in range(START, END):
            row[f"dau_{d}"] = d1[d] / d0[d] - 1
        truth_rows.append(row)
    TRUE = pd.DataFrame(truth_rows).mean()
    TRUE_SE = pd.DataFrame(truth_rows).std() / 2
    R["true_effects"] = {k: round(float(v), 4) for k, v in TRUE.items() if not k.startswith("dau_")}

    # ---- 0. pre-registration: power & MDE ----------------------------------------------------
    n = min(len(c), len(t))
    power_rows = []
    for metric, cov in [("test_active_days", "pre_active_days"), ("test_revenue", "pre_revenue"),
                        ("test_paid", None), ("test_week2_active", "pre_active_days")]:
        y = c[metric].to_numpy(float)
        sd = y.std(ddof=1)
        row = {"metric": metric, "mean": y.mean(), "mde_rel": ab.mde(sd, n) / y.mean()}
        if cov:
            adj, _ = ab.cuped(y, c[cov].to_numpy(float))
            vr = 1 - adj.var() / y.var()
            row.update(cuped_var_reduction=vr, mde_rel_cuped=ab.mde(adj.std(ddof=1), n) / y.mean())
        power_rows.append(row)
    power = pd.DataFrame(power_rows)
    power.to_csv(OUT / "power_mde.csv", index=False)
    R["power"] = power.round(4).to_dict(orient="records")

    # ---- 1. SRM -------------------------------------------------------------------------------
    R["srm_p"] = ab.srm_test(len(c), len(t))

    # ---- 2. main metrics ----------------------------------------------------------------------
    rows = []

    def add(name, est, method, true_key):
        rows.append({"metric": name, "method": method, **est.as_dict(), "true_rel": TRUE[true_key],
                     "true_rel_se": TRUE_SE[true_key]})

    for m, cov, label in [("test_active_days", "pre_active_days", "Active days / user"),
                          ("test_week2_active", "pre_active_days", "Week-2 retention"),
                          ("test_paid", "pre_payer", "Payer conversion"),
                          ("test_revenue", "pre_revenue", "Revenue / user (ARPU)")]:
        add(label, ab.mean_test(c[m], t[m]), "t-test", m)
        est, theta, vr = ab.cuped_test(c[m].to_numpy(float), c[cov].to_numpy(float),
                                       t[m].to_numpy(float), t[cov].to_numpy(float))
        add(label, est, f"CUPED (var -{vr:.0%})", m)

    add("Sessions / active day", ab.ratio_test(c.test_sessions, c.test_active_days,
                                               t.test_sessions, t.test_active_days), "delta method",
        "sessions_per_active_day")
    # winsorised ARPU — robust to a single whale
    cap = np.quantile(u.test_revenue, 0.999)
    add("Revenue / user (ARPU)", ab.mean_test(c.test_revenue.clip(upper=cap), t.test_revenue.clip(upper=cap)),
        "winsorised p99.9", "test_revenue")
    res_tab = pd.DataFrame(rows)
    res_tab["rel_ci_low"] = res_tab.ci_low / res_tab.control
    res_tab["rel_ci_high"] = res_tab.ci_high / res_tab.control
    res_tab.to_csv(OUT / "results_table.csv", index=False)
    R["results"] = res_tab[["metric", "method", "rel_diff", "rel_ci_low", "rel_ci_high", "p_value",
                            "true_rel"]].round(4).to_dict(orient="records")
    lo, hi = ab.bootstrap_diff(c.test_revenue.to_numpy(), t.test_revenue.to_numpy(), n_boot=1000)
    R["arpu_bootstrap_ci_abs"] = [lo, hi]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = [f"{r.metric}\n{r.method}" for r in res_tab.itertuples()]
    ypos = np.arange(len(res_tab))[::-1]
    for y, r in zip(ypos, res_tab.itertuples()):
        sig = r.p_value < 0.05
        col = PALETTE[2] if sig else GREY
        ax.plot([r.rel_ci_low, r.rel_ci_high], [y, y], color=col, lw=2.2)
        ax.plot(r.rel_diff, y, "o", color=col, ms=6)
        ax.plot(r.true_rel, y, marker="|", color=PALETTE[3], ms=16, mew=2.2)
        ax.text(max(r.rel_ci_high, r.true_rel) + 0.004, y, f"p={r.p_value:.3f}", va="center", fontsize=8.5,
                color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(ypos, labels, fontsize=8.5)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x")
    ax.set_title("Relative treatment effect, 95% CI  (red tick = true effect from the oracle)")
    ax.plot([], [], "o-", color=PALETTE[2], label="significant (p<0.05)")
    ax.plot([], [], "o-", color=GREY, label="not significant")
    ax.legend(loc="upper right")
    ax.set_xlim(right=ax.get_xlim()[1] + 0.03)
    save(fig, FIG / "01_effects_forest.png")

    # ---- 3. novelty effect ------------------------------------------------------------------
    dc, dt = daily_dau(res, c.index) / len(c), daily_dau(res, t.index) / len(t)
    lift = dt / dc - 1
    se = np.sqrt((dt * (1 - dt)) / len(t) + (dc * (1 - dc)) / len(c)) / dc
    days = np.arange(1, len(lift) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(days, lift - 1.96 * se, lift + 1.96 * se, color=T_COL, alpha=0.18, label="95% CI")
    ax.plot(days, lift, "o-", color=T_COL, label="observed DAU lift")
    ax.plot(days, [TRUE[f"dau_{d}"] for d in range(START, END)], "--", color=PALETTE[3], label="true lift")
    w1, w2 = lift.iloc[:7].mean(), lift.iloc[7:].mean()
    ax.hlines([w1], 1, 7, color=INK, lw=1, ls=":")
    ax.hlines([w2], 8, 14, color=INK, lw=1, ls=":")
    box = dict(facecolor="white", edgecolor="none", pad=1.5)
    ax.text(4, w1 - 0.009, f"week 1 mean: {w1:+.1%}", ha="center", bbox=box)
    ax.text(11, w2 - 0.009, f"week 2 mean: {w2:+.1%}", ha="center", bbox=box)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("day of experiment")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Novelty effect: the DAU lift decays during the first week")
    ax.legend()
    save(fig, FIG / "02_novelty.png")
    R["dau_lift_week1"], R["dau_lift_week2"] = float(w1), float(w2)

    # ---- 4. heterogeneous effects ----------------------------------------------------------
    seg_rows = []
    for seg_col, cov_map in [("pre_payer", None), ("engagement", None), ("platform", None)]:
        for seg in u[seg_col].dropna().unique():
            cs, ts = c[c[seg_col] == seg], t[t[seg_col] == seg]
            for m, cov in [("test_revenue", "pre_revenue"), ("test_active_days", "pre_active_days")]:
                est, _, _ = ab.cuped_test(cs[m].to_numpy(float), cs[cov].to_numpy(float),
                                          ts[m].to_numpy(float), ts[cov].to_numpy(float))
                key = f"{seg_col}={seg}|{m}"
                seg_rows.append({"segment": f"{seg_col} = {seg}", "metric": m, "n": len(cs) + len(ts),
                                 **est.as_dict(), "true_rel": TRUE.get(key, np.nan)})
    seg = pd.DataFrame(seg_rows)
    seg["p_holm"] = np.nan
    for m in seg.metric.unique():
        mask = seg.metric == m
        seg.loc[mask, "p_holm"] = ab.holm(seg.loc[mask, "p_value"])
    seg["rel_lo"], seg["rel_hi"] = seg.ci_low / seg.control, seg.ci_high / seg.control
    seg.to_csv(OUT / "segments.csv", index=False)
    R["segments"] = seg[["segment", "metric", "n", "rel_diff", "rel_lo", "rel_hi", "p_value", "p_holm",
                         "true_rel"]].round(4).to_dict(orient="records")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, m, title in [(axes[0], "test_revenue", "Revenue / user"), (axes[1], "test_active_days", "Active days / user")]:
        s = seg[seg.metric == m].reset_index(drop=True)
        yy = np.arange(len(s))[::-1]
        for y, r in zip(yy, s.itertuples()):
            col = PALETTE[2] if r.p_holm < 0.05 else GREY
            ax.plot([r.rel_lo, r.rel_hi], [y, y], color=col, lw=2.2)
            ax.plot(r.rel_diff, y, "o", color=col)
            if not np.isnan(r.true_rel):
                ax.plot(r.true_rel, y, marker="|", color=PALETTE[3], ms=14, mew=2)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(yy, s.segment)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x")
        ax.set_title(f"{title} — CUPED, Holm-adjusted")
    save(fig, FIG / "03_segments.png")

    # ---- 5. A/A tests: why the delta method and fixed horizons matter -----------------------
    print("running A/A simulations ...")
    act = res.activity
    ca = act[(act.day >= START) & (act.day < END) & act.player_id.isin(c.index)]
    user_ids = c.index.to_numpy()
    sess_u = c.test_sessions.to_numpy(float)
    days_u = c.test_active_days.to_numpy(float)
    row_user = pd.Index(user_ids).get_indexer(ca.player_id)
    row_sess = ca.sessions.to_numpy(float)
    # cumulative active days per user per test day for peeking
    cum = np.zeros((len(user_ids), END - START))
    np.add.at(cum, (row_user, ca.day.to_numpy() - START), 1)
    cum = cum.cumsum(axis=1)

    rng = np.random.default_rng(1)
    n_sims = 1000
    rej = {"naive_session_level": 0, "delta_method": 0, "peeking_daily": 0, "fixed_horizon": 0}
    for _ in range(n_sims):
        g = rng.random(len(user_ids)) < 0.5
        # session-level naive test: each active-day row treated as independent
        gr = g[row_user]
        p_naive = ab.mean_test(row_sess[~gr], row_sess[gr]).p_value
        p_delta = ab.ratio_test(sess_u[~g], days_u[~g], sess_u[g], days_u[g]).p_value
        rej["naive_session_level"] += p_naive < 0.05
        rej["delta_method"] += p_delta < 0.05
        m1, m0 = cum[g].mean(0), cum[~g].mean(0)
        s = np.sqrt(cum[g].var(0, ddof=1) / g.sum() + cum[~g].var(0, ddof=1) / (~g).sum())
        z = np.abs(m1 - m0) / s
        rej["peeking_daily"] += (z[1:] > 1.96).any()  # peek from day 2 onwards
        rej["fixed_horizon"] += z[-1] > 1.96
    fpr = {k: v / n_sims for k, v in rej.items()}
    R["aa_false_positive_rate"] = fpr

    fig, ax = plt.subplots(figsize=(8.6, 4))
    names = ["Session-level\nt-test (naive)", "Delta method\n(user-level)", "Daily peeking\n(stop at p<.05)",
             "Fixed horizon\n(14 days)"]
    vals = [fpr["naive_session_level"], fpr["delta_method"], fpr["peeking_daily"], fpr["fixed_horizon"]]
    cols = [PALETTE[3], PALETTE[2], PALETTE[3], PALETTE[2]]
    ax.bar(names, vals, color=cols, width=0.6)
    ax.axhline(0.05, color=INK, ls="--", lw=1)
    ax.text(-0.42, 0.054, "nominal α = 5%", ha="left", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.1%}", ha="center", fontweight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(f"False-positive rate in {n_sims} A/A tests on real experiment data")
    save(fig, FIG / "04_aa_tests.png")

    # ---- 6. SRM scenario: a logging bug in the new Android client ---------------------------
    # Exposure is logged on the first session of the test. The new client drops the exposure event
    # for 50% of Android treatment users with <=2 active days (crash on cold start).
    bug_rng = np.random.default_rng(5)
    lost = (u.group == "treatment") & (u.platform == "android") & (u.test_active_days <= 2) & (
        bug_rng.random(len(u)) < 0.5)
    exposed = u[(u.test_active_days > 0) & ~lost]
    cb, tb = exposed[exposed.group == "control"], exposed[exposed.group == "treatment"]
    est_bug = ab.mean_test(cb.test_active_days, tb.test_active_days)
    exposed_ok = u[u.test_active_days > 0]
    est_ok = ab.mean_test(exposed_ok[exposed_ok.group == "control"].test_active_days,
                          exposed_ok[exposed_ok.group == "treatment"].test_active_days)
    R["srm_bug"] = {"n_control": len(cb), "n_treatment": len(tb), "srm_p": ab.srm_test(len(cb), len(tb)),
                    "lift_with_bug": est_bug.rel_diff, "lift_without_bug_same_population": est_ok.rel_diff,
                    "lift_itt": float(res_tab.query("metric=='Active days / user' and method=='t-test'").rel_diff.iloc[0]),
                    "true_itt": float(TRUE["test_active_days"])}

    # ---- 7. decision ----------------------------------------------------------------------
    arpu = res_tab[(res_tab.metric == "Revenue / user (ARPU)") & res_tab.method.str.startswith("CUPED")].iloc[0]
    eligible_base = len(u)
    monthly = arpu.abs_diff * eligible_base * 30 / 14
    R["decision"] = {"eligible_players": eligible_base, "arpu_lift_14d_abs": float(arpu.abs_diff),
                     "arpu_lift_ci": [float(arpu.ci_low), float(arpu.ci_high)],
                     "monthly_revenue_impact": float(monthly),
                     "monthly_ci": [float(arpu.ci_low * eligible_base * 30 / 14),
                                    float(arpu.ci_high * eligible_base * 30 / 14)]}

    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    print(json.dumps({k: R[k] for k in ["n_control", "n_treatment", "srm_p", "aa_false_positive_rate",
                                        "srm_bug", "decision", "dau_lift_week1", "dau_lift_week2"]},
                     indent=2, default=float))
    print(res_tab[["metric", "method", "rel_diff", "rel_ci_low", "rel_ci_high", "p_value", "true_rel"]].round(4))
    print(power.round(4))


if __name__ == "__main__":
    main()
