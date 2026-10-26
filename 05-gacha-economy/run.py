"""
Project 05 — Gacha banner economy: pity systems, fairness and revenue.

Usage:  python 05-gacha-economy/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from gacha import Banner, featured_pmf, first_5star_pmf, simulate_featured, summarize  # noqa: E402
from gamesim.plotting import GREY, INK, PALETTE, save  # noqa: E402

FIG = HERE / "reports" / "figures"
OUT = HERE / "reports"
OUT.mkdir(parents=True, exist_ok=True)

USD_PER_PULL = 2.00          # 160 premium gems per pull at the best-value pack price
N_PLAYERS = 300_000
F2P_PULLS_PER_PATCH = 40     # free income per 6-week content patch (assumption, see README)

BASE = Banner("Baseline: soft pity 74, hard 90, 50/50 + guarantee")
_mean_first = summarize(first_5star_pmf(BASE))["mean"]
DESIGNS = [
    BASE,
    Banner("No pity, flat rate (same mean)", base_rate=1 / _mean_first, soft_pity=None, hard_pity=None),
    Banner("Hard pity 90 only, no soft pity", soft_pity=None),
    Banner("Generous: soft 64, hard 80", soft_pity=64, hard_pity=80),
    Banner("Baseline + spark at 150 pulls", spark=150),
]
SHORT = ["Baseline", "No pity", "Hard pity only", "Generous", "Spark 150"]
COLORS = dict(zip(SHORT, [PALETTE[0], PALETTE[3], PALETTE[1], PALETTE[2], PALETTE[4]]))


def theoretical_worst(b: Banner) -> float:
    if b.hard_pity is None and b.spark is None:
        return np.inf
    w = b.hard_pity * (2 if (b.guarantee and b.featured_share < 1) else 1) if b.hard_pity else np.inf
    return min(w, b.spark) if b.spark else w


def population(rng: np.random.Generator, n: int) -> pd.DataFrame:
    seg = rng.choice(["F2P", "dolphin", "whale"], size=n, p=[0.75, 0.20, 0.05])
    budget = np.where(seg == "F2P", 0.0,
                      np.where(seg == "dolphin", rng.lognormal(np.log(60), 0.8, n), rng.lognormal(np.log(600), 0.9, n)))
    free = np.clip(np.round(rng.lognormal(np.log(60), 0.5, n)), 10, 200)
    return pd.DataFrame({"segment": seg, "budget_usd": budget, "free_pulls": free, "u": rng.random(n)})


def play_banner(pop: pd.DataFrame, pmf: np.ndarray) -> pd.DataFrame:
    """Each player pulls until they get the featured unit or run out of free pulls + budget.
    Pulls needed are drawn by inverse CDF from the exact distribution with the SAME uniform per player
    across designs (common random numbers -> lower-variance comparisons)."""
    cdf = np.cumsum(pmf)
    need = np.searchsorted(cdf, pop.u.to_numpy() * cdf[-1]) + 1
    affordable = pop.free_pulls.to_numpy() + np.floor(pop.budget_usd.to_numpy() / USD_PER_PULL)
    success = need <= affordable
    paid_pulls = np.where(success, np.maximum(need - pop.free_pulls.to_numpy(), 0),
                          affordable - pop.free_pulls.to_numpy())
    spend = paid_pulls * USD_PER_PULL
    out = pop.copy()
    out["need"], out["success"], out["spend"] = need, success, spend
    out["regret"] = (~success) & (spend >= 50)
    return out


def main() -> None:
    R: dict = {"usd_per_pull": USD_PER_PULL}
    rng = np.random.default_rng(2026)

    # ---- 1. exact distributions + Monte Carlo verification ----------------------------------
    rows, pmfs = [], {}
    for b, s in zip(DESIGNS, SHORT):
        pmf = featured_pmf(b, max_pulls=6000)
        pmfs[s] = pmf
        ex = summarize(pmf)
        mc = simulate_featured(b, 200_000, rng)
        rows.append({"design": s, "description": b.name, **{k: ex[k] for k in ["mean", "sd", "median", "p90", "p99"]},
                     "worst_case": theoretical_worst(b), "p_over_150": ex["p_over_150"],
                     "mc_mean": mc.mean(), "mc_p90": float(np.quantile(mc, 0.9)),
                     "mean_cost_usd": ex["mean"] * USD_PER_PULL, "p99_cost_usd": ex["p99"] * USD_PER_PULL})
    dist = pd.DataFrame(rows)
    dist.to_csv(OUT / "distributions.csv", index=False)
    R["distributions"] = dist.round(3).replace([np.inf], "unbounded").to_dict(orient="records")
    print(dist.round(2).to_string())

    # hazard + first-5★ distribution for the baseline, DP vs MC
    n = np.arange(1, 91)
    first = first_5star_pmf(BASE, 90)
    mc_first = []
    pity = np.zeros(200_000, dtype=int)
    got = np.full(200_000, 0)
    active = np.arange(200_000)
    for k in range(1, 91):
        hit = rng.random(active.size) < BASE.rate(k)
        got[active[hit]] = k
        active = active[~hit]
    mc_hist = np.bincount(got, minlength=91)[1:] / 200_000
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(n, BASE.rate(n), color=PALETTE[0], lw=2)
    axes[0].axvspan(74, 90, color=PALETTE[1], alpha=0.1)
    axes[0].text(72, 0.55, "soft pity:\n+6 pp per pull", color=INK, fontsize=9, ha="right")
    axes[0].set_title("Per-pull 5★ probability (hazard)")
    axes[0].set_xlabel("pulls since last 5★")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].bar(n, mc_hist, color=PALETTE[6], width=1.0, label="Monte Carlo, 200k players")
    axes[1].plot(n, first, color=INK, lw=1.4, label="exact (Markov chain)")
    axes[1].set_title(f"Pulls to first 5★: mean {summarize(first)['mean']:.1f}, "
                      f"{first[73:].sum():.0%} land in soft pity")
    axes[1].set_xlabel("pull number")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    axes[1].legend(loc="upper left")
    save(fig, FIG / "01_baseline_pity.png")
    R["baseline_first5_mean"] = float(summarize(first)["mean"])
    R["share_first5_in_soft_pity"] = float(first[73:].sum())
    R["consolidated_rate"] = float(1 / summarize(first)["mean"])

    # ---- 2. distribution of pulls to the featured unit across designs --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    t = np.arange(1, 6001)
    for s in SHORT:
        pmf = pmfs[s]
        axes[0].plot(t[:400], np.cumsum(pmf)[:400], color=COLORS[s], lw=2, label=s)
    axes[0].axvline(F2P_PULLS_PER_PATCH * 2, color=GREY, ls=":", lw=1)
    axes[0].text(F2P_PULLS_PER_PATCH * 2 + 3, 0.05, "F2P savings,\n2 patches", fontsize=8.5, color=INK)
    axes[0].set_xlabel("pulls")
    axes[0].set_ylabel("P(featured unit obtained)")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("CDF: chance of getting the featured unit")
    axes[0].legend(fontsize=8.5, loc="lower right")
    x = np.arange(len(dist))
    axes[1].bar(x, dist.p99 * USD_PER_PULL, color=[COLORS[s] for s in SHORT], alpha=0.35, label="p99")
    axes[1].bar(x, dist.p90 * USD_PER_PULL, color=[COLORS[s] for s in SHORT], alpha=0.7, label="p90")
    axes[1].bar(x, dist["mean"] * USD_PER_PULL, color=[COLORS[s] for s in SHORT], label="mean")
    for i, r in dist.iterrows():
        axes[1].text(i, r.p99 * USD_PER_PULL + 8, f"${r.p99 * USD_PER_PULL:.0f}", ha="center", fontsize=8.5)
    axes[1].set_xticks(x, SHORT, fontsize=8.5)
    axes[1].set_title("Cost of a guaranteed copy if every pull is bought, $ (mean / p90 / p99)")
    axes[1].legend(fontsize=8.5)
    save(fig, FIG / "02_designs_distribution.png")

    # ---- 3. revenue vs player experience -----------------------------------------------------
    pop = population(rng, N_PLAYERS)
    econ = []
    played = {}
    for s in SHORT:
        g = play_banner(pop, pmfs[s])
        played[s] = g
        payers = g[g.segment != "F2P"]
        econ.append({
            "design": s,
            "revenue_per_player": g.spend.mean(),
            "success_all": g.success.mean(),
            "success_f2p": g[g.segment == "F2P"].success.mean(),
            "success_payers": payers.success.mean(),
            "regret_share_payers": payers.regret.mean(),
            "spend_whales": g[g.segment == "whale"].spend.mean(),
            "spend_dolphins": g[g.segment == "dolphin"].spend.mean(),
        })
    econ = pd.DataFrame(econ)
    base_rev = econ.set_index("design").loc["Baseline", "revenue_per_player"]
    econ["revenue_vs_baseline"] = econ.revenue_per_player / base_rev - 1

    # Long-term view (explicit assumption): payers who spend >= $50 and still fail churn with
    # probability 35% (vs 8% otherwise); a retained payer is worth 3 more banners of similar spend.
    CHURN_REGRET, CHURN_BASE, FUTURE_BANNERS = 0.35, 0.08, 3
    lt = []
    for s in SHORT:
        g = played[s]
        payer_future = g.budget_usd.clip(upper=g.budget_usd.quantile(0.999)) * 0.5  # expected future spend/banner
        churn = np.where(g.regret, CHURN_REGRET, CHURN_BASE)
        lt.append((g.spend + (1 - churn) * payer_future * FUTURE_BANNERS * (g.segment != "F2P")).mean())
    econ["lifetime_adjusted_revenue"] = lt
    econ["lifetime_vs_baseline"] = econ.lifetime_adjusted_revenue / econ.set_index("design").loc[
        "Baseline", "lifetime_adjusted_revenue"] - 1
    econ.to_csv(OUT / "economics.csv", index=False)
    R["economics"] = econ.round(4).to_dict(orient="records")
    print(econ.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for _, r in econ.iterrows():
        axes[0].scatter(r.success_payers, r.revenue_per_player, s=140, color=COLORS[r.design], zorder=3)
        axes[0].annotate(r.design, (r.success_payers, r.revenue_per_player), (8, 6), textcoords="offset points")
    axes[0].set_xlabel("payers who obtain the featured unit")
    axes[0].set_ylabel("banner revenue per player, $")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("Short term: revenue vs payer success")
    axes[0].set_xlim(econ.success_payers.min() - 0.01, econ.success_payers.max() + 0.03)
    axes[0].grid(axis="x")
    x = np.arange(len(econ))
    w = 0.38
    axes[1].bar(x - w / 2, econ.revenue_vs_baseline, w, color=PALETTE[1], label="banner revenue")
    axes[1].bar(x + w / 2, econ.lifetime_vs_baseline, w, color=PALETTE[0], label="lifetime-adjusted (assumption)")
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_xticks(x, econ.design, fontsize=8.5)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Revenue vs baseline")
    axes[1].legend(fontsize=8.5, loc="upper right")
    save(fig, FIG / "03_revenue_vs_experience.png")

    # ---- 4. sensitivity of the lifetime result to the churn assumption -------------------------
    sens = []
    for cr in [0.10, 0.20, 0.35, 0.50, 0.70, 0.90]:
        vals = {}
        for s in SHORT:
            g = played[s]
            payer_future = g.budget_usd.clip(upper=g.budget_usd.quantile(0.999)) * 0.5
            churn = np.where(g.regret, cr, CHURN_BASE)
            vals[s] = (g.spend + (1 - churn) * payer_future * FUTURE_BANNERS * (g.segment != "F2P")).mean()
        sens.append({"churn_after_regret": cr, **{s: vals[s] / vals["Baseline"] - 1 for s in SHORT}})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "sensitivity.csv", index=False)
    R["sensitivity"] = sens.round(4).to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for s in SHORT[1:]:
        ax.plot(sens.churn_after_regret, sens[s], "o-", color=COLORS[s], label=s)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(CHURN_REGRET, color=GREY, ls=":", lw=1)
    ax.set_xlabel("churn probability of a payer who spent ≥ $50 and failed")
    ax.set_ylabel("lifetime-adjusted revenue vs baseline")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Lifetime-adjusted revenue vs baseline: the ranking survives any churn assumption")
    ax.legend(fontsize=8.5)
    save(fig, FIG / "04_sensitivity.png")

    R["f2p"] = {s: {"p_featured_with_1_patch": float(np.cumsum(pmfs[s])[F2P_PULLS_PER_PATCH - 1]),
                    "p_featured_with_2_patches": float(np.cumsum(pmfs[s])[2 * F2P_PULLS_PER_PATCH - 1]),
                    "p_featured_with_3_patches": float(np.cumsum(pmfs[s])[3 * F2P_PULLS_PER_PATCH - 1])}
                for s in SHORT}
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    print(json.dumps(R["f2p"], indent=1))
    print(sens.round(3))


if __name__ == "__main__":
    main()
