"""
Project 06 — Economy design & balance of a medieval city-management game.

Usage:  python 06-medieval-trade-sim/run.py
"""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from economy import CITY_NAMES, STRATEGIES, V1, V2, play  # noqa: E402
from gamesim.plotting import GREY, INK, PALETTE, save  # noqa: E402

FIG = HERE / "reports" / "figures"
OUT = HERE / "reports"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = list(STRATEGIES)
S_COL = dict(zip(NAMES, [PALETTE[1], PALETTE[0], PALETTE[3], PALETTE[4], PALETTE[5]]))
FAUCETS = ["taxes", "sales"]
DRAINS = ["purchases", "upkeep", "construction", "recruitment", "bandits"]


def tournament(rules, seeds: int = 8, keep_money: bool = False):
    """Every multiset of 3 strategies (35 line-ups) × seeds, with random seating on the map."""
    rows, money, flows = [], [], []
    for combo in itertools.combinations_with_replacement(NAMES, 3):
        for s in range(seeds):
            seat = list(np.random.default_rng(s).permutation(combo))
            w = play(rules, seat, seed=s)
            sc = [w.score(c) for c in w.cities]
            win = int(np.argmax(sc))
            for i, c in enumerate(w.cities):
                rows.append({"strategy": c.strategy.name, "city": CITY_NAMES[i], "score": sc[i], "win": int(i == win),
                             "pop": c.pop, "gold": c.gold, "stability": c.stability})
            if keep_money:
                pop0 = 3000.0
                money.append(np.array(w.money) / pop0)
                flows.append({k: v / rules.turns for k, v in w.flows.items()})
    return pd.DataFrame(rows), np.array(money), pd.DataFrame(flows)


def win_table(d):
    g = d.groupby("strategy").win
    t = pd.DataFrame({"win_rate": g.mean(), "n": g.size()})
    t["se"] = np.sqrt(t.win_rate * (1 - t.win_rate) / t.n)
    return t


def main() -> None:
    R: dict = {}
    print("tournaments v1 / v2 ...")
    d1, m1, f1 = tournament(V1, keep_money=True)
    d2, m2, f2 = tournament(V2, keep_money=True)
    w1, w2 = win_table(d1), win_table(d2)
    c1, c2 = d1.groupby("city").win.mean(), d2.groupby("city").win.mean()
    R["win_rates"] = {"v1": w1.win_rate.round(3).to_dict(), "v2": w2.win_rate.round(3).to_dict(),
                      "v1_city": c1.round(3).to_dict(), "v2_city": c2.round(3).to_dict(),
                      "games_per_version": len(d1) // 3}
    R["outcomes"] = {"v1": d1.groupby("strategy")[["pop", "gold", "stability"]].mean().round(2).to_dict(),
                     "v2": d2.groupby("strategy")[["pop", "gold", "stability"]].mean().round(2).to_dict()}

    # ---- 1. win rates ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(NAMES))
    w = 0.38
    for off, t, lab, col in [(-w / 2, w1, "v1 (launch rules)", PALETTE[3]), (w / 2, w2, "v2 (after balance pass)", PALETTE[0])]:
        axes[0].bar(x + off, t.win_rate.loc[NAMES], w, yerr=1.96 * t.se.loc[NAMES], color=col, label=lab,
                    error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
    axes[0].axhline(1 / 3, color=INK, ls="--", lw=1)
    axes[0].text(4.6, 1 / 3 + 0.015, "fair = 1/3", ha="right", fontsize=8.5)
    axes[0].set_xticks(x, NAMES)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("Win rate by strategy (3-player games, 95% CI)")
    axes[0].legend()
    xc = np.arange(3)
    axes[1].bar(xc - w / 2, c1.loc[CITY_NAMES], w, color=PALETTE[3])
    axes[1].bar(xc + w / 2, c2.loc[CITY_NAMES], w, color=PALETTE[0])
    axes[1].axhline(1 / 3, color=INK, ls="--", lw=1)
    axes[1].set_xticks(xc, CITY_NAMES)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Win rate by starting city (map balance)")
    save(fig, FIG / "01_win_rates.png")

    # ---- 2. money supply ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.4))
    t = np.arange(m1.shape[1])
    for m, lab, col in [(m1, "v1", PALETTE[3]), (m2, "v2", PALETTE[0])]:
        med = np.median(m, axis=0)
        ax.fill_between(t, np.quantile(m, 0.1, axis=0), np.quantile(m, 0.9, axis=0), color=col, alpha=0.15)
        ax.plot(t, med, color=col, lw=2, label=f"{lab}: ×{med[-1] / med[0]:.0f} in 10 years")
    ax.set_yscale("log")
    ax.set_xlabel("month")
    ax.set_ylabel("gold held by mayors, per starting inhabitant")
    ax.set_title("Money supply (median and 10–90% band across all games)")
    ax.legend()
    save(fig, FIG / "02_money_supply.png")
    R["money_growth_10y"] = {"v1": float(np.median(m1[:, -1] / m1[:, 0])), "v2": float(np.median(m2[:, -1] / m2[:, 0]))}

    # ---- 3. faucets & drains ----------------------------------------------------------------
    fl = pd.DataFrame({"v1": f1.mean(), "v2": f2.mean()})
    R["flows_per_month"] = fl.round(1).to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, v in zip(axes, ["v1", "v2"]):
        vals_f = fl.loc[FAUCETS, v]
        vals_d = fl.loc[DRAINS, v]
        left = 0
        for i, (k, val) in enumerate(vals_f.items()):
            ax.barh(1, val, left=left, color=[PALETTE[2], PALETTE[6]][i], label=f"faucet: {k}")
            left += val
        left = 0
        for i, (k, val) in enumerate(vals_d.items()):
            ax.barh(0, val, left=left, color=[PALETTE[3], PALETTE[1], PALETTE[4], PALETTE[5], GREY][i], label=f"drain: {k}")
            left += val
        ax.set_yticks([0, 1], ["drains", "faucets"])
        ax.grid(axis="x")
        net = vals_f.sum() - vals_d.sum()
        ax.set_title(f"{v}: gold per game-month (net {net:+,.0f})")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    save(fig, FIG / "03_faucets_drains.png")

    # ---- 4. the exploit: craft-and-sell arbitrage ------------------------------------------
    ex = {}
    for rules in [V1, V2]:
        wgame = play(rules, ["Merchant", "Builder", "Zealot"], seed=1, record=True)
        h = pd.DataFrame(wgame.history)
        ex[rules.version] = h
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for v, col in [("v1", PALETTE[3]), ("v2", PALETTE[0])]:
        hm = ex[v][ex[v].strategy == "Merchant"]
        axes[0].plot(hm.t, hm.workshops, color=col, lw=2, label=v)
        axes[1].plot(hm.t, hm.gold.clip(lower=1), color=col, lw=2, label=v)
    axes[0].set_title("Merchant's workshops: v1 compounds one per month")
    axes[0].set_xlabel("month")
    axes[1].set_yscale("log")
    axes[1].set_title("Merchant's gold")
    axes[1].set_xlabel("month")
    axes[0].legend()
    save(fig, FIG / "04_arbitrage_exploit.png")
    v1m = ex["v1"][ex["v1"].strategy == "Merchant"].iloc[-1]
    v2m = ex["v2"][ex["v2"].strategy == "Merchant"].iloc[-1]
    R["exploit"] = {"v1_workshops": int(v1m.workshops), "v2_workshops": int(v2m.workshops),
                    "weapon_margin_v1": V1.base_prices[4] * 0.9 - (V1.base_prices[2] + V1.base_prices[1]) * 1.1,
                    "weapon_margin_v2_no_impact": V2.base_prices[4] * 0.9 * (1 - V2.transport_fee)
                    - (V2.base_prices[2] + V2.base_prices[1]) * 1.1 * (1 + V2.transport_fee)}

    # ---- 5. tax "Laffer curve" for the player ------------------------------------------------
    print("tax sweep ...")
    rows = []
    for tax in np.round(np.arange(0.05, 0.46, 0.05), 2):
        custom = dict(STRATEGIES)
        custom["Player"] = replace(STRATEGIES["Builder"], name="Player", tax=float(tax))
        import economy  # temporarily register the custom strategy
        economy.STRATEGIES["Player"] = custom["Player"]
        for s in range(12):
            seat = list(np.random.default_rng(s).permutation(["Player", "Zealot", "Merchant"]))
            wg = play(V2, seat, seed=s, record=True)
            h = pd.DataFrame(wg.history)
            pc = [c for c in wg.cities if c.strategy.name == "Player"][0]
            hp = h[h.strategy == "Player"]
            sc = [wg.score(c) for c in wg.cities]
            rows.append({"tax": tax, "score": wg.score(pc), "pop": pc.pop, "stability": hp.stability.mean(),
                         "win": int(sc.index(max(sc)) == wg.cities.index(pc))})
        del economy.STRATEGIES["Player"]
    lf = pd.DataFrame(rows).groupby("tax").mean()
    R["tax_sweep"] = lf.round(3).reset_index().to_dict(orient="records")
    best_tax = float(lf.score.idxmax())
    R["best_tax"] = best_tax
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(lf.index, lf.score, "o-", color=PALETTE[0], lw=2, label="final prosperity score")
    ax.set_xlabel("tax rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("prosperity score")
    ax2 = ax.twinx()
    ax2.plot(lf.index, lf.stability, "s--", color=PALETTE[1], lw=1.5, label="average stability")
    ax2.set_ylabel("average stability")
    ax2.grid(False)
    ax.axvline(best_tax, color=GREY, ls=":")
    ax.set_title(f"Tax sweep (v2): a clear sweet spot at {best_tax:.0%} — no 'always max' or 'always min' answer")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left")
    save(fig, FIG / "05_tax_sweep.png")

    # ---- 6. balance search heatmap -------------------------------------------------------------
    print("balance grid ...")
    tus, tps = [3.0, 4.5, 6.0], [1.5, 2.5, 3.5]
    heat = np.zeros((len(tus), len(tps)))
    detail = []
    for i, tu in enumerate(tus):
        for j, tp in enumerate(tps):
            dd, _, _ = tournament(replace(V2, temple_upkeep=tu, tax_stability_penalty=tp), seeds=4)
            wr = dd.groupby("strategy").win.mean()
            heat[i, j] = (wr - 1 / 3).abs().max()
            detail.append({"temple_upkeep": tu, "tax_penalty": tp, "max_dev": heat[i, j], **wr.round(3).to_dict()})
    R["balance_grid"] = detail
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(heat, cmap="RdYlGn_r", vmin=0.05, vmax=0.35)
    ax.set_xticks(range(len(tps)), [f"{p}" for p in tps])
    ax.set_yticks(range(len(tus)), [f"{u}" for u in tus])
    ax.set_xlabel("stability penalty per unit of tax above 15%")
    ax.set_ylabel("temple upkeep, gold/month")
    for (a, b), v in np.ndenumerate(heat):
        ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.grid(False)
    ax.set_title("Balance search: max |win rate − 1/3|")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, FIG / "06_balance_grid.png")

    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    print(json.dumps({k: R[k] for k in ["win_rates", "money_growth_10y", "exploit", "best_tax"]}, indent=1, default=float))
    print(fl.round(1))
    print(lf.round(3))


if __name__ == "__main__":
    main()
