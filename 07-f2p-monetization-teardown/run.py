"""
Project 07 — Monetisation teardown of Genshin Impact, quantified.

All prices / rates are public (see README sources). The wish-probability engine is reused from
project 05, with the Character Event Wish parameters.

Usage:  python 07-f2p-monetization-teardown/run.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "05-gacha-economy"))
from gacha import Banner, featured_pmf, summarize  # noqa: E402
from gamesim.plotting import GREY, INK, PALETTE, save  # noqa: E402

FIG = HERE / "reports" / "figures"
OUT = HERE / "reports"
OUT.mkdir(parents=True, exist_ok=True)
PRIMOS_PER_WISH = 160
PATCH_DAYS = 42

# ---- public offer catalogue (USD) -------------------------------------------------------------
TOPUPS = [  # price, base crystals, regular bonus
    (0.99, 60, 0), (4.99, 300, 30), (14.99, 980, 110), (29.99, 1980, 260), (49.99, 3280, 600), (99.99, 6480, 1600),
]
# ---- free income per ~42-day version (community estimate for v5.4; varies by patch) -------------
F2P_PRIMOS = 8420
F2P_INTERTWINED = 5          # Paimon's Bargains (Stardust shop); Acquaint Fates are standard-banner only
WELKIN_PER_PATCH = 90 * PATCH_DAYS            # daily primogems; the 300 crystals on purchase are extra
BP_PRIMOS, BP_FATES = 680, 4                  # Gnostic Hymn

# Character Event Wish: 0.6% base, soft pity ~74, hard 90, 50/50 with guarantee.
# Capturing Radiance (v5.0+) raises the consolidated featured share of a 50/50 to ~55%; here it is
# approximated as a flat 55/45 coin (the real mechanic is counter-based).
GENSHIN = Banner("Character Event Wish", base_rate=0.006, soft_pity=74, soft_step=0.06, hard_pity=90,
                 featured_share=0.5, guarantee=True)
GENSHIN_CR = replace(GENSHIN, name="with Capturing Radiance (approx.)", featured_share=0.55)


def offers() -> pd.DataFrame:
    rows = []
    for price, base, bonus in TOPUPS:
        rows.append({"offer": f"${price:.2f} top-up", "type": "top-up", "price": price, "primos": base + bonus})
        rows.append({"offer": f"${price:.2f} top-up (first-time ×2)", "type": "first-time", "price": price,
                     "primos": 2 * base})
    rows.append({"offer": "Welkin Moon ($4.99, 30 days)", "type": "subscription", "price": 4.99,
                 "primos": 300 + 90 * 30})
    rows.append({"offer": "Gnostic Hymn battle pass ($9.99)", "type": "battle pass", "price": 9.99,
                 "primos": BP_PRIMOS + BP_FATES * PRIMOS_PER_WISH})
    df = pd.DataFrame(rows)
    df["wishes"] = df.primos / PRIMOS_PER_WISH
    df["usd_per_wish"] = df.price / df.wishes
    return df.sort_values("usd_per_wish").reset_index(drop=True)


def n_copies_pmf(pmf: np.ndarray, k: int) -> np.ndarray:
    """After obtaining the featured unit the state resets (pity 0, no guarantee), so the number of
    wishes for k copies is the k-fold convolution of the single-copy distribution."""
    out = pmf.copy()
    for _ in range(k - 1):
        out = np.convolve(out, pmf)
    # convolve of pmfs indexed from 1: shift by one each time
    return np.concatenate([np.zeros(k - 1), out])[: len(out)]


def main() -> None:
    R: dict = {}
    off = offers()
    off.to_csv(OUT / "offers.csv", index=False)
    R["offers"] = off.round(3).to_dict(orient="records")
    print(off.round(3).to_string())

    # ---- 1. price per wish ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    col = {"subscription": PALETTE[2], "battle pass": PALETTE[0], "first-time": PALETTE[6], "top-up": PALETTE[1]}
    y = np.arange(len(off))[::-1]
    ax.barh(y, off.usd_per_wish, color=[col[t] for t in off.type])
    for yy, v in zip(y, off.usd_per_wish):
        ax.text(v + 0.03, yy, f"${v:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y, off.offer, fontsize=8.5)
    ax.grid(axis="x")
    ax.set_xlabel("USD per wish (160 primogems)")
    ax.set_title("Price ladder: the $4.99 subscription is ~8× cheaper per wish than top-ups")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=t) for t, c in col.items()], loc="upper right", fontsize=8.5)
    ax.set_xlim(0, off.usd_per_wish.max() * 1.12)
    save(fig, FIG / "01_price_per_wish.png")

    # ---- 2. spending tiers per version ----------------------------------------------------------
    f2p = F2P_PRIMOS / PRIMOS_PER_WISH + F2P_INTERTWINED
    welkin = WELKIN_PER_PATCH / PRIMOS_PER_WISH
    bp = BP_PRIMOS / PRIMOS_PER_WISH + BP_FATES
    tiers = pd.DataFrame([
        {"tier": "F2P", "wishes_per_version": f2p, "usd_per_version": 0.0},
        {"tier": "Welkin", "wishes_per_version": f2p + welkin, "usd_per_version": 4.99 * PATCH_DAYS / 30},
        {"tier": "Welkin + BP", "wishes_per_version": f2p + welkin + bp, "usd_per_version": 4.99 * PATCH_DAYS / 30 + 9.99},
    ])
    single = featured_pmf(GENSHIN_CR, 400)
    mean_featured = summarize(single)["mean"]
    tiers["featured_per_year_expected"] = tiers.wishes_per_version * (365 / PATCH_DAYS) / mean_featured
    tiers["usd_per_year"] = tiers.usd_per_version * 365 / PATCH_DAYS
    R["tiers"] = tiers.round(3).to_dict(orient="records")
    R["mean_wishes_per_featured"] = {"50/50": summarize(featured_pmf(GENSHIN, 400))["mean"],
                                     "with_capturing_radiance_approx": mean_featured}
    print(tiers.round(2))

    # P(get the featured unit) as a function of how many versions you save, per tier
    cdf = np.cumsum(single)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    versions = np.arange(1, 5)
    for i, r in tiers.iterrows():
        p = [cdf[min(int(r.wishes_per_version * v) - 1, len(cdf) - 1)] for v in versions]
        axes[0].plot(versions, p, "o-", color=PALETTE[i], lw=2, label=f"{r.tier} ({r.wishes_per_version:.0f} wishes/version)")
    axes[0].set_xticks(versions, [f"{v}" for v in versions])
    axes[0].set_xlabel("versions of saving before the banner")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("Chance to get a chosen limited character")
    axes[0].legend(fontsize=8.5, loc="lower right")

    # cost of copies for a pure spender at the best top-up rate
    best_rate = off[off.type == "top-up"].usd_per_wish.min()
    rows = []
    for k in range(1, 8):
        pk = n_copies_pmf(featured_pmf(GENSHIN_CR, 400), k)
        s = summarize(pk)
        rows.append({"copies": k, "label": f"C{k - 1}", "mean_wishes": s["mean"], "p90": s["p90"], "mean_usd": s["mean"] * best_rate, "p90_usd": s["p90"] * best_rate,
                     "p10_usd": np.searchsorted(np.cumsum(pk), 0.1) * best_rate + best_rate})
    cc = pd.DataFrame(rows)
    R["constellation_cost"] = cc.round(2).to_dict(orient="records")
    axes[1].bar(cc.label, cc.mean_usd, color=PALETTE[4], label="mean")
    axes[1].errorbar(cc.label, cc.mean_usd, yerr=[cc.mean_usd - cc.p10_usd, cc.p90_usd - cc.mean_usd], fmt="none",
                     ecolor=INK, capsize=3, lw=1, label="10th–90th percentile")
    for i, v in enumerate(cc.mean_usd):
        axes[1].text(i, v * 0.5, f"${v:,.0f}", ha="center", color="white", fontsize=8, fontweight="bold")
    axes[1].set_title(f"Cost of a limited character by constellation, top-up only (${best_rate:.2f}/wish)")
    axes[1].set_xlabel("constellation (C0 = first copy)")
    axes[1].legend(fontsize=8.5, loc="upper left")
    save(fig, FIG / "02_tiers_and_constellations.png")

    # ---- 3. how revenue is anchored: monthly spend for typical payer archetypes -----------------
    arche = pd.DataFrame([
        {"archetype": "Welkin only", "usd_per_month": 4.99},
        {"archetype": "Welkin + BP", "usd_per_month": 4.99 + 9.99 * 30 / PATCH_DAYS},
        {"archetype": "Dolphin: + 1 featured / version gap", "usd_per_month": None},
        {"archetype": "Whale: C6 of every 2nd banner", "usd_per_month": None},
    ])
    gap = max(0.0, mean_featured * 1 - tiers.loc[2, "wishes_per_version"])  # wishes to buy for 1 featured/version
    arche.loc[2, "usd_per_month"] = arche.loc[1, "usd_per_month"] + gap * best_rate * 30 / PATCH_DAYS
    # ~2 limited banners per version; "every 2nd banner" = one C6 (7 copies) per version
    whale_wishes = cc.loc[6, "mean_wishes"]
    arche.loc[3, "usd_per_month"] = arche.loc[1, "usd_per_month"] + max(
        0, whale_wishes - tiers.loc[2, "wishes_per_version"]) * best_rate * 30 / PATCH_DAYS
    R["archetypes"] = arche.round(2).to_dict(orient="records")
    print(arche.round(2))

    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    print(cc.round(1))


if __name__ == "__main__":
    main()
