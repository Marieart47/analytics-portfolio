"""
Builds the offer-experiment dataset.

1. gamesim generates 120 days of telemetry; the snapshot is taken on day 120 for every player
   active in the previous 14 days. Features are computed ONLY from observable telemetry.
2. A "50% off Gem Chest ($19.99 -> $9.99)" offer is shown for 7 days. Assignment follows the
   *legacy targeting rule* (more active players were more likely to get it), so the treatment
   propensity e(x) is between 0.2 and 0.8 and NOT uniform — and it is logged.
3. Outcome = gross revenue in the 14 days after the snapshot. The structural outcome model gives the
   true individual effect tau(x) = E[Y(1) - Y(0) | latent traits], which a real analyst never sees.

The economics built into the outcome model:
* price-sensitive "almost payers" (moderate spending propensity, active) are converted by the offer;
* existing big spenders who would have bought a regular pack anyway switch to the cheaper
  discounted chest -> **cannibalisation, negative uplift**.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamesim.simulator import SimConfig, simulate  # noqa: E402

SNAP = 120
OFFER_PRICE = 9.99


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def build(seed: int = 7, n_players: int = 380_000) -> pd.DataFrame:
    cfg = SimConfig(n_players=n_players, install_days=SNAP, horizon_days=SNAP, seed=seed)
    res = simulate(cfg)
    a, p = res.activity, res.purchases
    recent = a[a.day >= SNAP - 14]
    ids = recent.player_id.unique()

    f = res.players.set_index("player_id").loc[ids, ["install_day", "channel", "platform", "country_tier"]]
    f["days_since_install"] = SNAP - f.pop("install_day")
    g14 = recent.groupby("player_id").agg(active_days_14=("day", "nunique"), sessions_14=("sessions", "sum"),
                                          minutes_14=("minutes", "sum"))
    last = a[a.player_id.isin(ids)].groupby("player_id").agg(level=("level", "max"), last_seen=("day", "max"))
    pl = p[p.player_id.isin(ids)].groupby("player_id").agg(
        lifetime_revenue=("price_usd", "sum"), n_purchases=("price_usd", "size"),
        last_purchase_day=("day", "max"), max_purchase=("price_usd", "max"))
    p14 = p[(p.day >= SNAP - 14) & p.player_id.isin(ids)].groupby("player_id").price_usd.sum().rename("revenue_14")
    f = f.join(g14).join(last).join(pl).join(p14)
    f["days_since_last_seen"] = SNAP - 1 - f.pop("last_seen")
    f["days_since_purchase"] = (SNAP - f.pop("last_purchase_day")).fillna(999)
    f[["lifetime_revenue", "n_purchases", "max_purchase", "revenue_14"]] = f[
        ["lifetime_revenue", "n_purchases", "max_purchase", "revenue_14"]].fillna(0)
    f["is_payer"] = (f.n_purchases > 0).astype(int)

    # ---------- latent traits (hidden) -----------------------------------------------------------
    tr = res.truth.set_index("player_id").loc[ids]
    z_pay, z_eng = tr.z_pay.to_numpy(), tr.z_eng.to_numpy()
    rec = f.active_days_14.to_numpy() / 14
    payer = f.is_payer.to_numpy()

    rng = np.random.default_rng(seed + 100)
    n = len(f)

    # ---------- logging policy: legacy rule "show the offer to active players" -------------------
    act_z = (f.active_days_14 - f.active_days_14.mean()) / f.active_days_14.std()
    e = np.clip(_sigmoid(-0.2 + 0.9 * act_z.to_numpy() + 0.4 * payer), 0.2, 0.8)
    T = (rng.random(n) < e).astype(int)

    # ---------- structural outcome model ---------------------------------------------------------
    p_reg = _sigmoid(-4.6 + 1.0 * z_pay + 2.2 * rec + 2.2 * payer)
    mu_s = 2.3 + 0.6 * z_pay
    e_spend = np.exp(mu_s + 0.5**2 / 2)
    p_new = _sigmoid(-2.7 + 0.9 * z_pay + 1.8 * rec - 0.35 * z_pay**2 + 0.25 * z_eng)
    q_cannibal = _sigmoid(-1.0 + 0.8 * z_pay)

    tau = p_reg * q_cannibal * (OFFER_PRICE - e_spend) + (1 - p_reg) * p_new * OFFER_PRICE
    ey0 = p_reg * e_spend

    # realised potential outcomes with shared randomness (so Y1 - Y0 is meaningful per player)
    u_reg, u_new, u_can = rng.random(n), rng.random(n), rng.random(n)
    spend = np.round(np.exp(mu_s + 0.5 * rng.standard_normal(n)), 2)
    buys_reg = u_reg < p_reg
    y0 = np.where(buys_reg, spend, 0.0)
    y1 = np.where(buys_reg, np.where(u_can < q_cannibal, OFFER_PRICE, spend),
                  np.where(u_new < p_new, OFFER_PRICE, 0.0))
    y = np.where(T == 1, y1, y0)

    f = f.reset_index()
    f["treatment"] = T
    f["propensity"] = e
    f["revenue_next_14"] = y
    f["converted"] = (y > 0).astype(int)
    # hidden columns (prefixed) — used ONLY for evaluation against the truth
    f["_tau"] = tau
    f["_ey0"] = ey0
    f["_y0"], f["_y1"] = y0, y1
    f["_z_pay"], f["_z_eng"] = z_pay, z_eng
    return f


FEATURES = ["days_since_install", "active_days_14", "sessions_14", "minutes_14", "level",
            "days_since_last_seen", "lifetime_revenue", "n_purchases", "max_purchase", "revenue_14",
            "days_since_purchase", "is_payer", "channel", "platform", "country_tier"]
CATEGORICAL = ["channel", "platform", "country_tier"]


if __name__ == "__main__":
    df = build()
    print(len(df), df.treatment.mean(), df.revenue_next_14.mean(), df._tau.mean(), (df._tau < 0).mean())
    print(df.groupby("treatment").revenue_next_14.mean())
