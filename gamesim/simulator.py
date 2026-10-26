"""
Synthetic mobile F2P game telemetry generator.

Why synthetic?  Real game telemetry is proprietary, and public datasets (Cookie Cats etc.)
are tiny and over-used.  A simulator gives something more valuable for a portfolio:
**known ground truth**.  Every player has latent traits (engagement, spending propensity),
and experiments can inject effects of a known size, so analysis methods can be *validated*
— did my A/B pipeline recover the true +2% effect?  did my uplift model find the players
that truly respond?

The model (one row = one player, simulated day by day):

* acquisition: install date, channel, platform, country tier (each shifts latent traits);
* latent traits: engagement ``z_eng`` and spending propensity ``z_pay`` (correlated);
* day 0: a 4-step tutorial funnel; failing the tutorial makes churn very likely;
* survival: daily churn hazard that decays with age (``h0 * (1 + age) ** -decay``),
  heterogeneous across players, which produces realistic flattening retention curves;
* activity: when alive, a player is active with prob. ``p_active`` (weekend uplift);
* sessions / minutes / level progression;
* purchases: logistic purchase probability; repeat buyers are much likelier to buy again;
  SKU (price point) chosen with a softmax that shifts towards expensive packs for "whales".

Experiments plug in through :class:`Effects` — callables returning per-player logit shifts
for activity, purchase and churn on a given calendar day.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

START_DATE = pd.Timestamp("2026-03-02")  # a Monday

CHANNELS = {
    #               share   eng    pay    CPI ($)
    "organic": (0.35, 0.00, 0.00, 0.00),
    "paid_social": (0.30, -0.30, -0.15, 4.20),
    "paid_search": (0.20, 0.15, 0.20, 4.00),
    "influencer": (0.15, -0.10, 0.35, 6.50),
}
PLATFORMS = {"android": (0.60, 0.00), "ios": (0.40, 0.45)}
COUNTRY_TIERS = {"tier1": (0.30, 0.55), "tier2": (0.40, 0.00), "tier3": (0.30, -0.70)}

SKU_PRICES = np.array([0.99, 4.99, 9.99, 19.99, 49.99, 99.99])
SKU_NAMES = np.array(["gems_xs", "gems_s", "starter_pack", "gems_m", "gems_l", "gems_xl"])
_SKU_BASE = np.array([1.8, 1.4, 1.1, -0.1, -1.7, -2.9])
_SKU_SLOPE = np.array([-0.9, -0.3, 0.0, 0.35, 0.9, 1.3])

TUTORIAL_STEPS = ["tutorial_start", "tutorial_step_2", "first_battle", "tutorial_complete"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SimConfig:
    n_players: int = 60_000
    install_days: int = 60          # installs spread over the first N calendar days
    horizon_days: int = 160         # simulate until this calendar day (exclusive)
    install_growth: float = 0.6     # daily installs grow linearly by this fraction over the window
    churn_base: float = -0.95       # logit of day-0 churn hazard for an average player
    churn_eng: float = -0.85        # engaged players churn less
    churn_decay: float = 0.55       # hazard decays as (1+age)^-decay
    tutorial_fail_churn: float = 0.85
    active_base: float = 0.65
    active_eng: float = 0.70
    weekend_boost: float = 0.30
    buy_base: float = -7.0
    buy_pay: float = 1.35
    buy_level: float = 0.30
    repeat_buyer_boost: float = 1.2
    rho_eng_pay: float = 0.35
    seed: int = 42


@dataclass
class Effects:
    """Experiment hooks. Each callable gets (day, ctx) and returns an array of logit shifts
    (or multipliers for sessions) for the players in ``ctx['idx']``. ``None`` = no effect."""

    active_shift: Callable | None = None
    buy_shift: Callable | None = None
    churn_shift: Callable | None = None
    sessions_mult: Callable | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class SimResult:
    players: pd.DataFrame      # observable acquisition attributes
    truth: pd.DataFrame        # latent traits (NOT available to analysts in real life)
    activity: pd.DataFrame     # one row per player-active-day
    purchases: pd.DataFrame    # one row per transaction
    tutorial: pd.DataFrame     # tutorial funnel events (day 0)
    config: SimConfig

    def save(self, folder) -> None:
        from pathlib import Path

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name in ["players", "truth", "activity", "purchases", "tutorial"]:
            getattr(self, name).to_parquet(folder / f"{name}.parquet", index=False)


def _draw_categorical(rng, table: dict, n: int):
    keys = list(table)
    probs = np.array([table[k][0] for k in keys])
    return np.array(keys)[rng.choice(len(keys), size=n, p=probs / probs.sum())]


def make_players(cfg: SimConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = cfg.n_players
    # install day with linear growth in daily installs
    w = 1 + cfg.install_growth * np.arange(cfg.install_days) / max(cfg.install_days - 1, 1)
    install_day = rng.choice(cfg.install_days, size=n, p=w / w.sum())
    channel = _draw_categorical(rng, CHANNELS, n)
    platform = _draw_categorical(rng, PLATFORMS, n)
    tier = _draw_categorical(rng, COUNTRY_TIERS, n)

    eng_shift = np.array([CHANNELS[c][1] for c in channel])
    pay_shift = (
        np.array([CHANNELS[c][2] for c in channel])
        + np.array([PLATFORMS[p][1] for p in platform])
        + np.array([COUNTRY_TIERS[t][1] for t in tier])
    )
    cov = [[1, cfg.rho_eng_pay], [cfg.rho_eng_pay, 1]]
    z = rng.multivariate_normal([0, 0], cov, size=n)
    z_eng = z[:, 0] + eng_shift
    z_pay = z[:, 1] + pay_shift

    order = np.argsort(install_day, kind="stable")
    player_id = np.empty(n, dtype=np.int64)
    player_id[order] = np.arange(1, n + 1)

    players = pd.DataFrame(
        {
            "player_id": player_id,
            "install_day": install_day,
            "install_date": START_DATE + pd.to_timedelta(install_day, unit="D"),
            "channel": channel,
            "platform": platform,
            "country_tier": tier,
            "cpi": np.array([CHANNELS[c][3] for c in channel]),
        }
    )
    truth = pd.DataFrame({"player_id": player_id, "z_eng": z_eng, "z_pay": z_pay})
    return players, truth


def simulate(cfg: SimConfig | None = None, effects: Effects | None = None,
             players: pd.DataFrame | None = None, truth: pd.DataFrame | None = None) -> SimResult:
    """Run the simulation. Pass ``players``/``truth`` to re-simulate the *same* population
    under different effects (useful for counterfactuals)."""
    cfg = cfg or SimConfig()
    effects = effects or Effects()
    rng = np.random.default_rng(cfg.seed)
    if players is None:
        players, truth = make_players(cfg, rng)
    else:
        rng = np.random.default_rng(cfg.seed + 1)

    n = len(players)
    pid = players["player_id"].to_numpy()
    install_day = players["install_day"].to_numpy()
    z_eng = truth["z_eng"].to_numpy()
    z_pay = truth["z_pay"].to_numpy()

    alive = np.ones(n, dtype=bool)
    cum_sessions = np.zeros(n)
    n_buys = np.zeros(n)
    tut_ok = np.ones(n, dtype=bool)

    h0_logit = cfg.churn_base + cfg.churn_eng * z_eng
    act_logit = cfg.active_base + cfg.active_eng * z_eng

    act_rows, buy_rows, tut_rows = [], [], []
    ctx_static = {"players": players, "truth": truth}

    for d in range(cfg.horizon_days):
        idx = np.flatnonzero((install_day <= d) & alive)
        if idx.size == 0:
            continue
        age = d - install_day[idx]
        ctx = {**ctx_static, "idx": idx, "age": age, "day": d,
               "cum_sessions": cum_sessions[idx], "n_buys": n_buys[idx]}
        weekday = (START_DATE + pd.Timedelta(days=d)).dayofweek

        # --- tutorial on install day -------------------------------------------------
        new = age == 0
        if new.any():
            nidx = idx[new]
            p_step = _sigmoid(2.3 + 0.55 * z_eng[nidx])
            reached = np.ones(nidx.size, dtype=bool)
            for s, step in enumerate(TUTORIAL_STEPS):
                if s > 0:
                    reached &= rng.random(nidx.size) < p_step
                tut_rows.append(pd.DataFrame({"player_id": pid[nidx[reached]], "day": d, "step": step,
                                              "step_no": s + 1}))
            tut_ok[nidx] = reached

        # --- activity -----------------------------------------------------------------
        logit = act_logit[idx] + (cfg.weekend_boost if weekday >= 5 else 0.0)
        if effects.active_shift is not None:
            logit = logit + effects.active_shift(d, ctx)
        active = (rng.random(idx.size) < _sigmoid(logit)) | new
        aidx = idx[active]
        a_age = age[active]
        actx = {**ctx, "idx": aidx, "age": a_age, "cum_sessions": cum_sessions[aidx], "n_buys": n_buys[aidx]}

        lam = np.exp(0.35 + 0.30 * z_eng[aidx])
        if effects.sessions_mult is not None:
            lam = lam * effects.sessions_mult(d, actx)
        sessions = 1 + rng.poisson(lam)
        minutes = np.round(rng.gamma(2.0, 5.5 * np.exp(0.12 * z_eng[aidx]), size=aidx.size) * sessions, 1)
        cum_sessions[aidx] += sessions
        level = 1 + np.floor(2.2 * np.sqrt(cum_sessions[aidx])).astype(int)
        act_rows.append(pd.DataFrame({"player_id": pid[aidx], "day": d, "sessions": sessions,
                                      "minutes": minutes, "level": level}))

        # --- purchases ----------------------------------------------------------------
        b_logit = (cfg.buy_base + cfg.buy_pay * z_pay[aidx] + cfg.buy_level * np.log1p(level)
                   + cfg.repeat_buyer_boost * (n_buys[aidx] > 0))
        if effects.buy_shift is not None:
            b_logit = b_logit + effects.buy_shift(d, actx)
        buys = rng.random(aidx.size) < _sigmoid(b_logit)
        if buys.any():
            bidx = aidx[buys]
            util = _SKU_BASE[None, :] + _SKU_SLOPE[None, :] * z_pay[bidx, None]
            prob = np.exp(util - util.max(axis=1, keepdims=True))
            prob /= prob.sum(axis=1, keepdims=True)
            sku = (prob.cumsum(axis=1) > rng.random(bidx.size)[:, None]).argmax(axis=1)
            n_buys[bidx] += 1
            buy_rows.append(pd.DataFrame({"player_id": pid[bidx], "day": d,
                                          "sku": SKU_NAMES[sku], "price_usd": SKU_PRICES[sku]}))

        # --- churn at end of day -----------------------------------------------------
        h_logit = h0_logit[idx]
        if effects.churn_shift is not None:
            h_logit = h_logit + effects.churn_shift(d, ctx)
        hazard = _sigmoid(h_logit) * (1 + age) ** (-cfg.churn_decay)
        hazard = np.where(new & ~tut_ok[idx], np.maximum(hazard, cfg.tutorial_fail_churn), hazard)
        alive[idx[rng.random(idx.size) < hazard]] = False

    def _cat(rows, cols):
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)

    activity = _cat(act_rows, ["player_id", "day", "sessions", "minutes", "level"])
    purchases = _cat(buy_rows, ["player_id", "day", "sku", "price_usd"])
    tutorial = _cat(tut_rows, ["player_id", "day", "step", "step_no"])
    for df in (activity, purchases, tutorial):
        df["event_date"] = START_DATE + pd.to_timedelta(df["day"], unit="D")
    purchases.insert(0, "transaction_id", np.arange(1, len(purchases) + 1))

    return SimResult(players=players, truth=truth, activity=activity, purchases=purchases,
                     tutorial=tutorial, config=cfg)


if __name__ == "__main__":
    import time

    t = time.time()
    res = simulate(SimConfig(n_players=20_000))
    print(f"{time.time() - t:.1f}s", len(res.activity), len(res.purchases))
