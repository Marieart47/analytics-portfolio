"""
Gacha banner engine: exact Markov-chain distributions + vectorised Monte Carlo.

A banner design is defined by
* ``base_rate``     — probability of a top-rarity (5★) drop per pull before soft pity;
* ``soft_pity``     — pull number from which the rate ramps up by ``soft_step`` per pull;
* ``hard_pity``     — pull number at which a 5★ is guaranteed (None = no hard pity);
* ``featured_share``— probability that a 5★ is the featured unit (the "50/50");
* ``guarantee``     — after losing the 50/50, the next 5★ is the featured unit;
* ``spark``         — after this many pulls on the banner without the featured unit, the player can
                      exchange for it (None = no spark).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Banner:
    name: str
    base_rate: float = 0.006
    soft_pity: int | None = 74
    soft_step: float = 0.06
    hard_pity: int | None = 90
    featured_share: float = 0.5
    guarantee: bool = True
    spark: int | None = None

    def rate(self, pull_in_pity: np.ndarray | int) -> np.ndarray:
        """5★ probability on the n-th pull since the last 5★ (n starts at 1)."""
        n = np.asarray(pull_in_pity, dtype=float)
        r = np.full(n.shape, self.base_rate)
        if self.soft_pity is not None:
            r = r + np.clip(n - (self.soft_pity - 1), 0, None) * self.soft_step
        if self.hard_pity is not None:
            r = np.where(n >= self.hard_pity, 1.0, r)
        return np.clip(r, 0, 1)


# ---------------------------------------------------------------------------------------------
# Exact distributions
# ---------------------------------------------------------------------------------------------
def first_5star_pmf(b: Banner, max_pulls: int = 3000) -> np.ndarray:
    """P(first 5★ arrives on pull n), n = 1..max_pulls (index 0 = pull 1)."""
    n = np.arange(1, max_pulls + 1)
    r = b.rate(n)
    survive = np.concatenate([[1.0], np.cumprod(1 - r)[:-1]])
    return survive * r


def featured_pmf(b: Banner, max_pulls: int = 3000) -> np.ndarray:
    """Exact P(featured unit obtained on pull t), t = 1..max_pulls, via forward DP over
    states (pulls since last 5★, guaranteed flag)."""
    H = b.hard_pity if b.hard_pity is not None else (b.soft_pity + int(np.ceil(1 / b.soft_step)) + 1
                                                       if b.soft_pity is not None else 1)
    # state[c, g]: probability mass with c pulls since last 5★ (0..H-1), guarantee flag g
    state = np.zeros((H, 2))
    state[0, 0] = 1.0
    rates = b.rate(np.arange(1, H + 1))          # rate for the (c+1)-th pull
    out = np.zeros(max_pulls)
    f = b.featured_share
    for t in range(max_pulls):
        p = rates[:, None]                        # (H, 1)
        hit = state * p
        miss = state * (1 - p)
        got = hit[:, 1].sum() + f * hit[:, 0].sum() if b.guarantee else f * hit.sum()
        lost = (1 - f) * hit[:, 0].sum() if b.guarantee else (1 - f) * hit.sum()
        new = np.zeros_like(state)
        if H > 1:
            new[1:, :] = miss[:-1, :]
            new[H - 1, :] += miss[H - 1, :]       # cap (only reachable without hard pity)
        else:
            new[0, :] += miss[0, :]
        if b.guarantee:
            new[0, 1] += lost
        else:
            new[0, 0] += lost
        out[t] = got
        state = new
        if b.spark is not None and t + 1 == b.spark:
            out[t] += state.sum()
            state[:] = 0
        if state.sum() < 1e-12:
            break
    return out


def summarize(pmf: np.ndarray) -> dict:
    n = np.arange(1, len(pmf) + 1)
    cdf = np.cumsum(pmf)
    q = lambda x: int(n[np.searchsorted(cdf, x)])  # noqa: E731
    mean = float((n * pmf).sum())
    return {"mean": mean, "sd": float(np.sqrt(((n - mean) ** 2 * pmf).sum())), "median": q(0.5),
            "p90": q(0.9), "p99": q(0.99), "worst_case": int(n[np.nonzero(pmf > 1e-12)[0][-1]]),
            "p_over_150": float(1 - cdf[149]) if len(cdf) > 149 else 0.0, "mass": float(pmf.sum())}


# ---------------------------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------------------------
def simulate_featured(b: Banner, n_players: int, rng: np.random.Generator, max_pulls: int = 5000) -> np.ndarray:
    """Pulls needed to obtain the featured unit for each of n_players (vectorised)."""
    pulls = np.zeros(n_players, dtype=np.int64)
    pity = np.zeros(n_players, dtype=np.int64)
    guaranteed = np.zeros(n_players, dtype=bool)
    result = np.full(n_players, -1, dtype=np.int64)
    active = np.arange(n_players)
    for _ in range(max_pulls):
        if active.size == 0:
            break
        pulls[active] += 1
        pity[active] += 1
        hit = rng.random(active.size) < b.rate(pity[active])
        win = hit & (guaranteed[active] | (rng.random(active.size) < b.featured_share))
        lose = hit & ~win
        pity[active[hit]] = 0
        if b.guarantee:
            guaranteed[active[lose]] = True
        done = win
        if b.spark is not None:
            done = done | (pulls[active] >= b.spark)
        result[active[done]] = pulls[active[done]]
        active = active[~done]
    return result


def pulls_to_spend(pulls: np.ndarray, free_pulls: np.ndarray, usd_per_pull: float) -> np.ndarray:
    return np.maximum(pulls - free_pulls, 0) * usd_per_pull
