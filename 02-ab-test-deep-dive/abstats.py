"""Small, dependency-light A/B testing toolkit used by the analysis (and unit-tested in tests/)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Estimate:
    control: float
    treatment: float
    abs_diff: float
    rel_diff: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float

    def as_dict(self) -> dict:
        return {k: float(v) for k, v in self.__dict__.items()}


def _finish(mc: float, mt: float, diff: float, se: float, alpha: float = 0.05) -> Estimate:
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * (1 - stats.norm.cdf(abs(diff) / se)) if se > 0 else np.nan
    return Estimate(mc, mt, diff, diff / mc if mc else np.nan, se, diff - z * se, diff + z * se, p)


def mean_test(c: np.ndarray, t: np.ndarray, alpha: float = 0.05) -> Estimate:
    """Welch-type z-test for the difference in means (large samples)."""
    c, t = np.asarray(c, float), np.asarray(t, float)
    se = np.sqrt(c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t))
    return _finish(c.mean(), t.mean(), t.mean() - c.mean(), se, alpha)


def ratio_var(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Ratio sum(num)/sum(den) over *users* and its delta-method variance.

    Use for metrics whose analysis unit (session, active day, payer) differs from the
    randomisation unit (user): ARPPU, sessions per active day, CTR...
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    mx, my = num.mean(), den.mean()
    vx, vy = num.var(ddof=1), den.var(ddof=1)
    cxy = np.cov(num, den, ddof=1)[0, 1]
    r = mx / my
    var = (vx / my**2 - 2 * mx * cxy / my**3 + mx**2 * vy / my**4) / n
    return r, var


def ratio_test(num_c, den_c, num_t, den_t, alpha: float = 0.05) -> Estimate:
    rc, vc = ratio_var(num_c, den_c)
    rt, vt = ratio_var(num_t, den_t)
    return _finish(rc, rt, rt - rc, np.sqrt(vc + vt), alpha)


def cuped(y: np.ndarray, x: np.ndarray, theta: float | None = None) -> tuple[np.ndarray, float]:
    """CUPED adjustment Y - theta * (X - mean(X)); theta estimated on pooled data if not given."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    if theta is None:
        vx = x.var(ddof=1)
        theta = np.cov(y, x, ddof=1)[0, 1] / vx if vx > 0 else 0.0
    return y - theta * (x - x.mean()), theta


def cuped_test(yc, xc, yt, xt, alpha: float = 0.05) -> tuple[Estimate, float, float]:
    """Returns (estimate, theta, variance reduction share)."""
    y = np.concatenate([yc, yt])
    x = np.concatenate([xc, xt])
    adj, theta = cuped(y, x)
    ac, at = adj[: len(yc)], adj[len(yc):]
    est = mean_test(ac, at, alpha)
    est.control, est.treatment = float(np.mean(yc)), float(np.mean(yt))
    est.rel_diff = est.abs_diff / est.control
    var_red = 1 - adj.var(ddof=1) / y.var(ddof=1)
    return est, float(theta), float(var_red)


def srm_test(n_c: int, n_t: int, expected_share_t: float = 0.5) -> float:
    """Chi-square sample-ratio-mismatch test. Returns p-value."""
    total = n_c + n_t
    exp = np.array([total * (1 - expected_share_t), total * expected_share_t])
    return float(stats.chisquare([n_c, n_t], exp).pvalue)


def bootstrap_diff(c, t, stat=np.mean, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    c, t = np.asarray(c), np.asarray(t)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = stat(rng.choice(t, len(t))) - stat(rng.choice(c, len(c)))
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def mde(sd: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimal detectable absolute effect for a two-sample mean comparison, equal groups."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd * np.sqrt(2 / n_per_group))


def sample_size(sd: float, effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sd / effect) ** 2))


def holm(pvals) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
