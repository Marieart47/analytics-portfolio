import numpy as np

from run import policy_value  # 03-uplift-offer-targeting/run.py


def test_snips_recovers_policy_value_under_non_uniform_logging():
    rng = np.random.default_rng(0)
    n = 200_000
    x = rng.random(n)
    e = 0.2 + 0.6 * x                       # logging propensity depends on x
    t = (rng.random(n) < e).astype(int)
    y0 = 1 + x + rng.normal(0, 0.5, n)
    y1 = y0 + (x - 0.4)                     # positive effect only for x > 0.4
    y = np.where(t == 1, y1, y0)
    pi = (x > 0.4).astype(int)
    true = np.mean(np.where(pi == 1, y1, y0))
    assert abs(policy_value(pi, t, y, e) - true) < 0.01
    assert abs(policy_value(pi, t, y, e, "ips") - true) < 0.02
