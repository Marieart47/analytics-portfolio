import numpy as np

import abstats as ab


def test_delta_method_matches_simulation():
    """Delta-method SE of a ratio of user-level sums should match the empirical SE across replications."""
    rng = np.random.default_rng(0)
    ests = []
    for _ in range(400):
        den = rng.poisson(5, 2000) + 1
        num = rng.gamma(2, 1.5, 2000) * den
        ests.append(num.sum() / den.sum())
    den = rng.poisson(5, 2000) + 1
    num = rng.gamma(2, 1.5, 2000) * den
    _, var = ab.ratio_var(num, den)
    assert abs(np.sqrt(var) / np.std(ests) - 1) < 0.15


def test_cuped_reduces_variance_and_keeps_mean_difference():
    rng = np.random.default_rng(1)
    x = rng.normal(10, 3, 20000)
    y = x + rng.normal(0, 1, 20000)
    t = rng.random(20000) < 0.5
    y = y + 0.2 * t
    est, theta, vr = ab.cuped_test(y[~t], x[~t], y[t], x[t])
    assert vr > 0.8
    assert abs(theta - 1) < 0.05
    assert est.ci_low < 0.2 < est.ci_high


def test_srm():
    assert ab.srm_test(5000, 5000) > 0.9
    assert ab.srm_test(5000, 5400) < 0.001


def test_holm_known_values():
    adj = ab.holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_mde_and_sample_size_are_inverse():
    n = ab.sample_size(sd=1.0, effect=0.1)
    assert abs(ab.mde(1.0, n) - 0.1) < 0.002


def test_aa_false_positive_rate_is_nominal():
    rng = np.random.default_rng(2)
    rej = 0
    for _ in range(2000):
        a, b = rng.exponential(3, 3000), rng.exponential(3, 3000)
        rej += ab.mean_test(a, b).p_value < 0.05
    assert 0.035 < rej / 2000 < 0.065
