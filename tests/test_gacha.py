import numpy as np

from gacha import Banner, featured_pmf, first_5star_pmf, simulate_featured, summarize

BASE = Banner("baseline")


def test_consolidated_rate_matches_published_1_6_percent():
    mean = summarize(first_5star_pmf(BASE))["mean"]
    assert abs(1 / mean - 0.016) < 0.0005


def test_pmfs_sum_to_one():
    for b in [BASE, Banner("flat", base_rate=0.016, soft_pity=None, hard_pity=None), Banner("spark", spark=150)]:
        assert abs(featured_pmf(b, 6000).sum() - 1) < 1e-9


def test_exact_matches_monte_carlo():
    mc = simulate_featured(BASE, 100_000, np.random.default_rng(0))
    ex = summarize(featured_pmf(BASE))
    assert abs(mc.mean() - ex["mean"]) < 0.6
    assert abs(np.quantile(mc, 0.9) - ex["p90"]) <= 2


def test_hard_pity_and_spark_bound_the_worst_case():
    assert simulate_featured(BASE, 50_000, np.random.default_rng(1)).max() <= 180
    assert simulate_featured(Banner("s", spark=150), 50_000, np.random.default_rng(2)).max() <= 150
