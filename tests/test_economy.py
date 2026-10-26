import numpy as np

from economy import V1, V2, play


def test_v2_controls_inflation_and_closes_exploit():
    w1 = play(V1, ["Merchant", "Builder", "Zealot"], seed=1, record=True)
    w2 = play(V2, ["Merchant", "Builder", "Zealot"], seed=1, record=True)
    assert w1.money[-1] / w1.money[0] > 50 * (w2.money[-1] / w2.money[0])
    m1 = [c for c in w1.cities if c.strategy.name == "Merchant"][0]
    m2 = [c for c in w2.cities if c.strategy.name == "Merchant"][0]
    assert m1.workshops > 50 and m2.workshops < 10


def test_scores_are_finite():
    w = play(V2, ["Warlord", "Taxman", "Zealot"], seed=4)
    assert all(np.isfinite(w.score(c)) for c in w.cities)
