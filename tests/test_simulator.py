import numpy as np

from gamesim.simulator import SimConfig, simulate


def _small():
    return simulate(SimConfig(n_players=3000, install_days=20, horizon_days=40, seed=3))


def test_deterministic():
    a, b = _small(), _small()
    assert len(a.activity) == len(b.activity)
    assert np.isclose(a.purchases.price_usd.sum(), b.purchases.price_usd.sum())


def test_event_integrity():
    r = _small()
    inst = r.players.set_index("player_id").install_day
    assert (r.activity.day.to_numpy() >= inst.loc[r.activity.player_id].to_numpy()).all()
    active = set(zip(r.activity.player_id, r.activity.day))
    assert all((p, d) in active for p, d in zip(r.purchases.player_id, r.purchases.day))
    # every install is active on day 0 and starts the tutorial
    assert r.tutorial[r.tutorial.step_no == 1].player_id.nunique() == len(r.players)
