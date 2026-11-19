import numpy as np

from ambilight_music.audio import BeatClock

PERIOD = 0.5  # 120 BPM


def _run(clock, kicks, until, step=1 / 120):
    """Feeds kick onsets at the given times and polls the clock like the render loop does.
    Returns the times at which a pulse fired."""
    fires, kicks, t = [], list(kicks), 0.0
    while t < until:
        while kicks and kicks[0] <= t:
            if clock.onset(kicks.pop(0)):
                fires.append(clock.fired_at)
        if clock.update(t):
            fires.append(clock.fired_at)
        t += step
    return fires


def test_reacts_before_lock_then_fires_early_by_the_lead():
    clock = BeatClock(lead_s=0.1)
    kicks = [i * PERIOD for i in range(16)]
    fires = _run(clock, kicks, until=kicks[-1] + 0.3)

    assert clock.locked
    assert abs(clock.period - PERIOD) < 0.01
    # One pulse per kick, never doubled by prediction + confirmation.
    assert len(fires) == len(kicks)
    # The first kicks are plain reactions, later ones come ~100 ms early.
    assert fires[0] == kicks[0]
    late = np.array(fires[8:]) - np.array(kicks[8:])
    assert np.all((late > -0.12) & (late < -0.08))


def test_irregular_kicks_never_lock_and_still_pulse():
    clock = BeatClock(lead_s=0.1)
    rng = np.random.default_rng(3)
    kicks = np.cumsum(rng.uniform(0.3, 0.9, 14)).tolist()
    fires = _run(clock, kicks, until=kicks[-1] + 0.5)
    assert not clock.locked
    assert fires == kicks


def test_break_releases_the_lock():
    clock = BeatClock(lead_s=0.1)
    kicks = [i * PERIOD for i in range(10)]
    fires = _run(clock, kicks, until=10.0)  # 5 s of silence after the last kick
    assert not clock.locked
    # It may freewheel through a couple of expected beats, but not through the whole break.
    assert len([f for f in fires if f > kicks[-1] + 0.01]) <= 2
