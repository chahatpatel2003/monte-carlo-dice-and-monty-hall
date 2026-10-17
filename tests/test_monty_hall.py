import math
import random
from collections import Counter

import pytest

import monty_hall as mh


def test_pick_excluding_is_uniform_over_allowed_doors():
    rng = random.Random(0)
    counts = Counter(mh._pick_excluding(6, {1, 4}, rng) for _ in range(40_000))
    assert set(counts) == {0, 2, 3, 5}
    assert all(abs(c / 40_000 - 0.25) < 0.01 for c in counts.values())


def test_classic_three_door_values():
    assert mh.exact_standard(3, False) == pytest.approx(1 / 3)
    assert mh.exact_standard(3, True) == pytest.approx(2 / 3)


def test_unpredictable_game_reduces_to_standard_when_host_never_errs():
    for n in mh.DOOR_COUNTS:
        for switch in (False, True):
            assert mh.exact_unpredictable(n, switch, 0.0) == pytest.approx(mh.exact_standard(n, switch))


def test_unpredictable_game_with_certain_error_is_a_coin_flip_for_three_doors():
    assert mh.exact_unpredictable(3, False, 1.0) == pytest.approx(0.5)
    assert mh.exact_unpredictable(3, True, 1.0) == pytest.approx(0.5)


def test_ruined_rate_is_r_over_n():
    assert mh.exact_ruined_rate(3, 1.0) == pytest.approx(1 / 3)
    assert mh.exact_ruined_rate(20, 0.1) == pytest.approx(0.005)


def test_switching_wins_more_for_every_door_count():
    for n in mh.DOOR_COUNTS:
        assert mh.exact_standard(n, True) > mh.exact_standard(n, False)


@pytest.mark.parametrize("host_error", [0.1, 1.0])
def test_simulation_agrees_with_exact_probabilities(host_error):
    trials = 20_000
    simulated = mh.results_table(trials, host_error, seed=2025)
    exact = mh.exact_table(host_error)
    assert list(simulated.columns) == mh.COLUMNS
    for (_, s), (_, e) in zip(simulated.iterrows(), exact.iterrows()):
        not_ruined = trials * (1 - mh.exact_ruined_rate(e.Doors, host_error))
        for col, n in [("Standard_Stay", trials), ("Standard_Switch", trials), ("Unpredictable_Stay", not_ruined),
                       ("Unpredictable_Switch", not_ruined), ("Ruined_Games_Rate", trials)]:
            se = math.sqrt(e[col] * (1 - e[col]) / n)
            assert abs(s[col] - e[col]) < 4.5 * se, (e.Doors, col)


def test_results_are_reproducible_with_a_seed():
    a = mh.results_table(500, 0.1, seed=9)
    b = mh.results_table(500, 0.1, seed=9)
    assert a.equals(b)


def test_convergence_has_one_point_per_step():
    curve = mh.convergence(max_trials=10_000, points=100, seed=1)
    assert len(curve) == 100
    assert curve["Trials"].iloc[0] == 100 and curve["Trials"].iloc[-1] == 10_000
