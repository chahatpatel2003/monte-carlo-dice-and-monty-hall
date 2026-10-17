"""Checks the reconstructed model against the result tables recovered from the original report.

The original code was not recovered, so the reconstruction cannot be compared line by line. Instead these
tests check that the original results are statistically consistent with the reconstructed model's exact
probabilities, and that the inferred definitions (win rates exclude ruined games; the ruined rate comes
from the stay run) reproduce whole-number game counts.
"""
import math
from pathlib import Path

import pandas as pd
import pytest

import monty_hall as mh

TABLES = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "recovered_report_tables.csv")
CHI2_DF25_P999 = 52.62  # 99.9th percentile of chi-square with 25 degrees of freedom


def z_scores(figure, host_error):
    rows = TABLES[TABLES.Figure == figure]
    trials = rows.Trials.iloc[0]
    out = []
    for _, r in rows.iterrows():
        e = mh.exact_table(host_error, door_counts=[int(r.Doors)]).iloc[0]
        not_ruined = trials * (1 - e.Ruined_Games_Rate)
        for col, n in [("Standard_Stay", trials), ("Standard_Switch", trials), ("Unpredictable_Stay", not_ruined),
                       ("Unpredictable_Switch", not_ruined), ("Ruined_Games_Rate", trials)]:
            out.append((r[col] - e[col]) / math.sqrt(e[col] * (1 - e[col]) / n))
    return out


@pytest.mark.parametrize("figure,host_error", [(3, 0.1), (5, 1.0)])
def test_recovered_100000_trial_tables_match_the_model(figure, host_error):
    z = z_scores(figure, host_error)
    assert len(z) == 25
    assert sum(v * v for v in z) < CHI2_DF25_P999
    assert max(abs(v) for v in z) < 4


def test_recovered_ruin_rates_match_r_over_n():
    for _, r in TABLES[TABLES.Figure == 4].iterrows():
        p = mh.exact_ruined_rate(r.Doors, 1.0)
        assert abs(r.Ruined_Games_Rate - p) < 4 * math.sqrt(p * (1 - p) / r.Trials)


def test_figure_3_host_error_is_about_one_tenth():
    rows = TABLES[TABLES.Figure == 3]
    estimate = (rows.Ruined_Games_Rate * rows.Doors * rows.Trials).sum() / (rows.Trials.sum())
    assert estimate == pytest.approx(0.1, abs=0.005)


def test_unpredictable_win_rates_exclude_ruined_games():
    rows = TABLES.dropna(subset=["Unpredictable_Stay"])
    for _, r in rows.iterrows():
        played = r.Trials * (1 - r.Ruined_Games_Rate)
        wins = r.Unpredictable_Stay * played
        assert played == pytest.approx(round(played), abs=1e-6)
        assert wins == pytest.approx(round(wins), abs=1e-3), (r.Figure, r.Doors)
