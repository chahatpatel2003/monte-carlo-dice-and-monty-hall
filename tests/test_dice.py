import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dice_simulation as ds

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_COLUMNS = ["Sides", "Dice", "Trials", "Simulated Mean", "Expected Value", "Deviation"]


@pytest.fixture(scope="module")
def run():
    return ds.run_simulation(seed=154)


def test_expected_value_and_variance():
    assert ds.expected_value(6, 1) == 3.5
    assert ds.expected_value(20, 5) == 52.5
    assert ds.variance_of_sum(6, 1) == pytest.approx(35 / 12)
    assert ds.standard_error(6, 2, 100) == pytest.approx(math.sqrt(2 * 35 / 12 / 100))


@pytest.mark.parametrize("sides,dice", [(4, 1), (6, 2), (8, 3), (20, 5)])
def test_exact_pmf_matches_moments(sides, dice):
    pmf = ds.exact_pmf(sides, dice)
    assert min(pmf) == dice and max(pmf) == dice * sides
    assert sum(pmf.values()) == pytest.approx(1)
    mean = sum(t * p for t, p in pmf.items())
    var = sum((t - mean) ** 2 * p for t, p in pmf.items())
    assert mean == pytest.approx(ds.expected_value(sides, dice))
    assert var == pytest.approx(ds.variance_of_sum(sides, dice))


def test_two_d6_pmf():
    pmf = ds.exact_pmf(6, 2)
    assert pmf[7] == pytest.approx(6 / 36)
    assert pmf[2] == pytest.approx(1 / 36)


def test_rolls_stay_in_range_and_are_reproducible():
    a = ds.roll_dice(8, 3, 5000, np.random.default_rng(1))
    b = ds.roll_dice(8, 3, 5000, np.random.default_rng(1))
    assert a.min() >= 3 and a.max() <= 24
    assert np.array_equal(a, b)


def test_summary_covers_all_150_configurations(run):
    summary, histograms = run
    assert len(summary) == 6 * 5 * 5
    assert list(summary.columns[:6]) == ORIGINAL_COLUMNS
    assert set(histograms) == set(ds.HISTOGRAM_CONFIGS)
    assert all(len(v) == ds.HISTOGRAM_TRIALS for v in histograms.values())


def test_every_deviation_is_within_five_standard_errors(run):
    summary, _ = run
    assert summary["Deviation / SE"].max() < 5


def test_histogram_matches_exact_distribution():
    sums = ds.roll_dice(6, 2, 60_000, np.random.default_rng(7))
    pmf = ds.exact_pmf(6, 2)
    observed = np.array([(sums == t).sum() for t in pmf])
    expected = np.array([p * len(sums) for p in pmf.values()])
    chi2 = ((observed - expected) ** 2 / expected).sum()
    assert chi2 < 31.26  # 99.9th percentile of chi-square with 10 degrees of freedom


def test_cli_writes_summary(tmp_path):
    ds.main(["--seed", "3", "--out", str(tmp_path), "--no-plots"])
    out = pd.read_csv(tmp_path / "dice_roll_summary.csv")
    assert len(out) == 150


@pytest.mark.slow
def test_recovered_original_script_runs_unmodified(tmp_path):
    script = tmp_path / "dice_simulation_original.py"
    shutil.copy(ROOT / "original" / "dice_simulation_original.py", script)
    env = dict(os.environ, MPLBACKEND="Agg")
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, check=True, capture_output=True,
                   timeout=300)
    out = pd.read_csv(tmp_path / "dice_roll_summary.csv")
    assert list(out.columns) == ORIGINAL_COLUMNS
    assert len(out) == 150
    assert (out["Expected Value"] == out["Dice"] * (out["Sides"] + 1) / 2).all()
