"""Monte Carlo dice simulation (CSCI 154, Project 1): cleaned version of the recovered original.

The original Colab code is kept unchanged in original/dice_simulation_original.py. This version runs the
same experiment (every combination of 4/6/8/10/12/20-sided dice, 1-5 dice and 10-100,000 trials) and
writes the same summary columns, with these changes:

* a seeded NumPy generator, so a run can be reproduced exactly (the original was unseeded);
* figures are saved to files instead of plt.show(), so the script runs headless;
* the exact probability distribution of the sum is overlaid on the histograms;
* each row reports the standard error of the simulated mean, sqrt(dice * (sides^2 - 1) / 12 / trials).
  The deviation therefore shrinks in proportion to 1/sqrt(trials), not "exponentially" as the
  original report stated; the deviation plot shows this reference line.

Usage: python dice_simulation.py [--seed 154] [--out results/dice] [--no-plots]
"""
import argparse
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

DICE_SIDES = (4, 6, 8, 10, 12, 20)
DICE_COUNTS = (1, 2, 3, 4, 5)
TRIAL_COUNTS = (10, 100, 1_000, 10_000, 100_000)
HISTOGRAM_CONFIGS = ((1, 20), (2, 12), (3, 8), (4, 6), (5, 4))  # (dice, sides), as in the original
HISTOGRAM_TRIALS = 10_000


def expected_value(num_sides, num_dice):
    return num_dice * (num_sides + 1) / 2


def variance_of_sum(num_sides, num_dice):
    return num_dice * (num_sides ** 2 - 1) / 12


def standard_error(num_sides, num_dice, num_trials):
    return math.sqrt(variance_of_sum(num_sides, num_dice) / num_trials)


def exact_pmf(num_sides, num_dice):
    """Exact distribution of the sum of `num_dice` fair dice, as {total: probability}."""
    pmf = {0: 1.0}
    for _ in range(num_dice):
        nxt = {}
        for total, p in pmf.items():
            for face in range(1, num_sides + 1):
                nxt[total + face] = nxt.get(total + face, 0.0) + p / num_sides
        pmf = nxt
    return pmf


def roll_dice(num_sides, num_dice, num_trials, rng):
    """Sum of `num_dice` fair dice for each of `num_trials` independent rolls."""
    return rng.integers(1, num_sides + 1, size=(num_trials, num_dice)).sum(axis=1)


def run_simulation(seed=154, sides=DICE_SIDES, counts=DICE_COUNTS, trials=TRIAL_COUNTS):
    """Returns (summary DataFrame, {(dice, sides): sums for the 10,000-trial histogram runs})."""
    rng = np.random.default_rng(seed)
    rows, histograms = [], {}
    for num_sides, num_dice, num_trials in product(sides, counts, trials):
        sums = roll_dice(num_sides, num_dice, num_trials, rng)
        mean = float(sums.mean())
        ev = expected_value(num_sides, num_dice)
        se = standard_error(num_sides, num_dice, num_trials)
        rows.append({
            "Sides": num_sides,
            "Dice": num_dice,
            "Trials": num_trials,
            "Simulated Mean": mean,
            "Expected Value": ev,
            "Deviation": abs(mean - ev),
            "Standard Error": se,
            "Deviation / SE": abs(mean - ev) / se,
        })
        if num_trials == HISTOGRAM_TRIALS and (num_dice, num_sides) in HISTOGRAM_CONFIGS:
            histograms[(num_dice, num_sides)] = sums
    return pd.DataFrame(rows), histograms


def plot_histograms(histograms, path):
    import matplotlib.pyplot as plt

    configs = [c for c in HISTOGRAM_CONFIGS if c in histograms]
    fig, axes = plt.subplots(1, len(configs), figsize=(4 * len(configs), 3.6))
    for ax, (num_dice, num_sides) in zip(np.atleast_1d(axes), configs):
        sums = histograms[(num_dice, num_sides)]
        totals = np.arange(num_dice, num_dice * num_sides + 1)
        counts = np.array([(sums == t).sum() for t in totals])
        pmf = exact_pmf(num_sides, num_dice)
        ax.bar(totals, counts, color="skyblue", edgecolor="black", linewidth=0.4, label="simulated")
        ax.plot(totals, [pmf[t] * len(sums) for t in totals], "r.-", linewidth=1, label="exact")
        ax.set_title(f"{num_dice}d{num_sides}, {len(sums):,} rolls")
        ax.set_xlabel("Sum of dice")
    np.atleast_1d(axes)[0].set_ylabel("Frequency")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_convergence(summary, path):
    import matplotlib.pyplot as plt

    dice_values = sorted(summary["Dice"].unique())
    fig, axes = plt.subplots(2, len(dice_values), figsize=(4 * len(dice_values), 7), sharex=True)
    colors = dict(zip(DICE_SIDES, ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:brown", "black"]))
    for col, num_dice in enumerate(dice_values):
        top, bottom = axes[0, col], axes[1, col]
        for num_sides, cfg in summary[summary["Dice"] == num_dice].groupby("Sides"):
            cfg = cfg.sort_values("Trials")
            color = colors.get(num_sides, "gray")
            top.plot(cfg["Trials"], cfg["Simulated Mean"] - cfg["Expected Value"], "o-", color=color,
                     markersize=3, label=f"d{num_sides}")
            nonzero = cfg[cfg["Deviation"] > 0]  # an exact hit (deviation 0) cannot be drawn on a log axis
            bottom.plot(nonzero["Trials"], nonzero["Deviation"], "o-", color=color, markersize=3)
            bottom.plot(cfg["Trials"], cfg["Standard Error"], ":", color=color, linewidth=1)
        top.axhline(0, color="gray", linewidth=0.8)
        top.set_title(f"{num_dice} dice")
        bottom.set_xscale("log")
        bottom.set_yscale("log")
        bottom.set_xlabel("Trials")
    axes[0, 0].set_ylabel("Simulated mean - expected value")
    axes[1, 0].set_ylabel("|Deviation| (dotted: standard error)")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Convergence of the simulated mean: the error shrinks like 1/sqrt(trials)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--out", type=Path, default=Path("results/dice"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    summary, histograms = run_simulation(seed=args.seed)
    summary.to_csv(args.out / "dice_roll_summary.csv", index=False)
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        plot_histograms(histograms, args.out / "histograms.png")
        plot_convergence(summary, args.out / "convergence.png")

    final = summary[summary["Trials"] == max(TRIAL_COUNTS)]
    print(f"{len(summary)} configurations written to {args.out / 'dice_roll_summary.csv'}")
    print(f"At {max(TRIAL_COUNTS):,} trials: deviation < 0.01 in {(final['Deviation'] < 0.01).sum()} of "
          f"{len(final)} configurations; largest deviation / standard error = {final['Deviation / SE'].max():.2f}")


if __name__ == "__main__":
    main()
