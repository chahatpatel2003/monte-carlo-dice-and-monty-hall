"""Monty Hall simulation (CSCI 154, Project 2): RECONSTRUCTED implementation.

The original code for this project was not recovered. This module was written in October 2026 (with AI
assistance) from the project report's description of the model, and checked against the result tables
that were recovered from the report (data/recovered_report_tables.csv, tests/test_recovered_report.py).
It is not the original code and its numbers are new runs, not the original results.

Model, as described in the report:
* Nd doors (3, 6, 9, 20 and 100), one prize door p and the player's initial pick i, chosen uniformly.
* Standard game: the host knows where the prize is and opens one door that is neither i nor p.
* Unpredictable game: with probability r ("host error") the host opens a random door other than i.
  If that door hides the prize the game is ruined; otherwise play continues as in the standard game.
  P(ruined) = r * (Nd - 1)/Nd * 1/(Nd - 1) = r / Nd.
* Stay keeps door i; switch moves to a random door that is neither i nor the opened door.

Three details were inferred from the recovered tables rather than stated in the report:
* unpredictable-game win rates exclude ruined games (Figures 1-2: 0.625 = 5/8 with 2 of 10 games
  ruined, 0.351 = 33/94 with 6 of 100 ruined);
* the Ruined_Games_Rate column is the ruined share of the stay run;
* Figure 3 used r = 0.1: its ruined rates are 0.0334, 0.0163, 0.0103, 0.0048 and 0.0010 for 3-100 doors.

Usage: python monty_hall.py [--trials 100000] [--host-error 0.1] [--seed 154] [--out results/monty_hall]
"""
import argparse
import math
import random
from pathlib import Path

import pandas as pd

DOOR_COUNTS = (3, 6, 9, 20, 100)
COLUMNS = ["Doors", "Standard_Stay", "Standard_Switch", "Unpredictable_Stay", "Unpredictable_Switch",
           "Ruined_Games_Rate"]


def _pick_excluding(num_doors, excluded, rng):
    """A uniformly random door in range(num_doors) that is not in the set `excluded`."""
    k = rng.randrange(num_doors - len(excluded))
    for door in sorted(excluded):
        if k >= door:
            k += 1
    return k


def play_standard(num_doors, switch, rng):
    """One standard game. Returns True if the player wins."""
    prize = rng.randrange(num_doors)
    pick = rng.randrange(num_doors)
    opened = _pick_excluding(num_doors, {pick, prize}, rng)
    if switch:
        pick = _pick_excluding(num_doors, {pick, opened}, rng)
    return pick == prize


def play_unpredictable(num_doors, switch, host_error, rng):
    """One unpredictable game. Returns (won, ruined); a ruined game is neither won nor lost."""
    prize = rng.randrange(num_doors)
    pick = rng.randrange(num_doors)
    if rng.random() < host_error:
        opened = _pick_excluding(num_doors, {pick}, rng)
        if opened == prize:
            return False, True
    else:
        opened = _pick_excluding(num_doors, {pick, prize}, rng)
    if switch:
        pick = _pick_excluding(num_doors, {pick, opened}, rng)
    return pick == prize, False


def simulate_standard(num_doors, switch, trials, rng):
    """Share of `trials` standard games won."""
    return sum(play_standard(num_doors, switch, rng) for _ in range(trials)) / trials


def simulate_unpredictable(num_doors, switch, trials, host_error, rng):
    """Returns (wins / games not ruined, ruined games / trials)."""
    wins = ruined = 0
    for _ in range(trials):
        won, was_ruined = play_unpredictable(num_doors, switch, host_error, rng)
        wins += won
        ruined += was_ruined
    played = trials - ruined
    return (wins / played if played else math.nan), ruined / trials


def results_table(trials, host_error, seed=None, door_counts=DOOR_COUNTS):
    """Simulated table with the same columns as the recovered report tables."""
    rng = random.Random(seed)
    rows = []
    for doors in door_counts:
        unpredictable_stay, ruined_rate = simulate_unpredictable(doors, False, trials, host_error, rng)
        unpredictable_switch, _ = simulate_unpredictable(doors, True, trials, host_error, rng)
        rows.append([doors,
                     simulate_standard(doors, False, trials, rng),
                     simulate_standard(doors, True, trials, rng),
                     unpredictable_stay, unpredictable_switch, ruined_rate])
    return pd.DataFrame(rows, columns=COLUMNS)


def exact_standard(num_doors, switch):
    if not switch:
        return 1 / num_doors
    return (num_doors - 1) / (num_doors * (num_doors - 2))


def exact_ruined_rate(num_doors, host_error):
    return host_error / num_doors


def exact_unpredictable(num_doors, switch, host_error):
    """Win probability in the unpredictable game, given that the game was not ruined."""
    n, r = num_doors, host_error
    not_ruined = 1 - r / n
    if not switch:
        return (1 / n) / not_ruined
    # The player's first pick is wrong (prob (n-1)/n). The host then opens a non-prize door: knowingly with
    # prob 1-r, or by accident with prob r * (n-2)/(n-1). Either way the switch hits the prize with prob 1/(n-2).
    win = (n - 1) / n * ((1 - r) / (n - 2) + r / (n - 1))
    return win / not_ruined


def exact_table(host_error, door_counts=DOOR_COUNTS):
    return pd.DataFrame([[n, exact_standard(n, False), exact_standard(n, True),
                          exact_unpredictable(n, False, host_error), exact_unpredictable(n, True, host_error),
                          exact_ruined_rate(n, host_error)] for n in door_counts], columns=COLUMNS)


def convergence(num_doors=3, switch=False, max_trials=100_000, points=100, seed=None):
    """Standard-game win rate from independent runs of max_trials/points, 2*max_trials/points, ... trials.

    The recovered Figure 6 has 100 points for n = 100,000 whose values jump independently from point to
    point, so each point is taken to be a separate run rather than a running average.
    """
    rng = random.Random(seed)
    step = max_trials // points
    rows = [(n, simulate_standard(num_doors, switch, n, rng)) for n in range(step, max_trials + 1, step)]
    return pd.DataFrame(rows, columns=["Trials", "Win_Rate"])


def plot_convergence(curve, num_doors, switch, path):
    import matplotlib.pyplot as plt

    exact = exact_standard(num_doors, switch)
    se = [math.sqrt(exact * (1 - exact) / n) for n in curve["Trials"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["Trials"], curve["Win_Rate"], "o-", markersize=3, label=f"{'Switch' if switch else 'Stay'} strategy")
    ax.axhline(exact, color="black", linewidth=1, label=f"Exact = {exact:.4f}")
    ax.fill_between(curve["Trials"], [exact - 2 * s for s in se], [exact + 2 * s for s in se], color="gray",
                    alpha=0.2, label="Exact ± 2 standard errors")
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Winning probability")
    ax.set_title(f"Convergence of the Monty Hall simulation\n{num_doors} doors, strategy = "
                 f"{'switch' if switch else 'stay'}")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--host-error", type=float, default=0.1, help="probability r that the host errs")
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--out", type=Path, default=Path("results/monty_hall"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    simulated = results_table(args.trials, args.host_error, seed=args.seed)
    exact = exact_table(args.host_error)
    tag = f"n{args.trials}_r{args.host_error:g}"
    simulated.to_csv(args.out / f"simulated_{tag}.csv", index=False)
    exact.to_csv(args.out / f"exact_r{args.host_error:g}.csv", index=False)
    pd.set_option("display.width", 140)
    print(f"Simulated, {args.trials:,} games per cell, host error r = {args.host_error:g}:")
    print(simulated.round(5).to_string(index=False))
    print("\nExact probabilities:")
    print(exact.round(5).to_string(index=False))

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        curve = convergence(seed=args.seed)
        curve.to_csv(args.out / "convergence_3_doors_stay.csv", index=False)
        plot_convergence(curve, 3, False, args.out / "convergence_3_doors_stay.png")


if __name__ == "__main__":
    main()
