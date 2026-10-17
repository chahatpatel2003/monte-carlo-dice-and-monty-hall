# Provenance

These are CSCI 154 (Simulation) course projects from California State University, Fresno, Summer 2025. The files were recovered from Google Drive in October 2026. Only the two project reports (Word documents) survived; the Colab notebooks did not. This file records what came from where and what was changed.

## Project 1: dice simulation

**Source.** The report `154 project 1 report v7.docx` has an appendix containing the full Python code. That code is a Colab export of `CSCI_154_Project_1_Dice_Simulation (Complete).ipynb`.

**`original/dice_simulation_original.py`.** The appendix code, copied verbatim, including the Colab export header.
- SHA-256: `3fcac773b8a9acf5f744ef7112f642ce9dfed76aca2dcba1c259596aa51e8bbb`.
- It runs without modification: `tests/test_dice.py::test_recovered_original_script_runs_unmodified` runs it and checks its 150-row CSV.
- It is unseeded, so its numbers change on every run.

**Not recovered.** The original notebook with its saved outputs, and its `dice_roll_summary.csv`.

**`dice_simulation.py`.** Cleaned in October 2026, AI-assisted. It loops over the same experiment grid, keeps the same six summary columns in the same order, and makes the same five histogram choices. The changes are:

| Change | Why |
|---|---|
| NumPy `default_rng(seed)` with vectorised rolls, replacing an unseeded `random.randint` loop | Reproducible runs |
| Figures saved to files instead of `plt.show()` | Runs headless; outputs can be committed |
| Exact distribution of the sum overlaid on the histograms | Shows the distribution against theory, not just its shape |
| `Standard Error` and `Deviation / SE` columns, and standard-error reference lines on the deviation plot | Corrects the report's claim that deviation "drops exponentially". It falls like 1/√N. |
| 15 separate figures merged into 2 | Readability |

## Project 2: Monty Hall

**Source.** The report `CSCI 154 Project 2 Report.docx` describes:
- the model and its variables (doors, prize door, initial pick, host door, decision);
- the libraries used (`random`, `matplotlib`, `pandas`);
- separate functions for the standard and the random-host games;
- a user-chosen number of runs;
- a convergence graph.

It contains six result figures, which are spreadsheet screenshots and one plot. They are saved in `original/report_figures/`:

| File | Report figure | Trials | Host error *r* |
|---|---|---:|---|
| `monty_hall_fig1_n10.png` | Figure 1 | 10 | not stated |
| `monty_hall_fig2_n100.png` | Figure 2 | 100 | not stated |
| `monty_hall_fig3_n100000.png` | Figure 3 | 100,000 | not stated; 0.1 inferred from its ruined-game rates |
| `monty_hall_fig4_ruin_n10000_r1.png` | Figure 4 | 10,000 | 1 |
| `monty_hall_fig5_n100000_r1.png` | Figure 5 | 100,000 | 1 |
| `monty_hall_fig6_convergence.png` | Figure 6 | 100 points up to 100,000 | standard game, 3 doors, stay |

**`data/recovered_report_tables.csv`.** Figures 1–5, transcribed by hand. Some long decimals were cut off in the screenshots; only the visible digits are recorded.

**Not recovered.** The original Monty Hall code and the spreadsheet behind the figures.

**`monty_hall.py` is a reconstruction, not the original code.**
- It was written in October 2026, AI-assisted, from the report's description of the model.
- It follows the report's design: two game functions, `random`, `pandas` and `matplotlib`, and the same table columns.
- It also adds exact formulas and tests.
- Its results are new seeded runs; they are not the original results.

Three details are not stated in the report and were inferred from the recovered tables:

1. **Win rates in the unpredictable game exclude ruined games.**
   - In Figure 1, 0.625 = 5/8, with 2 of 10 games ruined.
   - In Figure 2, 0.351 = 33/94, with 6 of 100 games ruined.
   - Across all 20 table rows, stay rate × games not ruined is a whole number.
2. **The ruined-game column is the ruined share of the stay run.** The denominators in point 1 match it, while the switch run has its own ruined count.
3. **Figure 3 used *r* = 0.1.** Its ruined rates are 0.0334, 0.0163, 0.0103, 0.0048 and 0.0010, close to *r*/*N*.

**Validation.** `tests/test_recovered_report.py` compares both 100,000-game tables from the report with the reconstructed model's exact probabilities:
- Figure 3 (*r* = 0.1): χ² = 31.0 on 25 degrees of freedom.
- Figure 5 (*r* = 1): χ² = 28.8 on 25 degrees of freedom.
- No cell is more than 2.4 standard errors from its exact value.
- `tests/test_monty_hall.py` checks that the reconstruction's own simulations agree with the same exact probabilities.

## Corrections to the original reports

- **Dice, "the deviation drops exponentially as the number of trials increases":** incorrect. The deviation of the mean falls like 1/√N.
- **Dice, "most setups had variances smaller than 0.01 by 100,000 trials":** the deviations are meant. Theory predicts about 58% of the 30 configurations; the seeded run here gives 18 of 30. High-variance configurations such as 5d20 (standard error 0.041) usually stay above 0.01.
