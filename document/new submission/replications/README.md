# Replication Code for "Stable Causal Targeting with Machine Learning"

This folder contains the minimal production scripts needed to replicate the
JBES submission results.

## Files

| File | Purpose |
|------|---------|
| `evaluate_proposed_targeting_statistics.py` | Core statistics engine: fold-level calibration metrics (including `CalSlope`), targeting contrast evaluation, simulation and real-data fold-vector construction. Imported as `base` by the procedure script. |
| `screen_anchor_augmented_procedure.py` | Main procedure: anchor selection by `cal_slope`, nested-ensemble path construction, ensemble selection by `mean_delta_top10`. Runs both simulation (DGP1, DGP2) and real-data (billpayfa, debitfa, main) arms. |
| `screen_large_selection_bank_simfirst.py` | Complete development audit: defines 40 statistics on the calibration-slope nested path using all 23 estimators. |
| `build_40_stat_bank_from_cached_predictions.py` | Reuses the immutable source output, replaces two redundant labels with `median_delta_top10` and `trimmed_mean_delta_top10`, adds `lcb_delta_top10`, and produces 40 distinct rules without fitting estimators. |
| `audit_fitted_tree_forest_artifacts.py` | Verifies that the two causal trees and two causal forests are present in every real-data perturbation library and produce finite saved-fold and holdout predictions. |
| `review_large_selection_bank.py` | Refits selected ensembles for the development rules on the three real-data arms and merges those diagnostics with the simulation summaries. |
| `reproduce_bbar_first_stage_procedure.py` | Prediction-only sensitivity analysis that replaces calibration slope with `B_bar_0.9` for first-stage ordering while retaining `mean_delta_top10` for second-stage path-length selection. |
| `plot_primary_ensemble_true_cate_barchart.py` | Generates main-text Figure 5 from the primary-procedure output and saved prediction caches. |
| `plot_mean_delta_top10_ensemble_true_cate_barchart.py` | Generates the common-support calibration-slope ranking figure used as Supplementary Figure S1. |
| `plot_bbar_first_stage_ensemble_true_cate_barchart.py` | Generates the Supplementary ranking figure for the `B_bar_0.9`-first-stage ensemble. |

## Dependencies

- Python 3.10+
- numpy, pandas, scipy, joblib

## Key Method: `CalSlope`

The calibration slope (`cal_slope`) ranks candidate CATE estimators by how well
their predicted ordering of treatment effects matches observed effects on
validation data.

### Stratum formation (5 bins)

1. **Training-fold quintile cutoffs**: For each estimator-fold pair, sort the
   training-fold CATE predictions and compute the 20th, 40th, 60th, and 80th
   percentiles. These four values are the stratum boundaries.

2. **Apply to validation fold**: Assign each validation unit to a stratum by
   comparing its CATE prediction to these training-derived boundaries. This
   prevents data leakage — the bin boundaries are determined without using
   validation outcomes.

3. **Drop degenerate strata**: Strata with no treated or no control
   observations are omitted from the regression.

4. **OLS slope**: Regress observed stratum ATE (Neyman difference-in-means) on
   mean predicted CATE. The slope is `Cov(mean_CATE, ATE) / Var(mean_CATE)`.
   At least 2 valid strata with non-constant means are required.

5. **Fold average**: `CalSlope_m` is the mean of `CalSlope_{mf}` across all
   valid validation folds for estimator `m`.

### Interpretation

- **> 0**: Higher predicted-CATE strata have higher realized ATEs (ranking signal)
- **≈ 0**: No ranking signal
- **< 0**: Ranking is reversed on validation data

## Output

Simulation output is written to:
- `projects/causal-targeting-simulation/output/anchor_expanded_procedure/sim_detail.csv`
- `projects/causal-targeting-simulation/output/anchor_expanded_procedure/sim_summary.csv`
- `projects/causal-targeting-simulation/output/anchor_expanded_procedure/realdata_results.csv`

The statistic-bank development audit writes to:
- `projects/causal-targeting-simulation/output/large_selection_bank_simfirst_40/sim_detail.csv`
- `projects/causal-targeting-simulation/output/large_selection_bank_simfirst_40/sim_summary.csv`
- `projects/causal-targeting-simulation/output/large_selection_bank_simfirst_40/simulation_winner_ranking.csv`
- `projects/causal-targeting-simulation/output/large_selection_bank_simfirst_40/simulation_true_cate_ranks.csv`
- `projects/causal-targeting-simulation/output/large_selection_bank_simfirst_40/fitted_tree_forest_artifact_audit.csv`

The first-stage sensitivity analysis writes to:
- `projects/causal-targeting-simulation/output/bbar_first_stage_reproduction/simulation_detail.csv`
- `projects/causal-targeting-simulation/output/bbar_first_stage_reproduction/simulation_summary.csv`
- `projects/causal-targeting-simulation/output/bbar_first_stage_reproduction/simulation_method_ranks.csv`
- `projects/causal-targeting-simulation/output/bbar_first_stage_reproduction/paired_first_stage_scores.csv`
- `projects/causal-targeting-simulation/output/bbar_first_stage_reproduction/paired_first_stage_scores_summary.csv`
- `projects/causal-targeting-simulation/output/bbar_first_stage_reproduction/realdata_results.csv`

The bank audit is exploratory. The final production script fixes the selector to
`mean_delta_top10`; it does not rerun the bank or choose a statistic at runtime.
The `--smoke` option uses two seeds per DGP and writes to
`projects/causal-targeting-simulation/output/large_selection_bank_smoke/` so it
cannot overwrite the complete 40-rule results.

## Running

The scripts require pre-computed Monte Carlo caches and real-data fitted
libraries. These are in the sibling project directories:
- `projects/causal-targeting-simulation/output/dgp1/monte_carlo/`
- `projects/causal-targeting-simulation/output/dgp2/monte_carlo/`
- `projects/causal-targeting-main/output/analysis/fausebal/`
- `projects/causal-targeting-debitfa/output/analysis/fausebal/`
- `projects/causal-targeting-billpayfa/output/analysis/fausebal/`

```bash
cd projects/causal-targeting-simulation
python3 screen_anchor_augmented_procedure.py
python3 plot_primary_ensemble_true_cate_barchart.py
```

To reconstruct the 40-rule result from the saved fitted prediction objects:

```bash
cd "new submission/replications"
python3 build_40_stat_bank_from_cached_predictions.py
python3 audit_fitted_tree_forest_artifacts.py --arm billpayfa
python3 audit_fitted_tree_forest_artifacts.py --arm debitfa
python3 audit_fitted_tree_forest_artifacts.py --arm main
python3 audit_fitted_tree_forest_artifacts.py --merge
```

Running `screen_large_selection_bank_simfirst.py` performs a complete 40-rule
execution from the saved prediction caches. It is not required when the
immutable 39-rule output is already available.

To reproduce the alternative first-stage score and its figure without fitting
CATE estimators:

```bash
cd projects/causal-targeting-simulation
python3 reproduce_bbar_first_stage_procedure.py
python3 plot_mean_delta_top10_ensemble_true_cate_barchart.py
python3 plot_bbar_first_stage_ensemble_true_cate_barchart.py
```

Within each replication, both first-stage orderings use the same estimators
having finite top-decile contrasts, fold t statistics, and calibration slopes
on all 12 folds. The shared application cache does not contain full-fit arrays
for the four tree/forest estimators; the sensitivity script therefore reports
an application holdout estimate only when every selected member has available
cached training-validation and holdout predictions.
