# causal targetting for Turkey bank overdraft

Targeted CATE estimation and interpretable policy learning for Turkey bank overdraft messaging. This repo focuses on a four-step workflow centered on notebooks 02–05.

## Core notebooks (02–05)

- 02_tune_CATE_estimators.ipynb
  - Objective: Tune heterogeneous treatment effect (CATE) estimators (e.g., X-learners, causal forests) with cross-validation; perform imputation where needed.
  - Inputs: Prepared analysis data (see `data/`), configuration in `config/config.py`.
  - Key steps: define search spaces, k-fold CV, select best by validation metric; handling missing features.
  - Outputs: tuned params and imputation metadata in `output/params/` (e.g., `analysis_imputation_meta.pkl`, `fausebal_tuned_params.pkl`).

- 03a_GI_validate_CATE_estimators.ipynb
  - Objective: Train and validate tuned estimators; compute out-of-fold CATEs and evaluation metrics.
  - Inputs: artifacts from 02, analysis data.
  - Key steps: fit per-fold models, generate CATE predictions, compute calibration-style metrics.
  - Outputs: per-observation predictions and folds in `output/cate_data/fausebal/trainval_data.csv`; aggregated validation summaries in `output/analysis/fausebal/trainval_data.csv`.

- 04a_GI_rank_CATE_estimators.ipynb
  - Objective: Rank and compare estimators across metrics; create summary tables/figures.
  - Inputs: validation artifacts from 03a.
  - Key steps: aggregate metrics, tie-break, sensitivity checks; build comparison plots.
  - Outputs: tables in `output/tables/` (e.g., `analysis_regression_summary.csv`) and figures in `output/figures/`.

- 05a_GI_cell_search.ipynb
  - Objective: Learn interpretable policies using greedy cell search (rule-based segmentation) over CATEs and covariates.
  - Inputs: ranked estimator selection from 04a, predictions from 03a.
  - Key steps: call into `methods/greedy_cell_search.py` and `methods/cell_search.py` to construct cells; evaluate uplift and coverage; export human-readable rules.
  - Outputs: segmentation rules and policy summaries exported to `output/tables/` and `output/figures/`.

### Run order

1) 02_tune_CATE_estimators.ipynb → 2) 03a_GI_validate_CATE_estimators.ipynb: 1, Set up; 2 Calibration and R2 (predictability) → 3) 04a_GI_rank_CATE_estimators.ipynb → 4) 05a_GI_cell_search.ipynb → 5) 03a_GI_validate_CATE_estimators.ipynb: 3. Monotonicity; 4. Stability of CATE estimators

Open each in Jupyter and run top-to-bottom after verifying paths in `config/config.py`.


