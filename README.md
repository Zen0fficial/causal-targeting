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

- 05b_GI_rebuild_top_groups.ipynb
  - Objective: Rebuild top subgroups from prior steps and refit on the selected units; compute both Neyman pooled tests and model-based ATE with 95% CIs per estimator on fixed subgroups.
  - Inputs: artifacts from 02/03a/04a (fitted libraries, top estimator names), train/val and holdout csvs in `output/analysis/<outcome>/`.
  - Key steps: reconstruct subgroup masks, retune on subgroup, and produce a per-estimator table using each fold’s pre-trained meta-learner (`estimate_ate(pretrain=True)`) for model-based inference; no new subgroup construction.
  - Outputs: final subgroup stats tables in the notebook output; updated fitted subgroup libraries cached in `output/analysis/<outcome>/*subgroup_fitted_libraries.pkl`.

### Run order

1) 02_tune_CATE_estimators.ipynb → 2) 03a_GI_validate_CATE_estimators.ipynb: 1, Set up; 2 Calibration and R2 (predictability) → 3) 04a_GI_rank_CATE_estimators.ipynb → 4) 05a_GI_cell_search.ipynb → 5) 03a_GI_validate_CATE_estimators.ipynb: 3. Monotonicity; 4. Stability of CATE estimators → 6) 05b_GI_rebuild_top_groups.ipynb

Open each in Jupyter and run top-to-bottom after verifying paths in `config/config.py`.


## Setup and outputs

- The notebooks now auto-create the standard save locations at startup:
  - `output/analysis/`
  - `output/params/`
  - `output/figures/`
  - `output/tables/`
- Artifacts by step:
  - 02 saves tuned params and imputation/meta to `output/params/<outcome>/` and aligned datasets to `output/analysis/<outcome>/`.
  - 03a saves per-fold libraries and validation data under `output/analysis/<outcome>/`.
  - 04a writes comparison tables to `output/tables/` and figures to `output/figures/`.
  - 05a writes subgroup/policy tables and figures to `output/tables/` and `output/figures/`.
  - 05b saves subgroup refit libraries to `output/analysis/<outcome>/` and shows final ATE/CI tables inline.

Dependencies: install via `pip install -r requirements.txt`. `xgboost` is used as an optional base learner in 02/03a; if not available, you can comment it out in those notebooks.

