# causal targeting simulation

This project mirrors the core causal-targeting pipeline, but replaces the real analysis data with a synthetic dataset generated in Python.

It is standalone:

- no external data files are required
- no paths outside this folder are used
- the project can be copied to another machine and run from this directory alone

## What is kept

- Pipeline notebooks:
  - `02_tune_CATE_estimators.ipynb`
  - `03a_GI_validate_CATE_estimators.ipynb`
  - `04a_rank_estimators.ipynb`
  - `04b_validate_ensembles.ipynb`
  - `05b_GI_rebuild_top_groups.ipynb`
- Core implementation:
  - `config/config.py`
  - `methods/`
  - `requirements.txt`
- Synthetic data generator:
  - `generate_simulated_data.py`

## Synthetic input

The generator writes the notebook input CSV to:

`data/analysis/analysis_df.csv`

Default design:

- `n = 12,000`
- `p = 50`
- AR(1) covariance with `rho = 0.3`
- randomized treatment `treatment ~ Bernoulli(0.5)`
- binary outcome `outcome`

## Usage

## Windows setup

From this project folder in PowerShell or Command Prompt:

```text
py -m pip install -r requirements.txt
py generate_simulated_data.py
```

Then open Jupyter from the same folder:

```text
py -m jupyter lab
```

or

```text
py -m jupyter notebook
```

## Pipeline

Generate the simulated dataset:

```text
python generate_simulated_data.py
```

Then run the notebooks in the same order as the mirrored pipeline:

1. `02_tune_CATE_estimators.ipynb`
2. `03a_GI_validate_CATE_estimators.ipynb`
3. `04a_rank_estimators.ipynb`
4. `04b_validate_ensembles.ipynb`
5. `05b_GI_rebuild_top_groups.ipynb`

## Notes

- Run Jupyter with this folder as the working directory so relative paths resolve correctly.
- Forward slashes used in Python paths are cross-platform and work on Windows.
- Output folders are created by the notebooks as needed.
