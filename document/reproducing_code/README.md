# JBES Reproduction

This directory supports two workflows. The default workflow reproduces the
paper directly from the included canonical computation cache. The second
workflow rebuilds that cache by refitting the estimator libraries from the
bundled analysis inputs and then runs the same reporting checks.

The public selection rule is the same in both workflows: estimators are ordered
by their validation-fold calibration slopes, nested equal-weight ensembles are
formed in that order, and the ensemble with the largest mean validation
top-decile treatment-effect contrast is selected.

## Files

- `reproduce.py` computes tables, figures, and checks from a prediction cache.
- `regenerate.py` performs model refitting and builds a new prediction cache.
- `run.sh` provides the two run options.
- `requirements.txt` lists the direct Python dependencies.
- `cache/` is the read-only canonical computation cache.
- `data/` contains the prepared inputs needed for direct refitting.
- `refit/` contains the local Python estimator and selection implementation.

All runtime paths are relative to this directory. Cached reproduction does not
depend on `data/`, `refit/`, external repositories, or user-specific Python
modules. Direct refitting uses only the bundled `data/` and `refit/` files.

## Installation

Python 3.10 or newer is recommended.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Cached Reproduction

The default command reads `cache/` and writes the reproduced results under
`outputs/`:

```sh
./run.sh
```

The explicit equivalent is:

```sh
./run.sh cached
```

`outputs/figs/` contains the manuscript figures under the filenames used
by the LaTeX article. `outputs/tables/` contains the numerical results and the
data underlying the diagnostic figures. Checks against the article are kept in
memory, and the command exits with an error if any reported value differs.

## Direct Refitting

Direct refitting is time consuming. It rebuilds the estimator libraries from
the bundled analysis data and writes new prediction caches under
`cache_regenerated/`.

```sh
./run.sh regenerate
```
