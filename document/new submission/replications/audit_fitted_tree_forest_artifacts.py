#!/usr/bin/env python3
"""Verify saved causal-tree and causal-forest fits without refitting models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_proposed_targeting_statistics as base
import screen_anchor_augmented_procedure as current


OUTPUT_DIR = base.ROOT / "output" / "large_selection_bank_simfirst_40"
ESTIMATORS = [
    "causal_tree_1",
    "causal_tree_2",
    "causal_forest_1",
    "causal_forest_2",
]


def audit_arm(arm: str) -> pd.DataFrame:
    project_dir = base.PROJECTS[arm]
    data = current.load_real_project(project_dir)
    fitted_libs = data["fitted_libs"]
    holdout_df = data["holdout_df"]
    features = data["features"]
    x_hold = (
        holdout_df[features]
        .copy()
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    rows = []
    for perturbation, library in fitted_libs.items():
        if len(library) != 23:
            raise ValueError(f"{arm}/{perturbation}: expected 23 fitted estimators, found {len(library)}")
        for name in ESTIMATORS:
            if name not in library:
                raise KeyError(f"{arm}/{perturbation}: missing {name}")
            estimator = library[name]
            for fold, result in sorted(estimator.results.items()):
                tau_tv = np.asarray(result.tau, dtype=float)
                tau_ho = current.predict_on_holdout(result, estimator, x_hold)
                rows.append(
                    {
                        "arm": arm,
                        "perturbation": perturbation,
                        "estimator": name,
                        "fold": fold,
                        "n_trainval": len(tau_tv),
                        "n_holdout": len(tau_ho),
                        "n_finite_trainval": int(np.isfinite(tau_tv).sum()),
                        "n_finite_holdout": int(np.isfinite(tau_ho).sum()),
                    }
                )
    return pd.DataFrame(rows)


def merge_outputs() -> Path:
    paths = [OUTPUT_DIR / f"fitted_tree_forest_artifact_audit_{arm}.csv" for arm in base.PROJECTS]
    frames = [pd.read_csv(path) for path in paths]
    audit = pd.concat(frames, ignore_index=True)
    complete = (
        audit["n_finite_trainval"].eq(audit["n_trainval"])
        & audit["n_finite_holdout"].eq(audit["n_holdout"])
    )
    if not complete.all():
        raise ValueError("At least one saved fitted estimator produced non-finite predictions")
    output = OUTPUT_DIR / "fitted_tree_forest_artifact_audit.csv"
    audit.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(base.PROJECTS))
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.merge:
        print(f"Saved {merge_outputs()}")
        return
    if args.arm is None:
        parser.error("supply --arm or --merge")
    audit = audit_arm(args.arm)
    output = OUTPUT_DIR / f"fitted_tree_forest_artifact_audit_{args.arm}.csv"
    audit.to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
