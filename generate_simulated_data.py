from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def ar1_covariance(dimension: int, rho: float) -> np.ndarray:
    index = np.arange(dimension)
    return rho ** np.abs(index[:, None] - index[None, :])


def build_dataframe(
    n_rows: int,
    dimension: int,
    rho: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    covariance = ar1_covariance(dimension, rho)
    cholesky = np.linalg.cholesky(covariance)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        covariates = rng.normal(size=(n_rows, dimension)) @ cholesky.T
    treatment = rng.binomial(1, 0.5, size=n_rows).astype(int)

    baseline = (
        np.maximum.reduce(
            [
                np.zeros(n_rows),
                covariates[:, 0] + covariates[:, 1],
                covariates[:, 2],
            ]
        )
        + np.maximum(np.zeros(n_rows), covariates[:, 3] + covariates[:, 4])
    )
    treatment_effect = -0.05 + 0.15 * (
        covariates[:, 0] + softplus(covariates[:, 1])
    )
    probability = expit(baseline + (treatment - 0.5) * treatment_effect)
    outcome = rng.binomial(1, probability, size=n_rows).astype(int)

    frame = pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1),
            "group": 0,
            "sample_flag": 1,
            "treatment": treatment,
            "debit": 0,
            "control": 1 - treatment,
            "outcome": outcome,
            "subgroup_flag_1": (covariates[:, 0] + covariates[:, 1] > 0).astype(int),
            "subgroup_flag_2": (covariates[:, 3] + covariates[:, 4] > 0).astype(int),
            "aux_flag_1": rng.binomial(1, expit(0.6 * covariates[:, 5])),
            "aux_flag_3": rng.binomial(1, 0.5, size=n_rows),
            "assets": np.exp(1.2 + 0.5 * covariates[:, 0]),
            "deposits": np.exp(1.0 + 0.5 * covariates[:, 1]),
            "paymentmean": np.exp(0.7 + 0.4 * covariates[:, 2]),
            "debt": np.exp(0.9 + 0.5 * covariates[:, 3]),
            "minbal": 0.4 * covariates[:, 0] - 0.3 * covariates[:, 3] + 0.2 * covariates[:, 4],
            "creditcard": rng.binomial(1, expit(covariates[:, 4])),
        }
    )
    frame["aux_flag_2"] = 1 - frame["aux_flag_1"]

    for feature_index in range(dimension):
        frame[f"strat_{feature_index + 1}"] = covariates[:, feature_index]

    ordered_columns = [
        "id",
        "group",
        "sample_flag",
        "treatment",
        "debit",
        "control",
        "outcome",
        "subgroup_flag_1",
        "subgroup_flag_2",
        "aux_flag_1",
        "aux_flag_2",
        "aux_flag_3",
        "assets",
        "deposits",
        "paymentmean",
        "debt",
        "minbal",
        "creditcard",
    ] + [f"strat_{feature_index + 1}" for feature_index in range(dimension)]
    return frame.loc[:, ordered_columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--p", type=int, default=50)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=405)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis/analysis_df.csv"),
    )
    arguments = parser.parse_args()

    data_frame = build_dataframe(
        n_rows=arguments.n,
        dimension=arguments.p,
        rho=arguments.rho,
        seed=arguments.seed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(arguments.output, index=False)

    treated_probability = data_frame.loc[data_frame["treatment"] == 1, "outcome"].mean()
    control_probability = data_frame.loc[data_frame["treatment"] == 0, "outcome"].mean()
    print(
        {
            "rows": int(len(data_frame)),
            "columns": int(data_frame.shape[1]),
            "treated_share": float(data_frame["treatment"].mean()),
            "outcome_rate": float(data_frame["outcome"].mean()),
            "sample_ate": float(treated_probability - control_probability),
            "output": str(arguments.output),
        }
    )


if __name__ == "__main__":
    main()
