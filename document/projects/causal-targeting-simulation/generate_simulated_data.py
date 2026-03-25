from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ACTIVE_DIMENSION = 5
DEFAULT_SEED = 405


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
    if dimension < ACTIVE_DIMENSION:
        raise ValueError(
            f"dimension must be at least {ACTIVE_DIMENSION} because the DGP uses "
            "the first five coordinates."
        )

    rng = np.random.default_rng(seed)
    covariance = ar1_covariance(dimension, rho)
    cholesky = np.linalg.cholesky(covariance)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        covariates = rng.normal(size=(n_rows, dimension)) @ cholesky.T
    active_covariates = covariates[:, :ACTIVE_DIMENSION]
    treatment = rng.binomial(1, 0.5, size=n_rows).astype(int)

    baseline = (
        np.maximum.reduce(
            [
                np.zeros(n_rows),
                active_covariates[:, 0] + active_covariates[:, 1],
                active_covariates[:, 2],
            ]
        )
        + np.maximum(
            np.zeros(n_rows),
            active_covariates[:, 3] + active_covariates[:, 4],
        )
    )
    treatment_effect = -0.05 + 0.15 * (
        active_covariates[:, 0] + softplus(active_covariates[:, 1])
    )
    probability = expit(baseline + (treatment - 0.5) * treatment_effect)
    outcome = rng.binomial(1, probability, size=n_rows).astype(int)

    frame = pd.DataFrame({"treatment": treatment, "outcome": outcome})

    for feature_index in range(dimension):
        frame[f"X{feature_index + 1}"] = covariates[:, feature_index]

    ordered_columns = ["treatment", "outcome"] + [
        f"X{feature_index + 1}" for feature_index in range(dimension)
    ]
    return frame.loc[:, ordered_columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the synthetic analysis dataset. The default arguments match "
            "new submission/simulation_plan.tex."
        )
    )
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--p", type=int, default=50)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
