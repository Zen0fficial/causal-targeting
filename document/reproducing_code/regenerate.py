#!/usr/bin/env python3
"""Refit the CATE libraries and build a new prediction cache.

Simulation splits and ground-truth CATEs are reconstructed from the bundled
analysis inputs and DGP definitions. New prediction caches are written to a
separate directory; the canonical cache is never read or modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

RUNTIME_CACHE = Path(tempfile.gettempdir()) / "jbes-regeneration-cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")
warnings.filterwarnings(
    "ignore",
    message=r"Inconsistent values: penalty=.*",
    category=UserWarning,
    module=r"sklearn\.linear_model\._logistic",
)

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "cache_regenerated"
REFIT_ROOT = ROOT / "refit"
SIMULATION_DATA = ROOT / "data" / "simulation"
APPLICATION_DATA = ROOT / "data" / "application" / "analysis_df.csv"

sys.path.insert(0, str(REFIT_ROOT))

import methods.cate_estimator_validation as validation  # noqa: E402
from methods.cate_estimator_validation import (  # noqa: E402
    get_calibration_plot_data,
    get_cr2_plot_data,
    make_estimator_library,
)
from methods.cate_estimator_wrappers import CATEEstimatorResults, XLearnerWrapper  # noqa: E402

validation.mp.cpu_count = lambda: 2


N_SPLITS = 4
OUTCOME = "y"
TREATMENT = "TREATED"

PARAM_GRIDS = {
    "lasso": {"alpha": np.logspace(-5, 5, 500)},
    "logistic": {"penalty": ["l1", "l2"], "C": np.logspace(-5, 5, 500)},
    "rf": {
        "min_samples_leaf": [50, 100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "bootstrap": [False, True],
        "n_estimators": [100, 200, 300, 400, 500],
    },
    "xgb": {
        "max_depth": [5, 6, 7, 8, 9, 10, 11, 12],
        "gamma": [0, 0.1, 0.2, 0.3, 0.4],
        "subsample": [0.7, 0.75, 0.8, 1],
        "reg_lambda": [100, 150, 200, 250, 300, 350, 400],
        "n_estimators": [200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "min_child_weight": [4, 5, 6, 7, 8, 9, 10],
        "learning_rate": [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25],
    },
}

APPLICATION_FILTERS = {
    "billpayfa": lambda df: (
        ((df["message"] == 1) & (df["message_fa"] == 0) & (df["billpayfa"] == 0) & (df["debitfa"] == 0))
        | ((df["message"] == 1) & (df["message_fa"] == 1) & (df["billpayfa"] == 1) & (df["debitfa"] == 0))
    ),
    "debitfa": lambda df: (
        ((df["message"] == 1) & (df["message_fa"] == 0) & (df["billpayfa"] == 0) & (df["debitfa"] == 0))
        | ((df["message"] == 1) & (df["message_fa"] == 1) & (df["billpayfa"] == 0) & (df["debitfa"] == 1))
    ),
    "main": lambda df: (df["message"] == 1) & (df["billpayfa"] == 0) & (df["debitfa"] == 0),
}


def log(message: str) -> None:
    print(message, flush=True)


def softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def simulation_true_cate(dgp: str, frame: pd.DataFrame) -> np.ndarray:
    features = [column for column in frame if column.startswith("x")]
    x = frame[features].to_numpy(dtype=float)
    if dgp == "dgp1":
        baseline = (
            np.maximum.reduce([np.zeros(len(x)), x[:, 0] + x[:, 1], x[:, 2]])
            + 0.4 * (x[:, 3] ** 2 - 1.0)
            + 0.3 * x[:, 1] * x[:, 4]
            + 0.4 * np.maximum(0.0, x[:, 3] + x[:, 4])
            - 0.3
        )
        treatment_effect = -0.05 + 0.15 * (x[:, 0] + softplus(x[:, 1]))
    elif dgp == "dgp2":
        baseline = (
            0.8 * (x[:, 0] + x[:, 1] > 0.0)
            - 0.6 * (x[:, 2] > 0.5)
            + 0.5 * (x[:, 3] * x[:, 4] > 0.0)
            + 0.4 * np.maximum(0.0, x[:, 5] + x[:, 6])
            + 0.3 * (x[:, 7] ** 2 - 1.0)
            - 0.2 * x[:, 8] * x[:, 9]
            - 0.4
        )
        treatment_effect = (
            -0.05
            + 0.12 * x[:, 0]
            + 0.12 * softplus(x[:, 1])
            + 0.06 * softplus(x[:, 2])
        )
    else:
        raise ValueError(f"Unknown simulation design: {dgp}")
    return (
        expit(baseline + 0.5 * treatment_effect)
        - expit(baseline - 0.5 * treatment_effect)
    ).astype(np.float32)


def simulation_split(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return next(splitter.split(frame, frame[TREATMENT].to_numpy()))


def base_learners() -> dict:
    return {
        "lasso": Lasso(),
        "logistic": LogisticRegression(solver="liblinear", max_iter=500),
        "rf": RandomForestRegressor(n_jobs=1),
        "xgb": XGBRegressor(
            objective="reg:squarederror",
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        ),
    }


def select_features(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> tuple[list[str], pd.DataFrame]:
    def model() -> object:
        return make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LassoCV(cv=5, random_state=seed, max_iter=5000),
        )

    treated = model()
    control = model()
    treated.fit(x[t == 1], y[t == 1])
    control.fit(x[t == 0], y[t == 0])
    coef_t = np.abs(treated.named_steps["lassocv"].coef_.ravel())
    coef_c = np.abs(control.named_steps["lassocv"].coef_.ravel())
    top = set(pd.Series(coef_t, index=feature_names).nlargest(10).index)
    top.update(pd.Series(coef_c, index=feature_names).nlargest(10).index)
    selected = [name for name in feature_names if name in top]
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "treated_abs_coefficient": coef_t,
            "control_abs_coefficient": coef_c,
        }
    )
    return selected, coefficients


def predict_on_new(result: CATEEstimatorResults, wrapper, x_new: np.ndarray) -> np.ndarray:
    if isinstance(wrapper, XLearnerWrapper):
        propensity = np.mean(wrapper.t) * np.ones(x_new.shape[0])
        prediction = result.meta_learner.predict(x_new, p=propensity)
    else:
        prediction = result.meta_learner.predict(x_new)
    return np.asarray(prediction, dtype=float).squeeze()


def build_prediction_cache(
    libraries: dict,
    features: list[str],
    y_tv: np.ndarray,
    t_tv: np.ndarray,
    y_ho: np.ndarray,
    t_ho: np.ndarray,
    x_tv: np.ndarray,
    x_ho: np.ndarray,
    true_tv: np.ndarray | None = None,
    true_ho: np.ndarray | None = None,
    metadata: dict | None = None,
) -> dict:
    names = list(libraries["pert_none"])
    fold_cache = {}
    for perturbation, library in libraries.items():
        reference = library[names[0]]
        tau = np.empty((len(names), reference.n_splits, len(y_tv)), dtype=float)
        train = np.empty((reference.n_splits, len(y_tv)), dtype=bool)
        val = np.empty((reference.n_splits, len(y_tv)), dtype=bool)
        for fold in range(reference.n_splits):
            train[fold] = reference.results[fold].train_indicator
            val[fold] = reference.results[fold].val_indicator
        for estimator_index, name in enumerate(names):
            for fold in range(library[name].n_splits):
                tau[estimator_index, fold] = library[name].results[fold].tau
        fold_cache[perturbation] = {
            "tau": tau,
            "train_indicator": train,
            "val_indicator": val,
        }

    all_indices = np.arange(len(y_tv))
    tau_tv = []
    tau_ho = []
    for index, name in enumerate(names, start=1):
        log(f"    full fit {index}/{len(names)}: {name}")
        result = CATEEstimatorResults(
            all_indices,
            all_indices,
            libraries["pert_none"][name],
            save_metalearner=True,
        )
        tau_tv.append(np.asarray(result.tau, dtype=float))
        tau_ho.append(predict_on_new(result, libraries["pert_none"][name], x_ho))

    trainval = {"y": y_tv, "t": t_tv}
    holdout = {"y": y_ho, "t": t_ho}
    if true_tv is not None and true_ho is not None:
        trainval["true_tau"] = true_tv
        holdout["true_tau"] = true_ho
    return {
        "version": 3,
        "features": features,
        "estimator_names": names,
        "perturbation_names": list(libraries),
        "n_splits": N_SPLITS,
        "trainval": trainval,
        "holdout": holdout,
        "fold_cache": fold_cache,
        "full_fit": {"tau_tv": np.vstack(tau_tv), "tau_ho": np.vstack(tau_ho)},
        "missing_full_fit_estimators": [],
        "regeneration": metadata or {},
    }


def fit_libraries(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_iter: int,
    perturbation_seeds: tuple[int, int, int],
) -> dict:
    learners = base_learners()
    tuning_cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    log("    tuning estimator library")
    with parallel_backend("threading"):
        tuned = make_estimator_library(
            x,
            t,
            y,
            tuning_cv,
            learners,
            param_grids=PARAM_GRIDS,
            n_iter=n_iter,
            verbose=0,
        )
    tuned_params = {name: estimator.get_params() for name, estimator in tuned.items()}
    for name, params in tuned_params.items():
        if name.startswith("r_"):
            params["r_cv_n_jobs"] = 1

    libraries = {}
    for perturbation, split_seed in zip(
        ("pert_none", "pert_cv_0", "pert_cv_1"),
        perturbation_seeds,
    ):
        log(f"    fitting {perturbation}")
        cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=split_seed)
        library = make_estimator_library(
            x,
            t,
            y,
            cv,
            learners,
            tuned_params=tuned_params,
        )
        for estimator in library.values():
            np.random.seed(123123)
            estimator.fit()
        libraries[perturbation] = library
    return libraries


def atomic_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    if temporary.exists():
        raise FileExistsError(f"Incomplete prior write exists: {temporary}")
    joblib.dump(value, temporary)
    temporary.replace(path)


def simulation_source(dgp: str, seed: int) -> Path:
    return SIMULATION_DATA / dgp / f"seed_{seed}" / "analysis_df.csv"


def regenerate_simulation_seed(
    dgp: str,
    seed: int,
    output_cache: Path,
    n_iter: int,
) -> None:
    output = output_cache / "simulation" / dgp / f"seed_{seed}.pkl"
    if output.exists():
        log(f"[SKIP] {dgp}/seed_{seed}: regenerated cache already exists")
        return
    source_path = simulation_source(dgp, seed)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing simulation input for {dgp}/seed_{seed}")

    log(f"[REFIT] {dgp}/seed_{seed}")
    frame = pd.read_csv(source_path)
    if "t" in frame:
        frame = frame.rename(columns={"t": TREATMENT})
    tv_idx, ho_idx = simulation_split(frame, seed)
    tv = frame.iloc[tv_idx].reset_index(drop=True)
    ho = frame.iloc[ho_idx].reset_index(drop=True)
    true_tau = simulation_true_cate(dgp, frame)

    all_features = [column for column in frame if column.startswith("x")]
    selected, _ = select_features(
        tv[all_features].to_numpy(dtype=float),
        tv["y"].to_numpy(dtype=float),
        tv[TREATMENT].to_numpy(dtype=int),
        all_features,
        seed,
    )

    x_tv = tv[selected].to_numpy(dtype=float)
    x_ho = ho[selected].to_numpy(dtype=float)
    y_tv = tv["y"].to_numpy(dtype=float)
    t_tv = tv[TREATMENT].to_numpy(dtype=int)
    y_ho = ho["y"].to_numpy(dtype=float)
    t_ho = ho[TREATMENT].to_numpy(dtype=int)
    libraries = fit_libraries(x_tv, t_tv, y_tv, seed, n_iter, (seed, 2 * seed, 3 * seed))
    cache = build_prediction_cache(
        libraries,
        selected,
        y_tv,
        t_tv,
        y_ho,
        t_ho,
        x_tv,
        x_ho,
        true_tau[tv_idx],
        true_tau[ho_idx],
        {"kind": "simulation", "dgp": dgp, "seed": seed},
    )
    cache["tv_idx"] = tv_idx
    cache["ho_idx"] = ho_idx
    atomic_dump(cache, output)
    log(f"[SAVED] {output}")


def application_design(arm: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    raw = pd.read_csv(APPLICATION_DATA)
    data = raw.loc[APPLICATION_FILTERS[arm](raw)].copy()
    strat = [column for column in data if column.startswith("strat_")]
    categories = [
        column
        for column in ["reminder_freq", "reminder_infreq", "camp_short", "htefa", "htebal", "message"]
        if column in data
    ]
    category_values = data[categories].copy()
    missing_categories = [column for column in categories if category_values[column].isna().any()]
    category_missing = category_values[missing_categories].isna().astype(int)
    category_missing.columns = [f"{column}_missing" for column in missing_categories]
    category_values = category_values.fillna(0)

    numeric_names = [
        column
        for column in ["assets", "deposits", "paymentmean", "debt", "minbal", "creditcard"]
        if column in data
    ]
    numeric = data[numeric_names].copy()
    missing_numeric = [column for column in numeric_names if numeric[column].isna().any()]
    numeric_missing = numeric[missing_numeric].isna().astype(int)
    numeric_missing.columns = [f"{column}_missing" for column in missing_numeric]
    numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0)

    design = pd.concat(
        [data[strat], category_values, category_missing, numeric_missing, numeric],
        axis=1,
    )
    design = design.loc[:, ~design.columns.duplicated()].copy()
    full = pd.concat([design, data[["message_fa", "fausebal"]]], axis=1)
    train_indices, holdout_indices = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=405).split(
            full,
            full["message_fa"].to_numpy(),
        )
    )
    model = full.iloc[train_indices].reset_index(drop=True).rename(
        columns={"message_fa": TREATMENT}
    )
    holdout = full.iloc[holdout_indices].reset_index(drop=True).rename(
        columns={"message_fa": TREATMENT}
    )
    candidates = list(design)
    selected, coefficients = select_features(
        model[candidates].to_numpy(dtype=float),
        model["fausebal"].to_numpy(dtype=float),
        model[TREATMENT].to_numpy(dtype=int),
        candidates,
        405,
    )
    columns = selected + [TREATMENT, "fausebal"]
    return model[columns].copy(), holdout[columns].copy(), selected, coefficients


def load_descriptive_module():
    path = REFIT_ROOT / "descriptive_table.py"
    spec = importlib.util.spec_from_file_location("jbes_descriptive_table", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_diagnostics(
    cache_dir: Path,
    libraries: dict,
    model: pd.DataFrame,
    holdout: pd.DataFrame,
    features: list[str],
    coefficients: pd.DataFrame,
) -> None:
    destination = cache_dir / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(destination / "lasso_coefficients.csv", index=False)

    r2_rows = []
    fold_offset = 0
    for library in libraries.values():
        for name, estimator in library.items():
            scores = get_cr2_plot_data(estimator, 10, dir_neg=False)
            for fold, row in scores.iterrows():
                r2_rows.append(
                    {
                        "estimator": name,
                        "fold": fold_offset + int(fold),
                        "cr2_train": row["cr2_train"],
                        "cr2_val": row["cr2_val"],
                    }
                )
        fold_offset += N_SPLITS
    pd.DataFrame(r2_rows).to_csv(destination / "r2_scores.csv", index=False)

    calibration_rows = []
    for name in ("x_rf", "r_rflasso"):
        estimator = libraries["pert_none"][name]
        for kind, label in (("train", "train"), ("val", "validation")):
            values = get_calibration_plot_data(estimator, 0, 10, kind=kind)
            for bin_index, row in enumerate(values.itertuples(index=False), start=1):
                calibration_rows.append(
                    {
                        "estimator": name,
                        "split": label,
                        "bin": bin_index,
                        "observed_ate": row.Neyman_CATEs,
                        "observed_se": row.Neyman_CATEs_std,
                        "predicted_cate": row.model_CATEs,
                        "predicted_se": row.model_CATEs_std,
                    }
                )
    pd.DataFrame(calibration_rows).to_csv(destination / "calibration_bins.csv", index=False)

    import reproduce

    compact = build_prediction_cache_for_diagnostics(libraries, model, features)
    vectors = reproduce.fold_vectors(compact)
    rows = []
    for name, folds in vectors.items():
        row = {"estimator": name}
        for q in (0.1, 0.2, 0.3, 0.4, 0.5):
            values = []
            for tau, y, t, train in folds:
                metric = reproduce.top_vs_rest_metrics(y, t, tau, train, ~train, q=q)
                values.append(metric["b"])
            row[f"{1.0 - q:.1f}"] = float(np.nanmean(values))
        rows.append(row)
    pd.DataFrame(rows).to_csv(destination / "bbar_scores.csv", index=False)

    descriptive_module = load_descriptive_module()
    descriptive = descriptive_module.generate_csv_table(model, features)
    descriptive.to_csv(destination / "descriptive_lasso.csv", index=False)
    full_sample = pd.concat([model, holdout], ignore_index=True)
    descriptive_full = descriptive_module.generate_csv_table(
        full_sample,
        features + ["fausebal"],
    )
    descriptive_full.to_csv(destination / "descriptive_full_sample.csv", index=False)


def build_prediction_cache_for_diagnostics(
    libraries: dict,
    model: pd.DataFrame,
    features: list[str],
) -> dict:
    names = list(libraries["pert_none"])
    fold_cache = {}
    for perturbation, library in libraries.items():
        reference = library[names[0]]
        tau = np.empty((len(names), N_SPLITS, len(model)), dtype=float)
        train = np.empty((N_SPLITS, len(model)), dtype=bool)
        val = np.empty((N_SPLITS, len(model)), dtype=bool)
        for fold in range(N_SPLITS):
            train[fold] = reference.results[fold].train_indicator
            val[fold] = reference.results[fold].val_indicator
        for estimator_index, name in enumerate(names):
            for fold in range(N_SPLITS):
                tau[estimator_index, fold] = library[name].results[fold].tau
        fold_cache[perturbation] = {
            "tau": tau,
            "train_indicator": train,
            "val_indicator": val,
        }
    return {
        "estimator_names": names,
        "trainval": {
            "y": model["fausebal"].to_numpy(dtype=float),
            "t": model[TREATMENT].to_numpy(dtype=int),
        },
        "fold_cache": fold_cache,
    }


def regenerate_application(
    arm: str,
    output_cache: Path,
    n_iter: int,
) -> None:
    output = output_cache / "application" / f"{arm}.pkl"
    if output.exists():
        log(f"[SKIP] application/{arm}: regenerated cache already exists")
        return
    log(f"[REFIT] application/{arm}")
    model, holdout, features, coefficients = application_design(arm)
    x_tv = model[features].to_numpy(dtype=float)
    y_tv = model["fausebal"].to_numpy(dtype=float)
    t_tv = model[TREATMENT].to_numpy(dtype=int)
    x_ho = holdout[features].to_numpy(dtype=float)
    y_ho = holdout["fausebal"].to_numpy(dtype=float)
    t_ho = holdout[TREATMENT].to_numpy(dtype=int)
    libraries = fit_libraries(x_tv, t_tv, y_tv, 405, n_iter, (405, 0, 7))
    cache = build_prediction_cache(
        libraries,
        features,
        y_tv,
        t_tv,
        y_ho,
        t_ho,
        x_tv,
        x_ho,
        metadata={"kind": "application", "arm": arm},
    )
    cache["arm"] = arm
    atomic_dump(cache, output)
    if arm == "billpayfa":
        write_diagnostics(output_cache, libraries, model, holdout, features, coefficients)
    log(f"[SAVED] {output}")


def regenerate_development(output_cache: Path) -> None:
    destination = output_cache / "development" / "second_stage_detail.csv"
    if destination.exists():
        log("[SKIP] development cache already exists")
        return
    import selection_bank as bank
    import selection_metrics as base

    rows = []
    for dgp in ("dgp1", "dgp2"):
        for path in sorted((output_cache / "simulation" / dgp).glob("seed_*.pkl")):
            seed = int(path.stem.split("_")[1])
            cache = joblib.load(path)
            vectors = base.simulation_fold_vectors(cache)
            names = list(cache["estimator_names"])
            scores = base.fold_scores_from_vectors(names, vectors)
            anchor, anchor_score = bank.select_anchor(scores)
            candidates = bank.score_nested_candidates(names, vectors, scores, anchor)
            benchmark = base.evaluate_final(cache, bank.ANCHORS[dgp]).get("true_ate_ho", np.nan)
            for statistic in bank.SELECTION_STATS:
                chosen = bank.choose_candidate(candidates, statistic)
                final = base.evaluate_final(cache, bank.split_selected(chosen["selected"]))
                rows.append(
                    {
                        "dgp": dgp,
                        "seed": seed,
                        "procedure": bank.procedure_label(statistic),
                        "anchor_stat": bank.ANCHOR_STAT,
                        "anchor_score": anchor_score,
                        "add_rank_stat": bank.ADD_RANK_STAT,
                        "benchmark": "|".join(bank.ANCHORS[dgp]),
                        "benchmark_true_ate": benchmark,
                        **chosen,
                        **final,
                    }
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    log(f"[SAVED] {destination}")


def expected_simulation_paths() -> list[tuple[str, int]]:
    rows = []
    for dgp in ("dgp1", "dgp2"):
        for path in sorted((SIMULATION_DATA / dgp).glob("seed_*/analysis_df.csv")):
            rows.append((dgp, int(path.parent.name.split("_")[1])))
    return rows


def cache_is_complete(output_cache: Path) -> bool:
    simulations = expected_simulation_paths()
    return all(
        (output_cache / "simulation" / dgp / f"seed_{seed}.pkl").exists()
        for dgp, seed in simulations
    ) and all(
        (output_cache / "application" / f"{arm}.pkl").exists()
        for arm in APPLICATION_FILTERS
    )


def validate_regeneration_inputs() -> None:
    required = [
        APPLICATION_DATA,
        REFIT_ROOT / "descriptive_table.py",
        REFIT_ROOT / "selection_metrics.py",
        REFIT_ROOT / "selection_bank.py",
        REFIT_ROOT / "methods" / "cate_estimator_validation.py",
        REFIT_ROOT / "methods" / "cate_estimator_wrappers.py",
        REFIT_ROOT / "methods" / "causal_functions.py",
        REFIT_ROOT / "methods" / "data_processing.py",
    ]
    simulations = expected_simulation_paths()
    if not simulations:
        raise FileNotFoundError(f"No simulation inputs found in {SIMULATION_DATA}")
    required.extend(simulation_source(dgp, seed) for dgp, seed in simulations)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing regeneration inputs: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--scope", choices=("all", "simulation", "application", "derived"), default="all")
    parser.add_argument("--dgp", choices=("dgp1", "dgp2"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=tuple(APPLICATION_FILTERS))
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_cache = args.cache_dir.resolve()
    if output_cache == (ROOT / "cache").resolve():
        raise ValueError("Regeneration cannot write to the canonical cache")
    validate_regeneration_inputs()
    output_cache.mkdir(parents=True, exist_ok=True)
    log("[WARNING] Direct regeneration refits the estimator libraries and is time consuming")
    log(f"[OUTPUT] {output_cache}")

    try:
        if args.scope in ("all", "simulation"):
            simulations = expected_simulation_paths()
            if args.dgp is not None:
                simulations = [row for row in simulations if row[0] == args.dgp]
            if args.seed is not None:
                simulations = [row for row in simulations if row[1] == args.seed]
            for dgp, seed in simulations:
                regenerate_simulation_seed(dgp, seed, output_cache, args.n_iter)

        if args.scope in ("all", "application"):
            arms = [args.arm] if args.arm else list(APPLICATION_FILTERS)
            for arm in arms:
                regenerate_application(arm, output_cache, args.n_iter)

        if args.scope == "derived" or cache_is_complete(output_cache):
            regenerate_development(output_cache)

        if cache_is_complete(output_cache) and not args.skip_report:
            output_dir = ROOT / "outputs_regenerated"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "reproduce.py"),
                    "--cache-dir",
                    str(output_cache),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                check=False,
            )
            return result.returncode
        log("[DONE] Partial refit completed; reporting waits until the regenerated cache is complete")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
