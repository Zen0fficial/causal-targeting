#!/usr/bin/env python3
"""Reproduce the JBES manuscript results from a prediction cache."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import argparse
from pathlib import Path

RUNTIME_CACHE = Path(tempfile.gettempdir()) / "jbes-reproduction-cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))
(RUNTIME_CACHE / "fontconfig").mkdir(parents=True, exist_ok=True)

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
OUTPUT = ROOT / "outputs"
FINAL_FRACTION = 0.10
Z_90 = 1.6448536269514722
Z_975 = 1.959963984540054
ESTIMATOR_COUNT = 23
EXPECTED_SEEDS = {"dgp1": 99, "dgp2": 100}
BENCHMARKS = {"dgp1": ["s_rf"], "dgp2": ["x_rf"]}
SEED_RE = re.compile(r"seed_(\d+)\.pkl$")

EXPECTED_RESULTS = {
    "primary_simulation": {
        "dgp1": {"n_seeds": 99, "mean_true_ate_ho": 0.027743811729683116},
        "dgp2": {"n_seeds": 100, "mean_true_ate_ho": 0.07095820855464792},
    },
    "applications": {
        "billpayfa": {
            "selected": "x_rf|r_rflasso",
            "tv_ATE": 0.04069223415757067,
            "tv_SE": 0.01750400914905741,
            "tv_t": 2.324737939237306,
            "tv_n_subgroup": 2882,
            "ho_ATE": 0.06171307784211011,
            "ho_SE": 0.034617082623939366,
            "ho_t": 1.7827347992471372,
            "ho_n_subgroup": 723,
        },
        "debitfa": {
            "selected": "s_rf",
            "tv_ATE": 0.052000123235588946,
            "tv_SE": 0.0182878435801823,
            "tv_t": 2.8434256344984874,
            "tv_n_subgroup": 2883,
            "ho_ATE": -0.07237746220797064,
            "ho_SE": 0.03618508791497813,
            "ho_t": -2.0002013641097545,
            "ho_n_subgroup": 724,
        },
        "main": {
            "selected": "x_rf|x_lasso|t_lasso|x_logistic",
            "tv_ATE": 0.035139664804469284,
            "tv_SE": 0.0159556535754542,
            "tv_t": 2.2023331503340806,
            "tv_n_subgroup": 2882,
            "ho_ATE": -0.011894780968350449,
            "ho_SE": 0.03055404594053754,
            "ho_t": -0.38930297452256774,
            "ho_n_subgroup": 731,
        },
    },
    "first_stage_sensitivity": {
        "B_bar_0.9": {
            "dgp1": 0.024090246439199135,
            "dgp2": 0.06816723439228514,
        },
        "cal_slope": {
            "dgp1": 0.027725367574664494,
            "dgp2": 0.07078694664366124,
        },
    },
    "r2": {"train_mean": 0.30, "validation_mean": -0.21},
    "second_stage_ranks": {
        "weighted_b": {"dgp1": 2, "dgp2": 2},
        "mean_delta": {"dgp1": 3, "dgp2": 2},
        "mean_delta_top10": {"dgp1": 2, "dgp2": 2},
        "lcb_delta_top10": {"dgp1": 2, "dgp2": 2},
        "median_delta_top10": {"dgp1": 2, "dgp2": 2},
        "trimmed_mean_delta_top10": {"dgp1": 2, "dgp2": 2},
        "epm": {"dgp1": 3, "dgp2": 2},
        "snr": {"dgp1": 2, "dgp2": 2},
        "lcb_delta": {"dgp1": 3, "dgp2": 2},
        "top_lcb": {"dgp1": 2, "dgp2": 2},
        "rate": {"dgp1": 2, "dgp2": 2},
        "qini": {"dgp1": 2, "dgp2": 2},
        "cal_slope": {"dgp1": 2, "dgp2": 6},
        "cal_r2": {"dgp1": 3, "dgp2": 2},
        "cal_monotonicity": {"dgp1": 3, "dgp2": 2},
        "neg_cal_rmse": {"dgp1": 2, "dgp2": 2},
        "strata_ate_sd": {"dgp1": 3, "dgp2": 2},
        "top_bin_is_best": {"dgp1": 2, "dgp2": 2},
        "neg_targeting_slope": {"dgp1": 2, "dgp2": 3},
        "score_iqr_neg": {"dgp1": 2, "dgp2": 7},
        "score_sd_neg": {"dgp1": 2, "dgp2": 6},
        "score_mad_neg": {"dgp1": 2, "dgp2": 7},
        "score_p90_p10_neg": {"dgp1": 2, "dgp2": 6},
        "score_range_neg": {"dgp1": 2, "dgp2": 7},
        "top_tail_ratio": {"dgp1": 2, "dgp2": 7},
        "bbar_epm": {"dgp1": 3, "dgp2": 2},
        "bbar_delta": {"dgp1": 3, "dgp2": 2},
        "weighted_b_epm": {"dgp1": 3, "dgp2": 2},
        "positive_cal_epm": {"dgp1": 2, "dgp2": 5},
        "positive_lcb_epm": {"dgp1": 3, "dgp2": 2},
        "positive_cal_bbar_epm": {"dgp1": 2, "dgp2": 6},
        "rank_bbar_epm": {"dgp1": 3, "dgp2": 2},
        "rank_bbar_delta": {"dgp1": 3, "dgp2": 2},
        "rank_bbar_cal_epm": {"dgp1": 2, "dgp2": 2},
        "maximin_bbar_epm": {"dgp1": 3, "dgp2": 2},
        "cal_slope_x_delta": {"dgp1": 2, "dgp2": 5},
        "smooth_delta_ratio": {"dgp1": 2, "dgp2": 7},
        "smooth_lcb_ratio": {"dgp1": 2, "dgp2": 6},
        "top_balance_delta": {"dgp1": 3, "dgp2": 2},
        "stability_x_lcb": {"dgp1": 3, "dgp2": 2},
    },
    "descriptive": {
        "strat\\_5": ("6 (0.0)", "12 (0.1)"),
        "strat\\_6": ("160 (0.9)", "157 (0.9)"),
        "strat\\_8": ("427 (2.4)", "427 (2.4)"),
        "strat\\_61": ("252 (1.4)", "251 (1.4)"),
        "strat\\_62": ("787 (4.4)", "788 (4.4)"),
        "strat\\_66": ("326 (1.8)", "324 (1.8)"),
        "strat\\_70": ("102 (0.6)", "104 (0.6)"),
        "strat\\_162": ("314 (1.7)", "311 (1.7)"),
        "htefa": ("3170 (17.6)", "3094 (17.2)"),
        "htebal\\_missing": ("119 (0.7)", "110 (0.6)"),
        "assets": ("537.1 (1562.6)", "563.9 (1805.8)"),
        "debt": ("252.5 (670.3)", "251.7 (650.5)"),
        "minbal": ("93.9 (434.5)", "92.6 (441.5)"),
        "creditcard": ("0.5 (0.5)", "0.5 (0.5)"),
        "fausebal": ("5719 (31.7)", "5451 (30.3)"),
    },
}

PAPER_FIGURES = (
    "fausebal_lasso_coef.png",
    "fausebal_r2_distribution.png",
    "fausebal_calibration_plot.png",
    "fausebal_all_estimators.png",
    "mc_true_cate_top10_primary_ensemble_vs_estimators.pdf",
)
PAPER_TABLE_DATA = (
    "primary_simulation_summary.csv",
    "application_results.csv",
    "first_stage_sensitivity.csv",
    "individual_and_ensemble_ranks.csv",
    "second_stage_ranks.csv",
    "descriptive_lasso.csv",
    "lasso_coefficients.csv",
    "r2_scores.csv",
    "calibration_bins.csv",
    "bbar_scores.csv",
)

ESTIMATOR_LABELS = {
    "s_xgb": "S-XGB",
    "s_rf": "S-RF",
    "t_lasso": "T-Lasso",
    "t_logistic": "T-Logistic",
    "t_rf": "T-RF",
    "t_xgb": "T-XGB",
    "x_lasso": "X-Lasso",
    "x_logistic": "X-Logistic",
    "x_rf": "X-RF",
    "x_xgb": "X-XGB",
    "r_lassolasso": "R-Lasso/Lasso",
    "r_lassoxgb": "R-Lasso/XGB",
    "r_lassorf": "R-Lasso/RF",
    "r_rflasso": "R-RF/Lasso",
    "r_rfrf": "R-RF/RF",
    "r_rfxgb": "R-RF/XGB",
    "r_xgblasso": "R-XGB/Lasso",
    "r_xgbrf": "R-XGB/RF",
    "r_xgbxgb": "R-XGB/XGB",
    "causal_tree_1": "Causal tree (500)",
    "causal_tree_2": "Causal tree (2000)",
    "causal_forest_1": "Causal forest (500)",
    "causal_forest_2": "Causal forest (2000)",
}


def log(message: str) -> None:
    print(message, flush=True)


def neyman_ate(y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    treated = t == 1
    control = t == 0
    n1 = int(treated.sum())
    n0 = int(control.sum())
    if n1 <= 1 or n0 <= 1:
        return np.nan, np.nan
    ate = float(np.mean(y[treated]) - np.mean(y[control]))
    se = float(
        math.sqrt(np.var(y[treated], ddof=1) / n1 + np.var(y[control], ddof=1) / n0)
    )
    return ate, se


def subgroup_ate(y: np.ndarray, t: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return np.nan, np.nan
    return neyman_ate(y[mask], t[mask])


def bin_mask(
    tau: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    q_low: float,
    q_high: float,
) -> np.ndarray:
    train_tau = tau[train_mask]
    lo = np.quantile(train_tau, q_low)
    hi = np.quantile(train_tau, q_high)
    if q_high >= 1.0:
        mask = (tau >= lo) & (tau <= hi)
    else:
        mask = (tau >= lo) & (tau < hi)
    return mask & eval_mask


def top_vs_rest_metrics(
    y: np.ndarray,
    t: np.ndarray,
    tau: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    q: float = FINAL_FRACTION,
) -> dict[str, float]:
    threshold = np.quantile(tau[train_mask], 1.0 - q)
    top = (tau >= threshold) & val_mask
    rest = (~top) & val_mask
    top_ate, top_se = subgroup_ate(y, t, top)
    rest_ate, rest_se = subgroup_ate(y, t, rest)
    if not np.isfinite(top_ate) or not np.isfinite(rest_ate):
        return {"delta": np.nan, "t_stat": np.nan, "b": np.nan}
    delta = top_ate - rest_ate
    se = (
        math.sqrt(top_se**2 + rest_se**2)
        if np.isfinite(top_se) and np.isfinite(rest_se)
        else np.nan
    )
    return {
        "delta": float(delta),
        "t_stat": float(delta / se) if np.isfinite(se) and se > 0 else np.nan,
        "b": float(delta > 0),
    }


def calibration_slope(
    y: np.ndarray,
    t: np.ndarray,
    tau: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> float:
    tau_means = []
    ates = []
    edges = np.linspace(0.0, 1.0, 6)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = bin_mask(tau, train_mask, val_mask, low, high)
        ate, _ = subgroup_ate(y, t, mask)
        if np.any(mask) and np.isfinite(ate):
            tau_means.append(float(np.mean(tau[mask])))
            ates.append(float(ate))
    if len(ates) < 2 or np.std(tau_means) <= 1e-12:
        return np.nan
    x = np.asarray(tau_means)
    z = np.asarray(ates)
    return float(np.cov(x, z, ddof=1)[0, 1] / np.var(x, ddof=1))


def fold_vectors(cache: dict) -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    names = [str(name) for name in cache["estimator_names"]]
    y = np.asarray(cache["trainval"]["y"], dtype=float)
    t = np.asarray(cache["trainval"]["t"], dtype=int)
    out = {name: [] for name in names}
    for fold_data in cache["fold_cache"].values():
        tau = np.asarray(fold_data["tau"], dtype=float)
        train = np.asarray(fold_data["train_indicator"], dtype=bool)
        if tau.shape[:2] != (len(names), train.shape[0]):
            raise ValueError("Fold prediction cache has inconsistent dimensions")
        for estimator_index, name in enumerate(names):
            for fold_index in range(train.shape[0]):
                out[name].append((tau[estimator_index, fold_index], y, t, train[fold_index]))
    return out


def validate_prediction_cache(cache: dict, label: str) -> dict[str, int | str]:
    required = {"estimator_names", "trainval", "holdout", "fold_cache", "full_fit"}
    if not required.issubset(cache):
        raise ValueError(f"{label}: missing cache keys {sorted(required - set(cache))}")
    names = [str(name) for name in cache["estimator_names"]]
    if len(names) != ESTIMATOR_COUNT or len(set(names)) != ESTIMATOR_COUNT:
        raise ValueError(f"{label}: expected 23 distinct estimators")
    n_tv = len(cache["trainval"]["y"])
    n_ho = len(cache["holdout"]["y"])
    for split, n in [("trainval", n_tv), ("holdout", n_ho)]:
        for key in ("y", "t"):
            if len(cache[split][key]) != n:
                raise ValueError(f"{label}: inconsistent {split}/{key}")
    tau_tv = np.asarray(cache["full_fit"]["tau_tv"])
    tau_ho = np.asarray(cache["full_fit"]["tau_ho"])
    if tau_tv.shape != (ESTIMATOR_COUNT, n_tv) or tau_ho.shape != (ESTIMATOR_COUNT, n_ho):
        raise ValueError(f"{label}: inconsistent full-fit prediction dimensions")
    missing_full_fit = [str(name) for name in cache.get("missing_full_fit_estimators", [])]
    if missing_full_fit:
        missing_indices = [names.index(name) for name in missing_full_fit]
        available_indices = [index for index in range(len(names)) if index not in missing_indices]
        if not np.isnan(tau_tv[missing_indices]).all() or not np.isnan(tau_ho[missing_indices]).all():
            raise ValueError(f"{label}: unavailable full-fit rows must be explicit NaN arrays")
        if not np.isfinite(tau_tv[available_indices]).all() or not np.isfinite(tau_ho[available_indices]).all():
            raise ValueError(f"{label}: non-finite available full-fit predictions")
    elif not np.isfinite(tau_tv).all() or not np.isfinite(tau_ho).all():
        raise ValueError(f"{label}: non-finite full-fit predictions")
    vectors = fold_vectors(cache)
    fold_counts = {len(value) for value in vectors.values()}
    if fold_counts != {12}:
        raise ValueError(f"{label}: expected 12 fold predictions per estimator, got {fold_counts}")
    return {
        "label": label,
        "n_tv": n_tv,
        "n_ho": n_ho,
        "n_estimators": len(names),
        "n_folds": 12,
        "n_missing_full_fit": len(missing_full_fit),
    }


def first_stage_scores(vectors: dict) -> pd.DataFrame:
    rows = []
    for estimator, items in vectors.items():
        b_values = []
        deltas = []
        t_values = []
        slopes = []
        for tau, y, t, train in items:
            val = ~train
            metrics = top_vs_rest_metrics(y, t, tau, train, val)
            slope = calibration_slope(y, t, tau, train, val)
            if np.isfinite(metrics["b"]):
                b_values.append(metrics["b"])
            if np.isfinite(metrics["delta"]):
                deltas.append(metrics["delta"])
            if np.isfinite(metrics["t_stat"]):
                t_values.append(metrics["t_stat"])
            if np.isfinite(slope):
                slopes.append(slope)
        expected = len(items)
        rows.append(
            {
                "estimator": estimator,
                "B_bar_0.9": float(np.mean(b_values)) if b_values else np.nan,
                "delta_bar_0.9": float(np.mean(deltas)) if deltas else np.nan,
                "t_bar_0.9": float(np.mean(t_values)) if t_values else np.nan,
                "cal_slope": float(np.mean(slopes)) if slopes else np.nan,
                "expected_folds": expected,
                "n_top10_folds": len(deltas),
                "n_t_folds": len(t_values),
                "n_cal_slope_folds": len(slopes),
                "complete_support": (
                    expected > 0
                    and len(deltas) == expected
                    and len(t_values) == expected
                    and len(slopes) == expected
                ),
            }
        )
    return pd.DataFrame(rows)


def ordered_estimators(scores: pd.DataFrame, statistic: str, complete_only: bool) -> list[str]:
    pool = scores[scores["complete_support"]].copy() if complete_only else scores.copy()
    pool = pool.replace([np.inf, -np.inf], np.nan).dropna(subset=[statistic])
    if statistic == "B_bar_0.9":
        columns = ["B_bar_0.9", "delta_bar_0.9", "t_bar_0.9", "estimator"]
        ascending = [False, False, False, True]
    else:
        columns = ["cal_slope", "estimator"]
        ascending = [False, True]
    return (
        pool.sort_values(columns, ascending=ascending, kind="mergesort")["estimator"]
        .astype(str)
        .tolist()
    )


def choose_nested_ensemble(
    order: list[str],
    vectors: dict,
    require_complete_folds: bool,
) -> tuple[dict, pd.DataFrame]:
    if not order:
        return {}, pd.DataFrame()
    metrics = [{"deltas": [], "b": [], "t": []} for _ in order]
    first = vectors[order[0]]
    for fold_index in range(len(first)):
        tau_sum = None
        for k, name in enumerate(order):
            tau, y, t, train = vectors[name][fold_index]
            tau_sum = tau.astype(float, copy=True) if tau_sum is None else tau_sum + tau
            result = top_vs_rest_metrics(y, t, tau_sum / (k + 1), train, ~train)
            if np.isfinite(result["delta"]):
                metrics[k]["deltas"].append(result["delta"])
            if np.isfinite(result["b"]):
                metrics[k]["b"].append(result["b"])
            if np.isfinite(result["t_stat"]):
                metrics[k]["t"].append(result["t_stat"])
    rows = []
    for k, values in enumerate(metrics, start=1):
        rows.append(
            {
                "selected": "|".join(order[:k]),
                "n_estimators": k,
                "mean_delta_top10": float(np.mean(values["deltas"])) if values["deltas"] else np.nan,
                "ensemble_B_bar_0.9": float(np.mean(values["b"])) if values["b"] else np.nan,
                "ensemble_t_bar_0.9": float(np.mean(values["t"])) if values["t"] else np.nan,
                "n_folds": len(values["deltas"]),
            }
        )
    candidates = pd.DataFrame(rows)
    pool = candidates.dropna(subset=["mean_delta_top10"])
    if require_complete_folds:
        pool = pool[pool["n_folds"].eq(len(first))]
    if pool.empty:
        return {}, candidates
    chosen = pool.sort_values(
        ["mean_delta_top10", "n_estimators", "selected"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    return chosen.to_dict(), candidates


def stack_predictions(cache: dict, names: list[str], split: str) -> np.ndarray:
    estimator_names = [str(name) for name in cache["estimator_names"]]
    indices = [estimator_names.index(name) for name in names]
    key = "tau_tv" if split == "trainval" else "tau_ho"
    selected = np.asarray(cache["full_fit"][key], dtype=float)[indices]
    if not np.isfinite(selected).all():
        missing = [name for name, row in zip(names, selected) if not np.isfinite(row).all()]
        raise ValueError(f"Full-fit predictions are unavailable for selected estimators: {missing}")
    return np.mean(selected, axis=0)


def evaluate_final(cache: dict, names: list[str]) -> dict[str, float | int]:
    tau_tv = stack_predictions(cache, names, "trainval")
    tau_ho = stack_predictions(cache, names, "holdout")
    threshold = np.quantile(tau_tv, 1.0 - FINAL_FRACTION)
    mask_tv = tau_tv >= threshold
    mask_ho = tau_ho >= threshold
    y_tv = np.asarray(cache["trainval"]["y"], dtype=float)
    t_tv = np.asarray(cache["trainval"]["t"], dtype=int)
    y_ho = np.asarray(cache["holdout"]["y"], dtype=float)
    t_ho = np.asarray(cache["holdout"]["t"], dtype=int)
    tv_ate, tv_se = subgroup_ate(y_tv, t_tv, mask_tv)
    ho_ate, ho_se = subgroup_ate(y_ho, t_ho, mask_ho)
    result: dict[str, float | int] = {
        "tv_ATE": tv_ate,
        "tv_SE": tv_se,
        "tv_t": tv_ate / tv_se if np.isfinite(tv_se) and tv_se > 0 else np.nan,
        "tv_n_subgroup": int(mask_tv.sum()),
        "ho_ATE": ho_ate,
        "ho_SE": ho_se,
        "ho_t": ho_ate / ho_se if np.isfinite(ho_se) and ho_se > 0 else np.nan,
        "ho_CI90_lo": ho_ate - Z_90 * ho_se,
        "ho_CI90_hi": ho_ate + Z_90 * ho_se,
        "ho_CI95_lo": ho_ate - Z_975 * ho_se,
        "ho_CI95_hi": ho_ate + Z_975 * ho_se,
        "ho_n_subgroup": int(mask_ho.sum()),
    }
    if "true_tau" in cache["holdout"]:
        true_tv = np.asarray(cache["trainval"]["true_tau"], dtype=float)
        true_ho = np.asarray(cache["holdout"]["true_tau"], dtype=float)
        oracle_threshold = np.quantile(true_tv, 1.0 - FINAL_FRACTION)
        oracle = float(np.mean(true_tv[true_tv >= oracle_threshold]))
        true_selected = float(np.mean(true_ho[mask_ho]))
        result.update(
            {
                "oracle_ate": oracle,
                "true_ate_ho": true_selected,
                "regret": oracle - true_selected,
                "relative_regret": (oracle - true_selected) / oracle,
            }
        )
    return result


def discover_simulation_caches(dgp: str) -> list[tuple[int, Path]]:
    paths = []
    for path in sorted((CACHE / "simulation" / dgp).glob("seed_*.pkl")):
        match = SEED_RE.fullmatch(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    if len(paths) != EXPECTED_SEEDS[dgp]:
        raise ValueError(f"{dgp}: expected {EXPECTED_SEEDS[dgp]} caches, found {len(paths)}")
    return paths


def reproduce_primary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    sensitivity_rows = []
    individual_rows = []
    schema_rows = []
    for dgp in ("dgp1", "dgp2"):
        caches = discover_simulation_caches(dgp)
        log(f"[CACHE] {dgp.upper()} simulation")
        for index, (seed, path) in enumerate(caches, start=1):
            cache = joblib.load(path)
            schema_rows.append(validate_prediction_cache(cache, f"{dgp}/seed_{seed}"))
            vectors = fold_vectors(cache)
            scores = first_stage_scores(vectors)

            primary_order = ordered_estimators(scores, "cal_slope", complete_only=False)
            primary, _ = choose_nested_ensemble(
                primary_order,
                vectors,
                require_complete_folds=False,
            )
            selected = str(primary["selected"]).split("|")
            detail_rows.append(
                {"dgp": dgp, "seed": seed, "first_stage_score": "cal_slope", **primary, **evaluate_final(cache, selected)}
            )

            for statistic in ("cal_slope", "B_bar_0.9"):
                order = ordered_estimators(scores, statistic, complete_only=True)
                chosen, _ = choose_nested_ensemble(
                    order,
                    vectors,
                    require_complete_folds=True,
                )
                sensitivity_rows.append(
                    {"dgp": dgp, "seed": seed, "first_stage_score": statistic, **chosen, **evaluate_final(cache, str(chosen["selected"]).split("|"))}
                )

            names = [str(name) for name in cache["estimator_names"]]
            for name in names:
                individual_rows.append(
                    {"dgp": dgp, "seed": seed, "method": name, "true_ate_ho": evaluate_final(cache, [name])["true_ate_ho"]}
                )

    detail = pd.DataFrame(detail_rows)
    primary_summary = (
        detail.groupby("dgp")
        .agg(
            n_seeds=("seed", "nunique"),
            mean_true_ate_ho=("true_ate_ho", "mean"),
            se_true_ate_ho=("true_ate_ho", lambda s: float(s.std(ddof=1) / math.sqrt(len(s)))),
            mean_n_estimators=("n_estimators", "mean"),
        )
        .reset_index()
    )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_summary = (
        sensitivity.groupby(["first_stage_score", "dgp"])
        .agg(
            n_seeds=("seed", "nunique"),
            mean_true_ate_ho=("true_ate_ho", "mean"),
            se_true_ate_ho=("true_ate_ho", lambda s: float(s.std(ddof=1) / math.sqrt(len(s)))),
            mean_n_estimators=("n_estimators", "mean"),
        )
        .reset_index()
    )
    individual = pd.DataFrame(individual_rows)
    return detail, primary_summary, sensitivity_summary, individual, pd.DataFrame(schema_rows)


def reproduce_applications() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    schemas = []
    for arm in ("billpayfa", "debitfa", "main"):
        path = CACHE / "application" / f"{arm}.pkl"
        cache = joblib.load(path)
        log(f"[CACHE] application/{arm}")
        schemas.append(validate_prediction_cache(cache, f"application/{arm}"))
        vectors = fold_vectors(cache)
        scores = first_stage_scores(vectors)
        order = ordered_estimators(scores, "cal_slope", complete_only=False)
        chosen, _ = choose_nested_ensemble(
            order,
            vectors,
            require_complete_folds=False,
        )
        rows.append(
            {
                "arm": arm,
                "first_stage_score": "cal_slope",
                "anchor_model": order[0],
                "anchor_score": float(scores.set_index("estimator").loc[order[0], "cal_slope"]),
                **chosen,
                **evaluate_final(cache, str(chosen["selected"]).split("|")),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(schemas)


def individual_summary(individual: pd.DataFrame, primary_detail: pd.DataFrame) -> pd.DataFrame:
    ind = (
        individual.groupby(["dgp", "method"])
        .agg(n=("seed", "nunique"), mean_true_cate=("true_ate_ho", "mean"), sd_true_cate=("true_ate_ho", "std"))
        .reset_index()
    )
    selected = (
        primary_detail.groupby("dgp")
        .agg(n=("seed", "nunique"), mean_true_cate=("true_ate_ho", "mean"), sd_true_cate=("true_ate_ho", "std"))
        .reset_index()
    )
    selected["method"] = "selected_ensemble"
    out = pd.concat([ind, selected], ignore_index=True)
    out["se_true_cate"] = out["sd_true_cate"] / np.sqrt(out["n"])
    out["ci95"] = 1.96 * out["se_true_cate"]
    out["rank"] = out.groupby("dgp")["mean_true_cate"].rank(ascending=False, method="min").astype(int)
    return out


def plot_primary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.2))
    for ax, dgp in zip(axes, ("dgp1", "dgp2")):
        data = summary[summary["dgp"].eq(dgp)].copy()
        data["label"] = data["method"].map(ESTIMATOR_LABELS).fillna("Selected ensemble")
        data = data.sort_values(["mean_true_cate", "label"])
        pos = np.arange(len(data))
        selected = data["method"].eq("selected_ensemble")
        colors = np.where(selected, "#b51f2e", "#376996")
        ax.barh(pos, data["mean_true_cate"], xerr=data["ci95"], color=colors, alpha=0.9, capsize=2)
        ax.set_yticks(pos, data["label"], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(dgp.upper())
        ax.set_xlabel("Mean true CATE in selected holdout top 10%")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Selected nested ensemble versus individual CATE estimators", fontweight="bold")
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "figs" / "mc_true_cate_top10_primary_ensemble_vs_estimators.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def reproduce_diagnostics() -> dict:
    diagnostic_dir = CACHE / "diagnostics"
    figure_dir = OUTPUT / "figs"

    lasso = pd.read_csv(diagnostic_dir / "lasso_coefficients.csv")
    selected = set(lasso.nlargest(10, "treated_abs_coefficient")["feature"]) | set(
        lasso.nlargest(10, "control_abs_coefficient")["feature"]
    )
    selected_ordered = [name for name in lasso["feature"] if name in selected]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, column, title in [
        (axes[0], "treated_abs_coefficient", "Treated"),
        (axes[1], "control_abs_coefficient", "Control"),
    ]:
        data = lasso.nlargest(10, column).sort_values(column)
        ax.barh(data["feature"], data[column], color="#777777")
        ax.set_title(title)
        ax.set_xlabel("Absolute standardized Lasso coefficient")
    fig.tight_layout()
    fig.savefig(figure_dir / "fausebal_lasso_coef.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    lasso.to_csv(OUTPUT / "tables" / "lasso_coefficients.csv", index=False)

    r2 = pd.read_csv(diagnostic_dir / "r2_scores.csv")
    r2_stats = {
        "train_mean": float(r2["cr2_train"].mean()),
        "train_median": float(r2["cr2_train"].median()),
        "validation_mean": float(r2["cr2_val"].mean()),
        "validation_median": float(r2["cr2_val"].median()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    finite = pd.concat([r2["cr2_train"], r2["cr2_val"]]).replace([np.inf, -np.inf], np.nan).dropna()
    bins = np.linspace(float(finite.min()), float(finite.max()), 14)
    for ax, column, title in zip(
        axes,
        ("cr2_train", "cr2_val"),
        ("Training folds", "Validation folds"),
    ):
        values = r2[column].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(values, bins=bins, facecolor="white", edgecolor="black")
        ax.axvline(0, color="black", linestyle="-.", linewidth=1.0, label="Zero-reference")
        ax.axvline(values.mean(), color="black", linestyle="--", linewidth=1.0, label="Mean")
        ax.axvline(values.median(), color="#555555", linestyle="-", linewidth=1.0, label="Median")
        ax.set_title(title)
        ax.set_xlabel("Calibration R-squared")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Distribution of calibration R-squared scores")
    fig.tight_layout()
    fig.savefig(figure_dir / "fausebal_r2_distribution.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    r2.to_csv(OUTPUT / "tables" / "r2_scores.csv", index=False)

    calibration = pd.read_csv(diagnostic_dir / "calibration_bins.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for row_index, estimator in enumerate(("x_rf", "r_rflasso")):
        for column_index, split in enumerate(("train", "validation")):
            ax = axes[row_index, column_index]
            data = calibration[(calibration["estimator"] == estimator) & (calibration["split"] == split)]
            ax.errorbar(
                data["bin"],
                data["predicted_cate"],
                yerr=data["predicted_se"],
                fmt="^-",
                color="black",
                linewidth=1.2,
                capsize=2,
                label="Predicted CATE",
            )
            ax.errorbar(
                data["bin"],
                data["observed_ate"],
                yerr=data["observed_se"],
                fmt="o--",
                color="#999999",
                linewidth=1.2,
                capsize=2,
                label="Difference in means",
            )
            ax.axhline(-0.0158, color="black", linestyle="--", linewidth=0.8)
            if row_index == 0:
                ax.set_title("Training folds" if split == "train" else "Validation folds")
            ax.text(0.5, 0.10, estimator, transform=ax.transAxes, ha="center", fontsize=11)
            if row_index == 1:
                ax.set_xlabel("Predicted-CATE decile")
            ax.set_ylabel("Treatment effect")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "fausebal_calibration_plot.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    calibration.to_csv(OUTPUT / "tables" / "calibration_bins.csv", index=False)

    bbar = pd.read_csv(diagnostic_dir / "bbar_scores.csv")
    columns = [column for column in bbar if column != "estimator"]
    grayscale_palette = [
        "#1a1a1a",
        "#2f2f2f",
        "#454545",
        "#5c5c5c",
        "#727272",
        "#898989",
        "#a0a0a0",
        "#b6b6b6",
        "#cdcdcd",
        "#e3e3e3",
    ]
    figure4_rc = {
        "lines.markersize": 10,
        "grid.linewidth": 2.5,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "savefig.transparent": True,
        "figure.facecolor": "white",
        "figure.edgecolor": "black",
        "axes.facecolor": "white",
        "axes.prop_cycle": cycler(color=grayscale_palette),
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "figure.titlesize": 22,
    }
    with plt.style.context("fivethirtyeight"), plt.rc_context(figure4_rc):
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(
            data=bbar.loc[:, columns],
            ax=ax,
            fliersize=10,
            linewidth=2,
            width=0.6,
            color="white",
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 2},
            whiskerprops={"color": "black", "linewidth": 2},
            capprops={"color": "black", "linewidth": 2},
            medianprops={"color": "black", "linewidth": 2},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 6,
            },
        )
        lines = ax.get_lines()
        for category in ax.get_xticks():
            median = round(lines[4 + int(category) * 6].get_ydata()[0], 2)
            ax.text(
                category,
                median,
                f"{median}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                size=30,
                bbox={"facecolor": "#4d4d4d", "edgecolor": "none", "pad": 3},
            )
        labels = [rf"$\overline{{B}}_{{{column}}}$" for column in columns]
        ax.set_ylim(0, 1.1)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=25)
        ax.grid(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylabel("Value", fontsize=26)
        fig.tight_layout()
        fig.savefig(
            figure_dir / "fausebal_all_estimators.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)
    bbar.to_csv(OUTPUT / "tables" / "bbar_scores.csv", index=False)

    descriptive_path = diagnostic_dir / "descriptive_full_sample.csv"
    if not descriptive_path.exists():
        raise FileNotFoundError(
            f"Missing full-sample descriptive cache: {descriptive_path}. "
            "Regenerate the derived diagnostic cache."
        )
    descriptive = pd.read_csv(descriptive_path)
    descriptive.to_csv(OUTPUT / "tables" / "descriptive_lasso.csv", index=False)
    descriptive_values = {
        str(row.Abbreviation): {
            "Control": str(row.Control).replace(")%", ")"),
            "Treatment": str(row.Treatment).replace(")%", ")"),
        }
        for row in descriptive.itertuples(index=False)
        if isinstance(row.Abbreviation, str) and row.Abbreviation
    }
    return {
        "selected_feature_count": len(selected_ordered),
        "selected_features": selected_ordered,
        "r2": r2_stats,
        "descriptive_rows": len(descriptive),
        "descriptive_values": descriptive_values,
    }


def reproduce_development_bank(individual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.read_csv(CACHE / "development" / "second_stage_detail.csv")
    summary = (
        detail.groupby(["selection_stat", "dgp"])
        .agg(n_seeds=("seed", "nunique"), mean_true_ate_ho=("true_ate_ho", "mean"))
        .reset_index()
    )
    individual_means = individual.groupby(["dgp", "method"])["true_ate_ho"].mean().reset_index()
    rank_rows = []
    for row in summary.itertuples(index=False):
        comparison = individual_means[individual_means["dgp"].eq(row.dgp)]["true_ate_ho"]
        rank = 1 + int(np.sum(comparison > row.mean_true_ate_ho))
        rank_rows.append({"selection_stat": row.selection_stat, "dgp": row.dgp, "rank": rank, "mean_true_ate_ho": row.mean_true_ate_ho})
    ranks = pd.DataFrame(rank_rows)
    return summary, ranks


def compare_results(
    primary: pd.DataFrame,
    applications: pd.DataFrame,
    sensitivity: pd.DataFrame,
    diagnostics: dict,
    bank_ranks: pd.DataFrame,
    schemas: pd.DataFrame,
) -> dict:
    expected = EXPECTED_RESULTS
    checks = []

    def add(name: str, actual, target, tolerance: float = 0.0, category: str = "recomputed") -> None:
        if isinstance(target, str):
            passed = str(actual) == target
        elif isinstance(target, list):
            passed = list(actual) == target
        else:
            passed = bool(abs(float(actual) - float(target)) <= tolerance)
        checks.append(
            {"result": name, "source": category, "reproduced": actual, "reported": target, "tolerance": tolerance, "match": passed}
        )

    for dgp, values in expected["primary_simulation"].items():
        row = primary[primary["dgp"].eq(dgp)].iloc[0]
        add(f"{dgp}.n_seeds", int(row.n_seeds), values["n_seeds"])
        add(f"{dgp}.mean_true_ate_ho", row.mean_true_ate_ho, values["mean_true_ate_ho"], 5e-13)

    for arm, values in expected["applications"].items():
        row = applications[applications["arm"].eq(arm)].iloc[0]
        add(f"{arm}.selected", row.selected, values["selected"])
        for key in ("tv_ATE", "tv_SE", "tv_t", "tv_n_subgroup", "ho_ATE", "ho_SE", "ho_t", "ho_n_subgroup"):
            tolerance = 0.0 if key.endswith("subgroup") else 5e-12
            add(f"{arm}.{key}", row[key], values[key], tolerance)

    for statistic, dgp_values in expected["first_stage_sensitivity"].items():
        for dgp, target in dgp_values.items():
            row = sensitivity[(sensitivity["first_stage_score"].eq(statistic)) & (sensitivity["dgp"].eq(dgp))].iloc[0]
            add(f"sensitivity.{statistic}.{dgp}", row.mean_true_ate_ho, target, 5e-13)

    add("diagnostics.selected_feature_count", diagnostics["selected_feature_count"], 14)
    for key, target in expected["r2"].items():
        add(f"r2.{key}", round(diagnostics["r2"][key], 2), target, 5e-12)

    for abbreviation, (control, treatment) in expected["descriptive"].items():
        values = diagnostics["descriptive_values"].get(abbreviation, {})
        add(f"descriptive.{abbreviation}.control", values.get("Control"), control)
        add(f"descriptive.{abbreviation}.treatment", values.get("Treatment"), treatment)

    for statistic, dgp_values in expected["second_stage_ranks"].items():
        for dgp, target in dgp_values.items():
            row = bank_ranks[(bank_ranks["selection_stat"].eq(statistic)) & (bank_ranks["dgp"].eq(dgp))].iloc[0]
            add(f"second_stage.{statistic}.{dgp}", int(row["rank"]), target, category="cache-verified")

    return checks


def verify_paper_outputs() -> None:
    paths = [OUTPUT / "figs" / name for name in PAPER_FIGURES]
    paths.extend(OUTPUT / "tables" / name for name in PAPER_TABLE_DATA)
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing paper reproduction artifacts: {missing}")


def main() -> int:
    global CACHE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cached",),
        default="cached",
        help="run from an existing prediction cache",
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    CACHE = args.cache_dir.resolve()
    OUTPUT = args.output_dir.resolve()
    for directory in (OUTPUT, OUTPUT / "figs", OUTPUT / "tables"):
        directory.mkdir(parents=True, exist_ok=True)

    log(f"[RUN] direct rerun from cache: {CACHE}")
    primary_detail, primary, sensitivity, individual, sim_schemas = reproduce_primary()
    applications, app_schemas = reproduce_applications()
    schemas = pd.concat([sim_schemas, app_schemas], ignore_index=True)
    methods = individual_summary(individual, primary_detail)
    plot_primary(methods)
    diagnostics = reproduce_diagnostics()
    bank_summary, bank_ranks = reproduce_development_bank(individual)

    comparison = compare_results(primary, applications, sensitivity, diagnostics, bank_ranks, schemas)
    primary.to_csv(OUTPUT / "tables" / "primary_simulation_summary.csv", index=False)
    applications.to_csv(OUTPUT / "tables" / "application_results.csv", index=False)
    sensitivity.to_csv(OUTPUT / "tables" / "first_stage_sensitivity.csv", index=False)
    methods.to_csv(OUTPUT / "tables" / "individual_and_ensemble_ranks.csv", index=False)
    bank_summary.to_csv(OUTPUT / "tables" / "second_stage_summary.csv", index=False)
    bank_ranks.to_csv(OUTPUT / "tables" / "second_stage_ranks.csv", index=False)
    (OUTPUT / "tables" / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    verify_paper_outputs()
    comparison = pd.DataFrame(comparison)
    if comparison["match"].all():
        log("[DONE] Reproduced results match the reported results")
        return 0
    log("[ERROR] Some reproduced results differ from the reported results")
    return 1


if __name__ == "__main__":
    sys.exit(main())
