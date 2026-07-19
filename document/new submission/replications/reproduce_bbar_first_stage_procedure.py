#!/usr/bin/env python3
"""Compare first-stage scores on a common complete-fold estimator set.

The experiment compares two first-stage anchor and path orderings:

1. Retain estimators with finite top-decile and calibration-slope values on all
   12 folds.
2. Rank the common set by calibration slope or validation B_bar_0.9.
3. For B_bar_0.9, break ties by delta_bar_0.9, mean fold t, and estimator name.
3. Form the nested equal-weight path along that order.
4. Select the path member with the largest mean_delta_top10.

All calculations use saved fold predictions and saved full-fit predictions. No
CATE estimator is fitted by this script.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import evaluate_proposed_targeting_statistics as base
import screen_anchor_augmented_procedure as real_helpers


OUTPUT_DIR = base.ROOT / "output" / "bbar_first_stage_reproduction"
REAL_CACHE_DIR = base.ROOT / "output" / "realdata_replication" / "cache"
SELECTION_STAT = "mean_delta_top10"
FIRST_STAGE_SCORES = ["cal_slope", "B_bar_0.9"]

BENCHMARKS = {
    "dgp1": ["s_rf"],
    "dgp2": ["x_rf"],
}


def split_selected(selected: str) -> list[str]:
    return selected.split("|") if isinstance(selected, str) and selected else []


def first_stage_scores(
    estimator_names: list[str],
    fold_vectors: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]],
) -> pd.DataFrame:
    """Compute first-stage statistics and their fold support."""
    rows = []
    for estimator in estimator_names:
        b_values = []
        deltas = []
        t_values = []
        cal_slopes = []
        expected_folds = len(fold_vectors.get(estimator, []))
        for tau, y, t, train_mask in fold_vectors.get(estimator, []):
            if train_mask is None:
                train_mask = np.ones_like(tau, dtype=bool)
                val_mask = np.ones_like(tau, dtype=bool)
            else:
                val_mask = ~train_mask
            metrics = base.top_vs_rest_metrics(
                y,
                t,
                tau,
                train_mask,
                val_mask,
                base.FINAL_FRACTION,
            )
            if np.isfinite(metrics["b"]):
                b_values.append(float(metrics["b"]))
            if np.isfinite(metrics["delta"]):
                deltas.append(float(metrics["delta"]))
            if np.isfinite(metrics["t_stat"]):
                t_values.append(float(metrics["t_stat"]))
            calibration = base.calibration_metrics(y, t, tau, train_mask, val_mask)
            if np.isfinite(calibration["cal_slope"]):
                cal_slopes.append(float(calibration["cal_slope"]))
        complete_support = (
            expected_folds > 0
            and len(b_values) == expected_folds
            and len(deltas) == expected_folds
            and len(t_values) == expected_folds
            and len(cal_slopes) == expected_folds
        )
        rows.append(
            {
                "estimator": estimator,
                "B_bar_0.9": float(np.mean(b_values)) if b_values else np.nan,
                "delta_bar_0.9": float(np.mean(deltas)) if deltas else np.nan,
                "t_bar_0.9": float(np.mean(t_values)) if t_values else np.nan,
                "cal_slope": float(np.mean(cal_slopes)) if cal_slopes else np.nan,
                "expected_folds": expected_folds,
                "n_top10_folds": len(b_values),
                "n_t_folds": len(t_values),
                "n_cal_slope_folds": len(cal_slopes),
                "complete_support": complete_support,
                "n_zero_deltas": int(np.sum(np.asarray(deltas) == 0.0)),
            }
        )
    return pd.DataFrame(rows)


def ordered_estimators(scores: pd.DataFrame, statistic: str) -> list[str]:
    """Order the common complete-support estimator set deterministically."""
    pool = scores[scores["complete_support"]].copy()
    if statistic == "B_bar_0.9":
        columns = ["B_bar_0.9", "delta_bar_0.9", "t_bar_0.9", "estimator"]
        ascending = [False, False, False, True]
    elif statistic == "cal_slope":
        columns = ["cal_slope", "estimator"]
        ascending = [False, True]
    else:
        raise ValueError(f"Unsupported first-stage score: {statistic}")
    ordered = pool.sort_values(
        columns,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    )
    return ordered["estimator"].astype(str).tolist()


def nested_path(order: list[str]) -> list[tuple[str, ...]]:
    return [tuple(order[:k]) for k in range(1, len(order) + 1)]


def choose_nested_ensemble(
    order: list[str],
    fold_vectors: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]],
) -> tuple[dict, pd.DataFrame]:
    metrics_by_k = [
        {"deltas": [], "b_values": [], "t_values": []} for _ in order
    ]
    first = fold_vectors.get(order[0], [])
    for fold_idx in range(len(first)):
        tau_sum = None
        y = t = train_mask = None
        for k, name in enumerate(order):
            current = fold_vectors.get(name, [])
            if fold_idx >= len(current):
                break
            tau, y, t, train_mask = current[fold_idx]
            tau_sum = tau.astype(float, copy=True) if tau_sum is None else tau_sum + tau
            tau_mean = tau_sum / float(k + 1)
            metrics = base.top_vs_rest_metrics(
                y,
                t,
                tau_mean,
                train_mask,
                ~train_mask,
                base.FINAL_FRACTION,
            )
            if np.isfinite(metrics["delta"]):
                metrics_by_k[k]["deltas"].append(float(metrics["delta"]))
            if np.isfinite(metrics["b"]):
                metrics_by_k[k]["b_values"].append(float(metrics["b"]))
            if np.isfinite(metrics["t_stat"]):
                metrics_by_k[k]["t_values"].append(float(metrics["t_stat"]))

    rows = []
    for k, names in enumerate(nested_path(order)):
        values = metrics_by_k[k]
        rows.append(
            {
                "selected": "|".join(names),
                "n_estimators": len(names),
                "mean_delta_top10": (
                    float(np.mean(values["deltas"]))
                    if values["deltas"]
                    else np.nan
                ),
                "ensemble_B_bar_0.9": (
                    float(np.mean(values["b_values"]))
                    if values["b_values"]
                    else np.nan
                ),
                "ensemble_t_bar_0.9": (
                    float(np.mean(values["t_values"]))
                    if values["t_values"]
                    else np.nan
                ),
                "n_folds": len(values["deltas"]),
            }
        )
    candidates = pd.DataFrame(rows)
    expected_folds = len(first)
    pool = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=[SELECTION_STAT])
    pool = pool[pool["n_folds"].eq(expected_folds)]
    if pool.empty:
        return {}, candidates
    chosen = pool.sort_values(
        [SELECTION_STAT, "n_estimators", "selected"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    return chosen.to_dict(), candidates


def simulation_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    individual_rows = []
    stage_rows = []
    for dgp, cache_dir in [
        ("dgp1", base.ROOT / "output" / "dgp1" / "monte_carlo"),
        ("dgp2", base.ROOT / "output" / "dgp2" / "monte_carlo"),
    ]:
        seeds = base.discover_seeds(cache_dir)
        print(f"[SIM] {dgp}: {len(seeds)} saved prediction objects", flush=True)
        for idx, (seed, path) in enumerate(seeds, start=1):
            cache = joblib.load(path)
            estimator_names = list(cache["estimator_names"])
            if len(estimator_names) != 23:
                raise ValueError(
                    f"{dgp} seed {seed}: expected 23 estimators, found {len(estimator_names)}"
                )
            fold_vectors = base.simulation_fold_vectors(cache)
            scores = first_stage_scores(estimator_names, fold_vectors)
            benchmark = base.evaluate_final(cache, BENCHMARKS[dgp]).get("true_ate_ho", np.nan)
            eligible = scores[scores["complete_support"]]
            for statistic in FIRST_STAGE_SCORES:
                order = ordered_estimators(scores, statistic)
                if not order:
                    raise ValueError(f"{dgp} seed {seed}: no complete-support estimators")
                chosen, _ = choose_nested_ensemble(order, fold_vectors)
                final = base.evaluate_final(cache, split_selected(chosen.get("selected", "")))
                anchor_row = scores.set_index("estimator").loc[order[0]]
                anchor_regret = benchmark - final.get("true_ate_ho", np.nan)
                detail_rows.append(
                    {
                        "dgp": dgp,
                        "seed": seed,
                        "procedure": f"{statistic}_common_support__select_{SELECTION_STAT}",
                        "first_stage_score": statistic,
                        "selection_stat": SELECTION_STAT,
                        "anchor_model": order[0],
                        "anchor_score": anchor_row[statistic],
                        "n_stage1_eligible": len(eligible),
                        "n_stage1_excluded": len(estimator_names) - len(eligible),
                        "excluded_estimators": "|".join(
                            sorted(set(estimator_names) - set(order))
                        ),
                        "path_order": "|".join(order),
                        "benchmark": "|".join(BENCHMARKS[dgp]),
                        "benchmark_true_ate": benchmark,
                        **chosen,
                        **final,
                        "anchor_regret": anchor_regret,
                        "anchor_relative_regret": (
                            anchor_regret / benchmark
                            if np.isfinite(anchor_regret) and benchmark != 0
                            else np.nan
                        ),
                    }
                )
            for _, row in scores.iterrows():
                stage_rows.append({"dgp": dgp, "seed": seed, **row.to_dict()})
            for name in estimator_names:
                result = base.evaluate_final(cache, [name])
                individual_rows.append(
                    {
                        "dgp": dgp,
                        "seed": seed,
                        "estimator": name,
                        "true_ate_ho": result.get("true_ate_ho", np.nan),
                    }
                )
            if idx % 20 == 0:
                print(f"  processed {idx}/{len(seeds)}", flush=True)
    return pd.DataFrame(detail_rows), pd.DataFrame(individual_rows), pd.DataFrame(stage_rows)


def summarize_simulation(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["first_stage_score", "dgp"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_true_ate_ho=("true_ate_ho", "mean"),
            se_true_ate_ho=("true_ate_ho", lambda x: float(x.std(ddof=1) / math.sqrt(x.count()))),
            mean_benchmark_true_ate=("benchmark_true_ate", "mean"),
            mean_anchor_regret=("anchor_regret", "mean"),
            mean_anchor_relative_regret=("anchor_relative_regret", "mean"),
            mean_n_estimators=("n_estimators", "mean"),
            median_n_estimators=("n_estimators", "median"),
            mean_stage1_eligible=("n_stage1_eligible", "mean"),
            min_stage1_eligible=("n_stage1_eligible", "min"),
        )
        .reset_index()
    )


def method_ranks(detail: pd.DataFrame, individual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dgp, group in individual.groupby("dgp", sort=False):
        individual_means = group.groupby("estimator")["true_ate_ho"].mean().to_dict()
        for statistic in FIRST_STAGE_SCORES:
            means = dict(individual_means)
            method_name = f"{statistic} first-stage ensemble"
            means[method_name] = float(
                detail.loc[
                    detail["dgp"].eq(dgp)
                    & detail["first_stage_score"].eq(statistic),
                    "true_ate_ho",
                ].mean()
            )
            ordered = sorted(means.items(), key=lambda item: (-item[1], item[0]))
            for rank, (method, value) in enumerate(ordered, start=1):
                rows.append(
                    {
                        "dgp": dgp,
                        "first_stage_score": statistic,
                        "rank": rank,
                        "method": method,
                        "method_type": (
                            "First-stage ensemble"
                            if method == method_name
                            else "Individual estimator"
                        ),
                        "mean_true_ate_ho": value,
                    }
                )
    return pd.DataFrame(rows)


def compare_first_stage_scores(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = ["selected", "n_estimators", "true_ate_ho", "anchor_relative_regret"]
    wide = detail.pivot(index=["dgp", "seed"], columns="first_stage_score", values=value_columns)
    wide.columns = [f"{value}_{score}" for value, score in wide.columns]
    paired = wide.reset_index()
    paired["B_bar_0.9_minus_cal_slope_true_ate"] = (
        paired["true_ate_ho_B_bar_0.9"] - paired["true_ate_ho_cal_slope"]
    )
    rows = []
    for dgp, group in paired.groupby("dgp", sort=False):
        diff = group["B_bar_0.9_minus_cal_slope_true_ate"]
        se = float(diff.std(ddof=1) / math.sqrt(diff.count()))
        rows.append(
            {
                "dgp": dgp,
                "n": len(group),
                "mean_B_bar_0.9_true_ate": group["true_ate_ho_B_bar_0.9"].mean(),
                "mean_cal_slope_true_ate": group["true_ate_ho_cal_slope"].mean(),
                "mean_B_bar_0.9_minus_cal_slope": diff.mean(),
                "ci95_lo": diff.mean() - 1.96 * se,
                "ci95_hi": diff.mean() + 1.96 * se,
                "mean_B_bar_0.9_anchor_relative_regret": group[
                    "anchor_relative_regret_B_bar_0.9"
                ].mean(),
                "mean_cal_slope_anchor_relative_regret": group[
                    "anchor_relative_regret_cal_slope"
                ].mean(),
            }
        )
    return paired, pd.DataFrame(rows)


def evaluate_cached_real_final(cache: dict, names: list[str]) -> dict[str, float | int]:
    def valid_prediction(name: str, key: str, expected: int) -> bool:
        if name not in cache[key]:
            return False
        value = cache[key][name]
        if value is None:
            return False
        array = np.asarray(value)
        return array.ndim == 1 and array.size == expected and np.isfinite(array).all()

    available_tv = [
        name
        for name in names
        if valid_prediction(name, "tau_tv", int(cache["n_tv"]))
    ]
    available_hold = [
        name
        for name in names
        if valid_prediction(name, "tau_hold", int(cache["n_hold"]))
    ]
    available = sorted(set(available_tv).intersection(available_hold))
    if len(available) != len(names):
        missing_tv = sorted(set(names) - set(available_tv))
        missing_hold = sorted(set(names) - set(available_hold))
        return {
            "fullfit_predictions_available": False,
            "missing_tau_tv_predictions": "|".join(missing_tv),
            "missing_tau_hold_predictions": "|".join(missing_hold),
            "tv_ATE": np.nan,
            "tv_SE": np.nan,
            "tv_t": np.nan,
            "tv_n_subgroup": np.nan,
            "ho_ATE": np.nan,
            "ho_SE": np.nan,
            "ho_t": np.nan,
            "ho_CI90_lo": np.nan,
            "ho_CI95_lo": np.nan,
            "ho_n_subgroup": np.nan,
        }
    tau_tv = np.mean(np.vstack([cache["tau_tv"][name] for name in available]), axis=0)
    tau_hold = np.mean(
        np.vstack([cache["tau_hold"][name] for name in available]), axis=0
    )
    threshold = np.quantile(tau_tv, 1.0 - base.FINAL_FRACTION)
    tv_mask = tau_tv >= threshold
    hold_mask = tau_hold >= threshold
    tv_ate, tv_se = base.subgroup_ate(cache["y_tv"], cache["t_tv"], tv_mask)
    ho_ate, ho_se = base.subgroup_ate(cache["y_hold"], cache["t_hold"], hold_mask)
    return {
        "fullfit_predictions_available": True,
        "missing_tau_tv_predictions": "",
        "missing_tau_hold_predictions": "",
        "tv_ATE": tv_ate,
        "tv_SE": tv_se,
        "tv_t": tv_ate / tv_se if np.isfinite(tv_se) and tv_se > 0 else np.nan,
        "tv_n_subgroup": int(tv_mask.sum()),
        "ho_ATE": ho_ate,
        "ho_SE": ho_se,
        "ho_t": ho_ate / ho_se if np.isfinite(ho_se) and ho_se > 0 else np.nan,
        "ho_CI90_lo": ho_ate - base.Z_90 * ho_se,
        "ho_CI95_lo": ho_ate - base.Z_975 * ho_se,
        "ho_n_subgroup": int(hold_mask.sum()),
    }


def realdata_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows = []
    score_rows = []
    candidate_rows = []
    for arm, project_dir in base.PROJECTS.items():
        print(f"[REAL] {arm}: saved fold and full-fit predictions", flush=True)
        data = real_helpers.load_real_project(project_dir)
        fitted_libs = data["fitted_libs"]
        valid_names = list(fitted_libs["pert_none"].keys())
        if len(valid_names) != 23:
            raise ValueError(f"{arm}: expected 23 fitted estimators, found {len(valid_names)}")
        y = data["trainval_df"][real_helpers.OUTCOME].to_numpy(dtype=float)
        t = real_helpers.treatment_array(data["trainval_df"], data["treatment_var"])
        fold_vectors = base.real_fold_vectors(fitted_libs, valid_names, y, t)
        scores = first_stage_scores(valid_names, fold_vectors)
        prediction_cache = joblib.load(REAL_CACHE_DIR / f"{arm}_predictions.pkl")
        eligible = scores[scores["complete_support"]]
        for statistic in FIRST_STAGE_SCORES:
            order = ordered_estimators(scores, statistic)
            chosen, candidates = choose_nested_ensemble(order, fold_vectors)
            final = evaluate_cached_real_final(
                prediction_cache, split_selected(chosen.get("selected", ""))
            )
            result_rows.append(
                {
                    "arm": arm,
                    "procedure": f"{statistic}_common_support__select_{SELECTION_STAT}",
                    "first_stage_score": statistic,
                    "selection_stat": SELECTION_STAT,
                    "anchor_model": order[0],
                    "anchor_score": scores.set_index("estimator").loc[order[0], statistic],
                    "n_stage1_eligible": len(eligible),
                    "n_stage1_excluded": len(valid_names) - len(eligible),
                    "excluded_estimators": "|".join(
                        sorted(set(valid_names) - set(order))
                    ),
                    "path_order": "|".join(order),
                    **chosen,
                    **final,
                }
            )
            for rank, name in enumerate(order, start=1):
                row = scores.loc[scores["estimator"].eq(name)].iloc[0].to_dict()
                score_rows.append(
                    {"arm": arm, "first_stage_score": statistic, "stage1_rank": rank, **row}
                )
            for _, row in candidates.iterrows():
                candidate_rows.append(
                    {"arm": arm, "first_stage_score": statistic, **row.to_dict()}
                )
    return pd.DataFrame(result_rows), pd.DataFrame(score_rows), pd.DataFrame(candidate_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail, individual, stage = simulation_results()
    summary = summarize_simulation(detail)
    ranks = method_ranks(detail, individual)
    paired, paired_summary = compare_first_stage_scores(detail)

    detail.to_csv(OUTPUT_DIR / "simulation_detail.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "simulation_summary.csv", index=False)
    individual.to_csv(OUTPUT_DIR / "simulation_individual_detail.csv", index=False)
    stage.to_csv(OUTPUT_DIR / "simulation_stage1_scores.csv", index=False)
    ranks.to_csv(OUTPUT_DIR / "simulation_method_ranks.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "paired_first_stage_scores.csv", index=False)
    paired_summary.to_csv(
        OUTPUT_DIR / "paired_first_stage_scores_summary.csv", index=False
    )

    real, real_scores, real_candidates = realdata_results()
    real.to_csv(OUTPUT_DIR / "realdata_results.csv", index=False)
    real_scores.to_csv(OUTPUT_DIR / "realdata_stage1_scores.csv", index=False)
    real_candidates.to_csv(OUTPUT_DIR / "realdata_nested_candidates.csv", index=False)

    print("\nSimulation summary")
    print(summary.to_string(index=False))
    print("\nRanks of common-support procedures")
    print(
        ranks[ranks["method_type"].eq("First-stage ensemble")].to_string(index=False)
    )
    print("\nPaired comparison of first-stage scores")
    print(paired_summary.to_string(index=False))
    print("\nReal-data results")
    print(real.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
