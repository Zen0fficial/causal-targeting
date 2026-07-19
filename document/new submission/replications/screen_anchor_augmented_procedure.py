#!/usr/bin/env python3
"""Screen expanded anchor-augmented procedures.

The near-miss rule Q5__cal_slope__weighted_b selects a reliability model by
Q5 calibration slope and a distinct upside model by Q5 weighted_b.  This script
keeps the anchor idea but removes the two-model restriction:

  1. Select anchor model m0 by Q5 calibration slope.
  2. Rank the remaining models by a predeclared Q5 statistic.
  3. Consider the nested ensembles anchor + top k ranked models, k = 0..M-1.
  4. Score each actual ensemble on validation folds.
  5. Select the best candidate by a direct, predeclared validation statistic.

The screen intentionally drops cal_epm_* product scores.  Calibration enters by
the anchor choice; final ensemble selection is based on direct targeting
statistics.
"""

from __future__ import annotations

import math
import os
import pickle
import random
import sys
import types
import warnings

import joblib
import numpy as np
import pandas as pd

import evaluate_proposed_targeting_statistics as base

warnings.filterwarnings("ignore")

try:
    import duecredit  # noqa: F401
except ImportError:
    duecredit_stub = types.ModuleType("duecredit")
    duecredit_stub.due = types.SimpleNamespace(
        cite=lambda *args, **kwargs: None,
        dcite=lambda *args, **kwargs: (lambda obj: obj),
        dcite_dict=lambda *args, **kwargs: (lambda obj: obj),
    )
    duecredit_stub.BibTeX = lambda value: value
    duecredit_stub.Doi = lambda value: value
    duecredit_stub.Text = lambda value: value
    duecredit_stub.Url = lambda value: value
    sys.modules["duecredit"] = duecredit_stub

OUTPUT_DIR = base.ROOT / "output" / "anchor_expanded_procedure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTCOME = "fausebal"
SEED = 0

ANCHORS = {
    "dgp1": ["s_rf"],
    "dgp2": ["x_rf"],
}

Q_SET = "Q5"
Q_VALUES = base.Q_SETS[Q_SET]
ANCHOR_STAT = "cal_slope"

ADD_RANK_STATS = ["cal_slope"]

SELECTION_STATS = ["mean_delta_top10"]


def split_selected(selected: str) -> list[str]:
    return selected.split("|") if isinstance(selected, str) and selected else []


def selected_name(names: tuple[str, ...]) -> str:
    return "|".join(names)


def aggregate_estimator_scores(score_df: pd.DataFrame, statistic: str) -> pd.Series:
    sub = score_df[score_df["q"].isin(Q_VALUES)].copy()
    if sub.empty or statistic not in sub:
        return pd.Series(dtype=float)
    return (
        sub.groupby("estimator", sort=False)[statistic]
        .mean()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def select_anchor(score_df: pd.DataFrame) -> tuple[str, float]:
    scores = aggregate_estimator_scores(score_df, ANCHOR_STAT)
    if scores.empty:
        return "", np.nan
    ordered = scores.sort_values(ascending=False, kind="mergesort")
    return str(ordered.index[0]), float(ordered.iloc[0])


def ranked_models(score_df: pd.DataFrame, statistic: str, anchor: str) -> list[str]:
    scores = aggregate_estimator_scores(score_df, statistic)
    if scores.empty:
        return []
    scores = scores.drop(labels=[anchor], errors="ignore")
    ordered = scores.sort_values(ascending=False, kind="mergesort")
    return [str(name) for name in ordered.index]


def anchored_candidates(
    estimator_names: list[str],
    anchor: str,
    add_rank_stat: str,
    score_df: pd.DataFrame,
) -> list[tuple[str, ...]]:
    names = sorted(name for name in estimator_names if name)
    if anchor not in names:
        return []
    out: list[tuple[str, ...]] = [(anchor,)]
    ranked = ranked_models(score_df, add_rank_stat, anchor)
    current = [anchor]
    for name in ranked:
        if name in names and name not in current:
            current.append(name)
            out.append(tuple(current))
    return out


def summarize_targeting_values(
    deltas: list[float],
    t_values: list[float],
    b_values: list[float],
    suffix: str,
) -> dict[str, float]:
    d = np.asarray(deltas, dtype=float)
    mean_delta = float(np.mean(d)) if d.size else np.nan
    sd_delta = float(np.std(d, ddof=1)) if d.size > 1 else np.nan
    abs_sum = float(np.sum(np.abs(d))) if d.size else np.nan
    weighted_b = (
        float(np.sum(np.maximum(d, 0.0)) / abs_sum)
        if np.isfinite(abs_sum) and abs_sum > 0
        else np.nan
    )
    return {
        f"b_bar_{suffix}": float(np.mean(b_values)) if b_values else np.nan,
        f"weighted_b_{suffix}": weighted_b,
        f"epm_{suffix}": float(np.mean(np.maximum(d, 0.0))) if d.size else np.nan,
        f"mean_delta_{suffix}": mean_delta,
        f"mean_t_{suffix}": float(np.mean(t_values)) if t_values else np.nan,
        f"lcb_delta_{suffix}": (
            mean_delta - base.LCB_Z * (sd_delta / math.sqrt(d.size))
            if d.size > 1 and np.isfinite(sd_delta)
            else np.nan
        ),
        f"min_delta_{suffix}": float(np.min(d)) if d.size else np.nan,
    }


def candidate_metrics(
    names: tuple[str, ...],
    fold_vectors: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]],
) -> dict[str, float]:
    first = fold_vectors.get(names[0], [])
    by_q = {
        q: {"deltas": [], "t_values": [], "b_values": []}
        for q in Q_VALUES
    }
    for fold_idx in range(len(first)):
        tau_sum = None
        y = t = train_mask = None
        for name in names:
            current = fold_vectors.get(name, [])
            if fold_idx >= len(current):
                continue
            tau, y, t, train_mask = current[fold_idx]
            tau_sum = tau.astype(float, copy=True) if tau_sum is None else tau_sum + tau
        if tau_sum is None or y is None or t is None or train_mask is None:
            continue
        tau_mean = tau_sum / float(len(names))
        val_mask = ~train_mask
        for q in Q_VALUES:
            m = base.top_vs_rest_metrics(y, t, tau_mean, train_mask, val_mask, q)
            if np.isfinite(m["delta"]):
                by_q[q]["deltas"].append(m["delta"])
                by_q[q]["b_values"].append(m["b"])
            if np.isfinite(m["t_stat"]):
                by_q[q]["t_values"].append(m["t_stat"])

    all_deltas: list[float] = []
    all_t_values: list[float] = []
    all_b_values: list[float] = []
    for q in Q_VALUES:
        all_deltas.extend(by_q[q]["deltas"])
        all_t_values.extend(by_q[q]["t_values"])
        all_b_values.extend(by_q[q]["b_values"])

    top10 = by_q[base.FINAL_FRACTION]
    out = summarize_targeting_values(all_deltas, all_t_values, all_b_values, "Q5")
    out.update(
        summarize_targeting_values(
            top10["deltas"],
            top10["t_values"],
            top10["b_values"],
            "top10",
        )
    )
    return out


def score_anchored_candidates(
    estimator_names: list[str],
    fold_vectors: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]],
    anchor: str,
    add_rank_stat: str,
    score_df: pd.DataFrame,
    metrics_cache: dict[tuple[str, ...], dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for names in anchored_candidates(estimator_names, anchor, add_rank_stat, score_df):
        added = [name for name in names if name != anchor]
        row = {
            "selected": selected_name(names),
            "n_estimators": len(names),
            "anchor_model": anchor,
            "add_rank_stat": add_rank_stat,
            "add_k": len(added),
            "added_models": "|".join(added),
        }
        if names not in metrics_cache:
            metrics_cache[names] = candidate_metrics(names, fold_vectors)
        row.update(metrics_cache[names])
        rows.append(row)
    return pd.DataFrame(rows)


def choose_candidate(candidates: pd.DataFrame, statistic: str) -> dict:
    pool = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=[statistic]).copy()
    if pool.empty:
        return {
            "statistic": statistic,
            "selected": "",
            "n_estimators": 0,
            "anchor_model": "",
            "add_rank_stat": "",
            "add_k": 0,
            "added_models": "",
            "selection_score": np.nan,
        }
    ordered = pool.sort_values(
        [statistic, "n_estimators"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    )
    row = ordered.iloc[0].to_dict()
    row["statistic"] = statistic
    row["selection_score"] = row.get(statistic, np.nan)
    return row


def procedure_label(add_rank_stat: str, statistic: str) -> str:
    return f"{Q_SET}__{ANCHOR_STAT}_anchor__rank_{add_rank_stat}__select_{statistic}"


def run_simulation() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dgp, cache_dir in [
        ("dgp1", base.ROOT / "output" / "dgp1" / "monte_carlo"),
        ("dgp2", base.ROOT / "output" / "dgp2" / "monte_carlo"),
    ]:
        seeds = base.discover_seeds(cache_dir)
        print(f"[SIM] {dgp}: {len(seeds)} seeds", flush=True)
        for idx, (seed, path) in enumerate(seeds, start=1):
            if idx % 25 == 1:
                print(f"  seed {idx}/{len(seeds)}", flush=True)
            cache = joblib.load(path)
            fold_vectors = base.simulation_fold_vectors(cache)
            scores = base.fold_scores_from_vectors(list(cache["estimator_names"]), fold_vectors)
            anchor_model, anchor_score = select_anchor(scores)
            benchmark = base.evaluate_final(cache, ANCHORS[dgp]).get("true_ate_ho", np.nan)
            metrics_cache: dict[tuple[str, ...], dict[str, float]] = {}

            for add_rank_stat in ADD_RANK_STATS:
                candidates = score_anchored_candidates(
                    list(cache["estimator_names"]),
                    fold_vectors,
                    anchor_model,
                    add_rank_stat,
                    scores,
                    metrics_cache,
                )
                for statistic in SELECTION_STATS:
                    chosen = choose_candidate(candidates, statistic)
                    final = base.evaluate_final(cache, split_selected(chosen["selected"]))
                    anchor_regret = (
                        benchmark - final.get("true_ate_ho", np.nan)
                        if np.isfinite(benchmark)
                        else np.nan
                    )
                    rows.append(
                        {
                            "dgp": dgp,
                            "seed": seed,
                            "procedure": procedure_label(add_rank_stat, statistic),
                            "anchor_stat": ANCHOR_STAT,
                            "anchor_score": anchor_score,
                            "benchmark": "|".join(ANCHORS[dgp]),
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
            if idx % 25 == 0:
                pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sim_detail_partial.csv", index=False)

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["procedure", "add_rank_stat", "statistic", "dgp"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_anchor_relative_regret=("anchor_relative_regret", "mean"),
            median_anchor_relative_regret=("anchor_relative_regret", "median"),
            p90_anchor_relative_regret=(
                "anchor_relative_regret",
                lambda s: float(np.nanpercentile(s, 90)),
            ),
            mean_anchor_regret=("anchor_regret", "mean"),
            mean_true_ate_ho=("true_ate_ho", "mean"),
            mean_benchmark_true_ate=("benchmark_true_ate", "mean"),
            mean_true_oracle_relative_regret=("relative_regret", "mean"),
            mean_ho_t=("ho_t", "mean"),
            mean_n_estimators=("n_estimators", "mean"),
            median_n_estimators=("n_estimators", "median"),
        )
        .reset_index()
    )
    return detail, summary


def load_real_project(project_dir: base.Path) -> dict:
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    analysis_dir = project_dir / "output" / "analysis" / OUTCOME
    params_dir = project_dir / "output" / "params" / OUTCOME
    fitted_libs = joblib.load(analysis_dir / f"{OUTCOME}_fitted_libraries.pkl")
    with open(params_dir / "analysis_imputation_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    trainval_df = pd.read_csv(analysis_dir / "trainval_data.csv")
    holdout_df = pd.read_csv(analysis_dir / "holdout_data.csv")
    treatment_var = meta.get("treatment_var", "TREATED")
    return {
        "fitted_libs": fitted_libs,
        "features": meta["features"],
        "treatment_var": treatment_var,
        "trainval_df": trainval_df,
        "holdout_df": holdout_df,
    }


def treatment_array(df: pd.DataFrame, treatment_var: str) -> np.ndarray:
    column = "TREATED" if "TREATED" in df.columns else treatment_var
    return df[column].to_numpy(dtype=int)


def predict_on_holdout(result, estimator_wrapper, x_hold: np.ndarray) -> np.ndarray:
    from methods.cate_estimator_wrappers import XLearnerWrapper

    x_new = np.asarray(x_hold, dtype=float)
    if getattr(result, "_selector", None) is not None:
        n_features = estimator_wrapper.X.shape[1]
        col_names = [f"x_{i}" for i in range(n_features)]
        try:
            x_df = pd.DataFrame(x_new, columns=pd.Index(col_names))
            x_new = result._selector.transform(x_df).values
        except Exception:
            pass
    if isinstance(estimator_wrapper, XLearnerWrapper):
        propensity = np.mean(estimator_wrapper.t) * np.ones(x_new.shape[0])
        pred = result.meta_learner.predict(x_new, p=propensity)
    else:
        pred = result.meta_learner.predict(x_new)
    return np.asarray(pred, dtype=float).squeeze()


def force_single_process_fit(obj) -> None:
    """Avoid joblib/loky process spawning inside sandboxed real-data refits."""
    for attr in ["cv_n_jobs", "n_jobs"]:
        if hasattr(obj, attr):
            try:
                setattr(obj, attr, 1)
            except Exception:
                pass
    for attr in [
        "meta_learner",
        "model",
        "model_c",
        "model_t",
        "model_mu",
        "model_tau",
        "model_t",
        "model_y",
        "outcome_learner",
        "effect_learner",
        "propensity_learner",
        "treatment_outcome_learner",
        "control_outcome_learner",
        "treatment_effect_learner",
        "control_effect_learner",
        "learner",
    ]:
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            force_single_process_fit(child)


def real_full_predictions(
    project_dir: base.Path,
    fitted_libs: dict,
    trainval_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    features: list[str],
    name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from methods.cate_estimator_wrappers import CATEEstimatorResults

    lib = fitted_libs["pert_none"]
    if name not in lib:
        return None
    estimator = lib[name]
    force_single_process_fit(estimator)
    n_tv = len(next(iter(lib.values())).y)
    train_indices = np.arange(n_tv)
    val_indices = np.arange(n_tv)
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    result = CATEEstimatorResults(
        train_indices,
        val_indices,
        estimator,
        save_metalearner=True,
    )
    x_hold = (
        holdout_df[features]
        .copy()
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return np.asarray(result.tau, dtype=float), predict_on_holdout(result, estimator, x_hold)


def evaluate_real_final(
    project_dir: base.Path,
    fitted_libs: dict,
    trainval_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    features: list[str],
    treatment_var: str,
    names: list[str],
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    if not names:
        return {}
    for name in names:
        if name not in prediction_cache:
            preds = real_full_predictions(
                project_dir,
                fitted_libs,
                trainval_df,
                holdout_df,
                features,
                name,
            )
            if preds is not None:
                prediction_cache[name] = preds
    available = [name for name in names if name in prediction_cache]
    if not available:
        return {}
    tau_tv = np.mean(np.vstack([prediction_cache[name][0] for name in available]), axis=0)
    tau_ho = np.mean(np.vstack([prediction_cache[name][1] for name in available]), axis=0)
    threshold = np.quantile(tau_tv, 1.0 - base.FINAL_FRACTION)
    sg_tv = tau_tv >= threshold
    sg_ho = tau_ho >= threshold

    y_tv = trainval_df[OUTCOME].to_numpy(dtype=float)
    t_tv = treatment_array(trainval_df, treatment_var)
    y_ho = holdout_df[OUTCOME].to_numpy(dtype=float)
    t_ho = treatment_array(holdout_df, treatment_var)
    tv_ate, tv_se = base.subgroup_ate(y_tv, t_tv, sg_tv)
    ho_ate, ho_se = base.subgroup_ate(y_ho, t_ho, sg_ho)
    return {
        "tv_ATE": tv_ate,
        "tv_SE": tv_se,
        "tv_t": tv_ate / tv_se if np.isfinite(tv_se) and tv_se > 0 else np.nan,
        "tv_n_subgroup": int(sg_tv.sum()),
        "ho_ATE": ho_ate,
        "ho_SE": ho_se,
        "ho_t": ho_ate / ho_se if np.isfinite(ho_se) and ho_se > 0 else np.nan,
        "ho_CI90_lo": ho_ate - base.Z_90 * ho_se if np.isfinite(ho_ate) and np.isfinite(ho_se) else np.nan,
        "ho_CI95_lo": ho_ate - base.Z_975 * ho_se if np.isfinite(ho_ate) and np.isfinite(ho_se) else np.nan,
        "ho_n_subgroup": int(sg_ho.sum()),
    }


def run_realdata() -> pd.DataFrame:
    rows = []
    for arm, project_dir in base.PROJECTS.items():
        print(f"[REAL] {arm}", flush=True)
        data = load_real_project(project_dir)
        fitted_libs = data["fitted_libs"]
        trainval_df = data["trainval_df"]
        holdout_df = data["holdout_df"]
        features = data["features"]
        treatment_var = data["treatment_var"]
        valid_names = list(fitted_libs["pert_none"].keys())
        y = trainval_df[OUTCOME].to_numpy(dtype=float)
        t = treatment_array(trainval_df, treatment_var)
        fold_vectors = base.real_fold_vectors(fitted_libs, valid_names, y, t)
        scores = base.fold_scores_from_vectors(valid_names, fold_vectors)
        anchor_model, anchor_score = select_anchor(scores)
        metrics_cache: dict[tuple[str, ...], dict[str, float]] = {}
        prediction_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for add_rank_stat in ADD_RANK_STATS:
            candidates = score_anchored_candidates(
                valid_names,
                fold_vectors,
                anchor_model,
                add_rank_stat,
                scores,
                metrics_cache,
            )
            for statistic in SELECTION_STATS:
                chosen = choose_candidate(candidates, statistic)
                final = evaluate_real_final(
                    project_dir,
                    fitted_libs,
                    trainval_df,
                    holdout_df,
                    features,
                    treatment_var,
                    split_selected(chosen["selected"]),
                    prediction_cache,
                )
                rows.append(
                    {
                        "arm": arm,
                        "procedure": procedure_label(add_rank_stat, statistic),
                        "add_rank_stat": add_rank_stat,
                        "anchor_stat": ANCHOR_STAT,
                        "anchor_score": anchor_score,
                        **chosen,
                        **final,
                    }
                )
    return pd.DataFrame(rows)


def build_combined_summary(sim_summary: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    sim_piv = sim_summary.pivot_table(
        index=["procedure", "add_rank_stat", "statistic"],
        columns="dgp",
        values=[
            "mean_anchor_relative_regret",
            "mean_anchor_regret",
            "mean_true_ate_ho",
            "mean_benchmark_true_ate",
            "mean_true_oracle_relative_regret",
        ],
        aggfunc="first",
    )
    sim_piv.columns = [f"{a}_{b}" for a, b in sim_piv.columns]
    sim_piv = sim_piv.reset_index()

    real_piv = real_df.pivot_table(
        index=["procedure", "add_rank_stat", "statistic"],
        columns="arm",
        values=["ho_t", "ho_ATE", "ho_CI90_lo", "tv_t", "ho_n_subgroup", "selected"],
        aggfunc="first",
    )
    real_piv.columns = [f"{a}_{b}" for a, b in real_piv.columns]
    real_piv = real_piv.reset_index()

    out = sim_piv.merge(real_piv, on=["procedure", "add_rank_stat", "statistic"], how="outer")
    anchor_cols = [c for c in out.columns if c.startswith("mean_anchor_relative_regret_")]
    ho_cols = [c for c in out.columns if c.startswith("ho_t_")]
    out["max_anchor_relative_regret"] = out[anchor_cols].max(axis=1)
    out["max_real_ho_t"] = out[ho_cols].max(axis=1)
    out["passes_anchor_rel_10pct"] = out["max_anchor_relative_regret"] <= base.REL_REGRET_TARGET
    out["passes_any_real_ho_10pct"] = out["max_real_ho_t"] >= base.Z_90
    out["passes_both_anchor_targets"] = (
        out["passes_anchor_rel_10pct"] & out["passes_any_real_ho_10pct"]
    )
    return out.sort_values(
        ["passes_both_anchor_targets", "max_anchor_relative_regret"],
        ascending=[False, True],
    )


def main() -> None:
    sim_detail, sim_summary = run_simulation()
    sim_detail_path = OUTPUT_DIR / "sim_detail.csv"
    sim_summary_path = OUTPUT_DIR / "sim_summary.csv"
    sim_detail.to_csv(sim_detail_path, index=False)
    sim_summary.to_csv(sim_summary_path, index=False)
    print(f"Saved {sim_detail_path}")
    print(f"Saved {sim_summary_path}")

    real_df = run_realdata()
    real_path = OUTPUT_DIR / "realdata_results.csv"
    real_df.to_csv(real_path, index=False)
    print(f"Saved {real_path}")

    combined = build_combined_summary(sim_summary, real_df)
    combined_path = OUTPUT_DIR / "combined_summary.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved {combined_path}")

    cols = [
        "procedure",
        "passes_both_anchor_targets",
        "max_anchor_relative_regret",
        "max_real_ho_t",
        "mean_anchor_relative_regret_dgp1",
        "mean_anchor_relative_regret_dgp2",
        "selected_billpayfa",
        "ho_t_billpayfa",
        "selected_debitfa",
        "ho_t_debitfa",
        "selected_main",
        "ho_t_main",
    ]
    cols = [c for c in cols if c in combined.columns]
    print("\nTop expanded anchor-augmented procedures:")
    print(combined[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
