#!/usr/bin/env python3
"""Evaluate proposed targeting statistics against the current manuscript grid.

This script intentionally starts from the anonymous JBES submission design:

  * candidate quantile sets Q1..Q5 are {0.9}, {0.9,0.8}, ..., {0.9..0.5}
    in manuscript notation, implemented as top fractions {0.1}, {0.1,0.2}, ...
  * k varies from 1 to 10.
  * selected estimators are aggregated by a simple arithmetic mean.
  * final targeting is the top 10% subgroup.
  * holdout inference is Neyman only.

The only component varied here is the statistic used to choose the best
(Q, k) configuration from training/validation folds.
"""

from __future__ import annotations

import math
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ROOT points to the original project dir, two levels above replications/.
ROOT = Path(__file__).resolve().parents[2] / "projects" / "causal-targeting-simulation"
DOC_ROOT = ROOT.parents[1]
CACHE_DIR = ROOT / "output" / "realdata_replication" / "cache"
OUTPUT_DIR = ROOT / "output" / "proposed_targeting_statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECTS = {
    "billpayfa": DOC_ROOT / "projects" / "causal-targeting-billpayfa",
    "debitfa": DOC_ROOT / "projects" / "causal-targeting-debitfa",
    "main": DOC_ROOT / "projects" / "causal-targeting-main",
}

SEED_RE = re.compile(r"seed_(\d+)$")
FINAL_FRACTION = 0.10
Z_975 = 1.959963984540054
Z_90 = 1.6448536269514722
REL_REGRET_TARGET = 0.10
LCB_Z = Z_90

Q_SETS = {
    "Q1": [0.10],
    "Q2": [0.10, 0.20],
    "Q3": [0.10, 0.20, 0.30],
    "Q4": [0.10, 0.20, 0.30, 0.40],
    "Q5": [0.10, 0.20, 0.30, 0.40, 0.50],
}
K_VALUES = list(range(1, 11))

HIGHER_IS_BETTER = {
    "current_b_then_delta": True,
    "mean_delta": True,
    "mean_t": True,
    "weighted_b": True,
    "epm": True,
    "crossfold_t": True,
    "snr": True,
    "lcb_delta": True,
    "min_delta": True,
    "aut_delta": True,
    "targeting_monotonicity": True,
    "neg_targeting_slope": True,
    "qini": True,
    "rate": True,
    "cal_slope": True,
    "cal_r2": True,
    "cal_monotonicity": True,
    "strata_ate_sd": True,
    "targeting_sharpe": True,
    "cts": True,
    "sat": True,
}

ACTIVE_STATISTICS = [
    "current_b_then_delta",
    "mean_delta",
    "mean_t",
    "weighted_b",
    "epm",
    "crossfold_t",
    "snr",
    "lcb_delta",
    "min_delta",
    "cal_slope",
    "cal_r2",
    "cal_monotonicity",
    "strata_ate_sd",
    "targeting_sharpe",
    "sat",
    "cts",
]


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
    if int(mask.sum()) == 0:
        return np.nan, np.nan
    return neyman_ate(y[mask], t[mask])


def top_mask_from_train_quantile(
    tau: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray | None,
    q: float,
    dir_neg: bool = False,
) -> np.ndarray:
    train_tau = tau[train_mask]
    if train_tau.size == 0:
        mask = np.zeros_like(tau, dtype=bool)
    elif dir_neg:
        mask = tau <= np.quantile(train_tau, q)
    else:
        mask = tau >= np.quantile(train_tau, 1.0 - q)
    return mask if eval_mask is None else (mask & eval_mask)


def final_top_mask(tau: np.ndarray, q: float = FINAL_FRACTION, dir_neg: bool = False) -> np.ndarray:
    if dir_neg:
        return tau <= np.quantile(tau, q)
    return tau >= np.quantile(tau, 1.0 - q)


def bin_mask_from_train_quantile(
    tau: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    q_low: float,
    q_high: float,
    dir_neg: bool = False,
) -> np.ndarray:
    train_tau = tau[train_mask]
    if train_tau.size == 0:
        return np.zeros_like(tau, dtype=bool)
    if dir_neg:
        lo = np.quantile(train_tau, q_low)
        hi = np.quantile(train_tau, q_high)
        mask = (tau >= lo) & (tau <= hi) if q_low == 0 else (tau > lo) & (tau <= hi)
    else:
        # bins are expressed from the bottom of the priority distribution;
        # q_low=0.8,q_high=1.0 is the top 20%.
        lo = np.quantile(train_tau, q_low)
        hi = np.quantile(train_tau, q_high)
        mask = (tau >= lo) & (tau <= hi) if q_high >= 1.0 else (tau >= lo) & (tau < hi)
    return mask & eval_mask


def top_vs_rest_metrics(
    y: np.ndarray,
    t: np.ndarray,
    tau: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    q: float,
    dir_neg: bool = False,
) -> dict:
    top = top_mask_from_train_quantile(tau, train_mask, val_mask, q, dir_neg)
    rest = val_mask & (~top)
    top_ate, top_se = subgroup_ate(y, t, top)
    rest_ate, rest_se = subgroup_ate(y, t, rest)
    if np.isfinite(top_ate) and np.isfinite(rest_ate):
        delta = top_ate - rest_ate
        se = math.sqrt(top_se**2 + rest_se**2) if np.isfinite(top_se) and np.isfinite(rest_se) else np.nan
        t_stat = delta / se if np.isfinite(se) and se > 0 else np.nan
    else:
        delta = np.nan
        se = np.nan
        t_stat = np.nan
    if dir_neg and np.isfinite(delta):
        delta *= -1
        t_stat *= -1 if np.isfinite(t_stat) else np.nan
    return {
        "delta": float(delta) if np.isfinite(delta) else np.nan,
        "se": float(se) if np.isfinite(se) else np.nan,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "b": float(delta > 0) if np.isfinite(delta) else np.nan,
        "top_ate": float(top_ate) if np.isfinite(top_ate) else np.nan,
        "rest_ate": float(rest_ate) if np.isfinite(rest_ate) else np.nan,
        "n_top": int(top.sum()),
    }


def calibration_metrics(
    y: np.ndarray,
    t: np.ndarray,
    tau: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    bins: int = 5,
) -> dict:
    q_values = np.linspace(0.0, 1.0, bins + 1)
    tau_means: list[float] = []
    ates: list[float] = []
    weights: list[float] = []
    for idx in range(bins):
        mask = bin_mask_from_train_quantile(tau, train_mask, val_mask, q_values[idx], q_values[idx + 1])
        if mask.sum() == 0:
            continue
        ate, _ = subgroup_ate(y, t, mask)
        if not np.isfinite(ate):
            continue
        tau_means.append(float(np.nanmean(tau[mask])))
        ates.append(float(ate))
        weights.append(float(mask.sum()) / float(val_mask.sum()))
    if len(ates) < 2 or np.nanstd(tau_means) <= 1e-12:
        return {
            "cal_slope": np.nan,
            "cal_r2": np.nan,
            "cal_monotonicity": np.nan,
            "strata_ate_sd": np.nan,
        }
    x = np.asarray(tau_means, dtype=float)
    z = np.asarray(ates, dtype=float)
    slope = float(np.cov(x, z, ddof=1)[0, 1] / np.var(x, ddof=1))
    pred = z.mean() + slope * (x - x.mean())
    sst = float(np.sum((z - z.mean()) ** 2))
    r2 = float(1.0 - np.sum((z - pred) ** 2) / sst) if sst > 0 else np.nan
    pairs = 0
    good = 0
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            pairs += 1
            good += int(z[i] <= z[j])
    return {
        "cal_slope": slope,
        "cal_r2": r2,
        "cal_monotonicity": float(good / pairs) if pairs else np.nan,
        "strata_ate_sd": float(np.std(z, ddof=1)) if len(z) > 1 else np.nan,
    }


def rate_qini_metrics(
    y: np.ndarray,
    t: np.ndarray,
    tau: np.ndarray,
    val_mask: np.ndarray,
    q_max: float = FINAL_FRACTION,
) -> dict:
    idx = np.where(val_mask & np.isfinite(tau))[0]
    if idx.size < 10:
        return {"qini": np.nan, "rate": np.nan}
    priority = tau[idx]
    order = np.argsort(-priority, kind="mergesort")
    ordered = idx[order]
    ate_all, _ = subgroup_ate(y, t, val_mask)
    if not np.isfinite(ate_all):
        return {"qini": np.nan, "rate": np.nan}
    n = len(ordered)
    fractions = np.linspace(0.02, min(0.50, q_max), 25)
    qini_vals = []
    rate_vals = []
    for f in fractions:
        m = max(2, int(math.ceil(f * n)))
        mask = np.zeros_like(val_mask, dtype=bool)
        mask[ordered[:m]] = True
        ate_f, _ = subgroup_ate(y, t, mask)
        if np.isfinite(ate_f):
            rate_vals.append(ate_f)
            qini_vals.append(ate_f - ate_all)
    return {
        "qini": float(np.mean(qini_vals)) if qini_vals else np.nan,
        "rate": float(np.mean(rate_vals)) if rate_vals else np.nan,
    }


def fold_scores_from_vectors(
    estimator_names: list[str],
    fold_vectors: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]],
    include_rate_qini: bool = False,
) -> pd.DataFrame:
    rows = []
    for estimator in estimator_names:
        vectors = fold_vectors.get(estimator, [])
        for q in sorted({q for qs in Q_SETS.values() for q in qs}):
            deltas = []
            t_values = []
            b_values = []
            epm_values = []
            cal_rows = []
            qini_values = []
            rate_values = []
            for tau, y, t, train_mask in vectors:
                # For real-data fitted libraries, tau/y/t are full TV fold vectors and
                # train_mask is a boolean train indicator. For simulation vectors the
                # same convention is used.
                if train_mask is None:
                    train_mask = np.ones_like(tau, dtype=bool)
                    val_mask = np.ones_like(tau, dtype=bool)
                else:
                    val_mask = ~train_mask
                metrics = top_vs_rest_metrics(y, t, tau, train_mask, val_mask, q)
                if np.isfinite(metrics["delta"]):
                    deltas.append(metrics["delta"])
                    b_values.append(metrics["b"])
                    epm_values.append(max(0.0, metrics["delta"]))
                if np.isfinite(metrics["t_stat"]):
                    t_values.append(metrics["t_stat"])
                cal = calibration_metrics(y, t, tau, train_mask, val_mask)
                cal_rows.append(cal)
                if include_rate_qini:
                    rq = rate_qini_metrics(y, t, tau, val_mask, q_max=q)
                    if np.isfinite(rq["qini"]):
                        qini_values.append(rq["qini"])
                    if np.isfinite(rq["rate"]):
                        rate_values.append(rq["rate"])
            d = np.asarray(deltas, dtype=float)
            mean_delta = float(np.mean(d)) if d.size else np.nan
            sd_delta = float(np.std(d, ddof=1)) if d.size > 1 else np.nan
            crossfold_t = mean_delta / (sd_delta / math.sqrt(d.size)) if d.size > 1 and sd_delta > 0 else np.nan
            snr = mean_delta / sd_delta if d.size > 1 and sd_delta > 0 else np.nan
            abs_sum = float(np.sum(np.abs(d))) if d.size else np.nan
            weighted_b = float(np.sum(np.maximum(d, 0.0)) / abs_sum) if np.isfinite(abs_sum) and abs_sum > 0 else np.nan
            cv = sd_delta / abs(mean_delta) if np.isfinite(sd_delta) and np.isfinite(mean_delta) and mean_delta != 0 else np.nan
            sat = mean_delta * (1.0 - cv) if np.isfinite(cv) else np.nan
            cal_df = pd.DataFrame(cal_rows)
            rows.append(
                {
                    "estimator": estimator,
                    "q": q,
                    "mean_delta": mean_delta,
                    "mean_t": float(np.mean(t_values)) if t_values else np.nan,
                    "weighted_b": weighted_b,
                    "epm": float(np.mean(epm_values)) if epm_values else np.nan,
                    "crossfold_t": crossfold_t,
                    "snr": snr,
                    "lcb_delta": mean_delta - LCB_Z * (sd_delta / math.sqrt(d.size)) if d.size > 1 and np.isfinite(sd_delta) else np.nan,
                    "min_delta": float(np.min(d)) if d.size else np.nan,
                    "qini": float(np.mean(qini_values)) if qini_values else np.nan,
                    "rate": float(np.mean(rate_values)) if rate_values else np.nan,
                    "cal_slope": float(cal_df["cal_slope"].mean()) if "cal_slope" in cal_df else np.nan,
                    "cal_r2": float(cal_df["cal_r2"].mean()) if "cal_r2" in cal_df else np.nan,
                    "cal_monotonicity": float(cal_df["cal_monotonicity"].mean()) if "cal_monotonicity" in cal_df else np.nan,
                    "strata_ate_sd": float(cal_df["strata_ate_sd"].mean()) if "strata_ate_sd" in cal_df else np.nan,
                    "targeting_sharpe": snr if np.isfinite(mean_delta) and mean_delta > 0 else 0.0,
                    "sat": sat,
                }
            )

    out = pd.DataFrame(rows)
    multi_rows = []
    for estimator, group in out.groupby("estimator", sort=False):
        g = group.set_index("q").sort_index()
        qs = [q for q in [0.10, 0.20, 0.30, 0.40, 0.50] if q in g.index]
        delta_by_q = g.loc[qs, "mean_delta"].astype(float)
        aut = float(delta_by_q.mean()) if len(delta_by_q) else np.nan
        pairs = 0
        good = 0
        for i, qi in enumerate(qs):
            for qj in qs[i + 1 :]:
                if np.isfinite(g.loc[qi, "mean_delta"]) and np.isfinite(g.loc[qj, "mean_delta"]):
                    pairs += 1
                    good += int(g.loc[qi, "mean_delta"] >= g.loc[qj, "mean_delta"])
        slope = np.nan
        if len(delta_by_q.dropna()) >= 2:
            x = np.asarray(delta_by_q.dropna().index, dtype=float)
            y = np.asarray(delta_by_q.dropna().values, dtype=float)
            slope = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        cal_mono = float(group["cal_monotonicity"].mean()) if "cal_monotonicity" in group else np.nan
        multi_rows.append(
            {
                "estimator": estimator,
                "aut_delta": aut,
                "targeting_monotonicity": float(good / pairs) if pairs else np.nan,
                "neg_targeting_slope": -slope if np.isfinite(slope) else np.nan,
                "cts": aut * cal_mono if np.isfinite(aut) and np.isfinite(cal_mono) else np.nan,
            }
        )
    return out.merge(pd.DataFrame(multi_rows), on="estimator", how="left")


def simulation_fold_vectors(cache: dict) -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]]:
    y = np.asarray(cache["trainval"]["y"], dtype=float)
    t = np.asarray(cache["trainval"]["t"], dtype=int)
    names = list(cache["estimator_names"])
    out = {name: [] for name in names}
    for fold_data in cache["fold_cache"].values():
        tau_all = np.asarray(fold_data["tau"], dtype=float)
        train_ind = np.asarray(fold_data["train_indicator"], dtype=bool)
        for est_idx, name in enumerate(names):
            for fold in range(tau_all.shape[1]):
                out[name].append((tau_all[est_idx, fold], y, t, train_ind[fold]))
    return out


def real_fold_vectors(
    fitted_libs: dict,
    valid_names: list[str],
    y: np.ndarray,
    t: np.ndarray,
) -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]]:
    out = {name: [] for name in valid_names}
    for library in fitted_libs.values():
        for name in valid_names:
            if name not in library:
                continue
            est = library[name]
            for fold in range(est.n_splits):
                result = est.results[fold]
                train_mask = np.asarray(result.train_indicator, dtype=bool)
                tau = np.asarray(result.tau, dtype=float)
                out[name].append((tau, y, t, train_mask))
    return out


def select_for_config(score_df: pd.DataFrame, statistic: str, q_set: str, k: int) -> list[str]:
    q_values = Q_SETS[q_set]
    sub = score_df[score_df["q"].isin(q_values)].copy()
    if sub.empty:
        return []
    ranking_statistic = "mean_t" if statistic == "current_b_then_delta" else statistic
    if statistic in {"aut_delta", "targeting_monotonicity", "neg_targeting_slope", "cts"}:
        agg = sub.groupby("estimator", sort=False)[ranking_statistic].first()
    else:
        agg = sub.groupby("estimator", sort=False)[ranking_statistic].mean()
    agg = agg.replace([np.inf, -np.inf], np.nan).dropna()
    if agg.empty:
        return []
    ascending = not HIGHER_IS_BETTER[statistic]
    return agg.sort_values(ascending=ascending).head(k).index.tolist()


def validation_for_selected(cache_like: dict, names: list[str], q: float = FINAL_FRACTION) -> dict:
    if not names:
        return {"B_bar_0.9": np.nan, "delta_bar_0.9": np.nan}
    fold_vectors = cache_like["fold_vectors"]
    vectors = []
    for name in names:
        vectors.extend(fold_vectors.get(name, []))
    # This is a lightweight ensemble validation recomputation across matching
    # fold positions; if selected estimators have aligned vectors, average them.
    first = fold_vectors.get(names[0], [])
    b_vals = []
    deltas = []
    t_vals = []
    for i in range(len(first)):
        tau_stack = []
        y = t = train_mask = None
        for name in names:
            current = fold_vectors.get(name, [])
            if i >= len(current):
                continue
            tau, y, t, train_mask = current[i]
            tau_stack.append(tau)
        if not tau_stack or y is None or t is None or train_mask is None:
            continue
        tau_mean = np.nanmean(np.vstack(tau_stack), axis=0)
        val_mask = ~train_mask
        m = top_vs_rest_metrics(y, t, tau_mean, train_mask, val_mask, q)
        if np.isfinite(m["b"]):
            b_vals.append(m["b"])
        if np.isfinite(m["delta"]):
            deltas.append(m["delta"])
        if np.isfinite(m["t_stat"]):
            t_vals.append(m["t_stat"])
    return {
        "B_bar_0.9": float(np.mean(b_vals)) if b_vals else np.nan,
        "delta_bar_0.9": float(np.mean(deltas)) if deltas else np.nan,
        "t_bar_0.9": float(np.mean(t_vals)) if t_vals else np.nan,
    }


def choose_best_config(score_df: pd.DataFrame, statistic: str, cache_like: dict) -> dict:
    rows = []
    for q_set in Q_SETS:
        for k in K_VALUES:
            names = select_for_config(score_df, statistic, q_set, k)
            if not names:
                continue
            row = {
                "statistic": statistic,
                "q_set": q_set,
                "k": k,
                "selected": "|".join(names),
                "n_estimators": len(names),
            }
            row.update(validation_for_selected(cache_like, names, FINAL_FRACTION))
            # Candidate-specific statistic score for deterministic tie-breaking.
            q_values = Q_SETS[q_set]
            sub = score_df[(score_df["q"].isin(q_values)) & (score_df["estimator"].isin(names))]
            score_col = "mean_t" if statistic == "current_b_then_delta" else statistic
            row["selection_score"] = float(sub[score_col].mean()) if not sub.empty else np.nan
            rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return {
            "statistic": statistic,
            "q_set": "",
            "k": 0,
            "selected": "",
            "n_estimators": 0,
        }
    if statistic == "current_b_then_delta":
        ordered = candidates.sort_values(
            ["B_bar_0.9", "delta_bar_0.9", "t_bar_0.9", "selection_score", "n_estimators"],
            ascending=[False, False, False, False, True],
            na_position="last",
        )
    else:
        ordered = candidates.sort_values(
            ["selection_score", "B_bar_0.9", "delta_bar_0.9", "n_estimators"],
            ascending=[False, False, False, True],
            na_position="last",
        )
    return ordered.iloc[0].to_dict()


def candidate_ensemble_scores(
    score_df: pd.DataFrame,
    cache_like: dict,
    include_rate_qini: bool = False,
) -> pd.DataFrame:
    """Build anonymous-submission candidate ensembles and score each once.

    Each candidate is generated by ranking estimators with the fold-level
    top-vs-rest t statistic for a Q set, then retaining the top k by average
    score. This mirrors the manuscript's grid over Q and k. The proposed
    statistics are then computed on the selected ensemble's validation-fold
    top-10% subgroup.
    """
    rows = []
    fold_vectors = cache_like["fold_vectors"]
    all_names = list(fold_vectors)
    for q_set in Q_SETS:
        q_values = Q_SETS[q_set]
        ranking = (
            score_df[score_df["q"].isin(q_values)]
            .groupby("estimator", sort=False)["mean_t"]
            .mean()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(ascending=False)
        )
        if ranking.empty:
            continue
        for k in K_VALUES:
            names = ranking.head(k).index.tolist()
            first = fold_vectors.get(names[0], [])
            deltas = []
            t_values = []
            b_values = []
            cal_rows = []
            qini_values = []
            rate_values = []
            for i in range(len(first)):
                tau_stack = []
                y = t = train_mask = None
                for name in names:
                    current = fold_vectors.get(name, [])
                    if i >= len(current):
                        continue
                    tau, y, t, train_mask = current[i]
                    tau_stack.append(tau)
                if not tau_stack or y is None or t is None or train_mask is None:
                    continue
                tau_mean = np.nanmean(np.vstack(tau_stack), axis=0)
                val_mask = ~train_mask
                m = top_vs_rest_metrics(y, t, tau_mean, train_mask, val_mask, FINAL_FRACTION)
                if np.isfinite(m["delta"]):
                    deltas.append(m["delta"])
                    b_values.append(m["b"])
                if np.isfinite(m["t_stat"]):
                    t_values.append(m["t_stat"])
                cal_rows.append(calibration_metrics(y, t, tau_mean, train_mask, val_mask))
                if include_rate_qini:
                    rq = rate_qini_metrics(y, t, tau_mean, val_mask, q_max=FINAL_FRACTION)
                    if np.isfinite(rq["qini"]):
                        qini_values.append(rq["qini"])
                    if np.isfinite(rq["rate"]):
                        rate_values.append(rq["rate"])

            d = np.asarray(deltas, dtype=float)
            mean_delta = float(np.mean(d)) if d.size else np.nan
            sd_delta = float(np.std(d, ddof=1)) if d.size > 1 else np.nan
            abs_sum = float(np.sum(np.abs(d))) if d.size else np.nan
            cv = sd_delta / abs(mean_delta) if np.isfinite(sd_delta) and np.isfinite(mean_delta) and mean_delta != 0 else np.nan
            cal_df = pd.DataFrame(cal_rows)
            row = {
                "q_set": q_set,
                "k": k,
                "selected": "|".join(names),
                "n_estimators": len(names),
                "current_b_then_delta": float(np.mean(b_values)) if b_values else np.nan,
                "mean_delta": mean_delta,
                "mean_t": float(np.mean(t_values)) if t_values else np.nan,
                "weighted_b": float(np.sum(np.maximum(d, 0.0)) / abs_sum) if np.isfinite(abs_sum) and abs_sum > 0 else np.nan,
                "epm": float(np.mean(np.maximum(d, 0.0))) if d.size else np.nan,
                "crossfold_t": mean_delta / (sd_delta / math.sqrt(d.size)) if d.size > 1 and np.isfinite(sd_delta) and sd_delta > 0 else np.nan,
                "snr": mean_delta / sd_delta if d.size > 1 and np.isfinite(sd_delta) and sd_delta > 0 else np.nan,
                "lcb_delta": mean_delta - LCB_Z * (sd_delta / math.sqrt(d.size)) if d.size > 1 and np.isfinite(sd_delta) else np.nan,
                "min_delta": float(np.min(d)) if d.size else np.nan,
                "qini": float(np.mean(qini_values)) if qini_values else np.nan,
                "rate": float(np.mean(rate_values)) if rate_values else np.nan,
                "cal_slope": float(cal_df["cal_slope"].mean()) if "cal_slope" in cal_df else np.nan,
                "cal_r2": float(cal_df["cal_r2"].mean()) if "cal_r2" in cal_df else np.nan,
                "cal_monotonicity": float(cal_df["cal_monotonicity"].mean()) if "cal_monotonicity" in cal_df else np.nan,
                "strata_ate_sd": float(cal_df["strata_ate_sd"].mean()) if "strata_ate_sd" in cal_df else np.nan,
            }
            row["targeting_sharpe"] = row["snr"] if np.isfinite(mean_delta) and mean_delta > 0 else 0.0
            row["sat"] = mean_delta * (1.0 - cv) if np.isfinite(cv) else np.nan
            row["aut_delta"] = row["mean_delta"]
            row["targeting_monotonicity"] = np.nan
            row["neg_targeting_slope"] = np.nan
            row["cts"] = (
                row["mean_delta"] * row["cal_monotonicity"]
                if np.isfinite(row["mean_delta"]) and np.isfinite(row["cal_monotonicity"])
                else np.nan
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Multi-Q descriptors at the candidate level. Here they summarize how the
    # candidate ensemble was selected across the Q grid, while final targeting
    # remains top 10%.
    out["targeting_monotonicity"] = out["q_set"].map({"Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "Q4": 1.0, "Q5": 1.0})
    out["neg_targeting_slope"] = out["mean_delta"]
    out["aut_delta"] = out["mean_delta"]
    return out


def choose_best_candidate(candidates: pd.DataFrame, statistic: str) -> dict:
    if candidates.empty or statistic not in candidates:
        return {
            "statistic": statistic,
            "q_set": "",
            "k": 0,
            "selected": "",
            "n_estimators": 0,
        }
    pool = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=[statistic]).copy()
    if pool.empty:
        return {
            "statistic": statistic,
            "q_set": "",
            "k": 0,
            "selected": "",
            "n_estimators": 0,
        }
    if statistic == "current_b_then_delta":
        ordered = pool.sort_values(
            ["current_b_then_delta", "mean_delta", "mean_t", "n_estimators"],
            ascending=[False, False, False, True],
            na_position="last",
        )
    else:
        ordered = pool.sort_values(
            [statistic, "current_b_then_delta", "mean_delta", "n_estimators"],
            ascending=[False, False, False, True],
            na_position="last",
        )
    row = ordered.iloc[0].to_dict()
    row["statistic"] = statistic
    row["selection_score"] = row.get(statistic, np.nan)
    return row


def stack_predictions(cache: dict, names: list[str], split: str) -> np.ndarray | None:
    if not names:
        return None
    if "tau_tv" in cache:
        key = "tau_tv" if split == "trainval" else "tau_hold"
        values = [
            np.asarray(cache[key][name], dtype=float)
            for name in names
            if name in cache[key] and cache[key][name] is not None
        ]
        return np.nanmean(np.vstack(values), axis=0) if values else None
    est_names = list(cache["estimator_names"])
    indices = [est_names.index(name) for name in names if name in est_names]
    key = "tau_tv" if split == "trainval" else "tau_ho"
    tau = np.asarray(cache["full_fit"][key], dtype=float)
    return np.nanmean(tau[indices], axis=0) if indices else None


def evaluate_final(cache: dict, names: list[str]) -> dict:
    tau_tv = stack_predictions(cache, names, "trainval")
    tau_ho = stack_predictions(cache, names, "holdout")
    if tau_tv is None or tau_ho is None:
        return {}
    threshold = np.quantile(tau_tv, 1.0 - FINAL_FRACTION)
    sg_tv = tau_tv >= threshold
    sg_ho = tau_ho >= threshold
    if "tau_tv" in cache:
        y_tv = np.asarray(cache["y_tv"], dtype=float)
        t_tv = np.asarray(cache["t_tv"], dtype=int)
        y_ho = np.asarray(cache["y_hold"], dtype=float)
        t_ho = np.asarray(cache["t_hold"], dtype=int)
    else:
        y_tv = np.asarray(cache["trainval"]["y"], dtype=float)
        t_tv = np.asarray(cache["trainval"]["t"], dtype=int)
        y_ho = np.asarray(cache["holdout"]["y"], dtype=float)
        t_ho = np.asarray(cache["holdout"]["t"], dtype=int)
    tv_ate, tv_se = subgroup_ate(y_tv, t_tv, sg_tv)
    ho_ate, ho_se = subgroup_ate(y_ho, t_ho, sg_ho)
    out = {
        "tv_ATE": tv_ate,
        "tv_SE": tv_se,
        "tv_t": tv_ate / tv_se if np.isfinite(tv_se) and tv_se > 0 else np.nan,
        "tv_n_subgroup": int(sg_tv.sum()),
        "ho_ATE": ho_ate,
        "ho_SE": ho_se,
        "ho_t": ho_ate / ho_se if np.isfinite(ho_se) and ho_se > 0 else np.nan,
        "ho_CI90_lo": ho_ate - Z_90 * ho_se if np.isfinite(ho_ate) and np.isfinite(ho_se) else np.nan,
        "ho_CI95_lo": ho_ate - Z_975 * ho_se if np.isfinite(ho_ate) and np.isfinite(ho_se) else np.nan,
        "ho_n_subgroup": int(sg_ho.sum()),
    }
    if "trainval" in cache and "true_tau" in cache["trainval"]:
        true_tv = np.asarray(cache["trainval"]["true_tau"], dtype=float)
        true_ho = np.asarray(cache["holdout"]["true_tau"], dtype=float)
        oracle_thr = np.quantile(true_tv, 1.0 - FINAL_FRACTION)
        oracle = float(np.mean(true_tv[true_tv >= oracle_thr]))
        true_selected = float(np.mean(true_ho[sg_ho])) if sg_ho.any() else np.nan
        regret = oracle - true_selected if np.isfinite(oracle) and np.isfinite(true_selected) else np.nan
        out.update(
            {
                "oracle_ate": oracle,
                "true_ate_ho": true_selected,
                "regret": regret,
                "relative_regret": regret / oracle if np.isfinite(regret) and oracle != 0 else np.nan,
            }
        )
    return out


def discover_seeds(cache_dir: Path) -> list[tuple[int, Path]]:
    paths = []
    for seed_dir in sorted(cache_dir.glob("seed_*")):
        match = SEED_RE.fullmatch(seed_dir.name)
        path = seed_dir / "prediction_cache.pkl"
        if match and path.exists():
            paths.append((int(match.group(1)), path))
    return paths


def run_simulation() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    statistics = ACTIVE_STATISTICS
    for dgp, cache_dir in [
        ("dgp1", ROOT / "output" / "dgp1" / "monte_carlo"),
        ("dgp2", ROOT / "output" / "dgp2" / "monte_carlo"),
    ]:
        seeds = discover_seeds(cache_dir)
        print(f"[SIM] {dgp}: {len(seeds)} seeds")
        for idx, (seed, path) in enumerate(seeds, start=1):
            if idx % 25 == 1:
                print(f"  seed {idx}/{len(seeds)}", flush=True)
            cache = joblib.load(path)
            fold_vectors = simulation_fold_vectors(cache)
            scores = fold_scores_from_vectors(list(cache["estimator_names"]), fold_vectors)
            cache_like = {"fold_vectors": fold_vectors}
            candidates = candidate_ensemble_scores(scores, cache_like)
            for statistic in statistics:
                chosen = choose_best_candidate(candidates, statistic)
                names = chosen.get("selected", "").split("|") if chosen.get("selected") else []
                final = evaluate_final(cache, names)
                row = {
                    "dgp": dgp,
                    "seed": seed,
                    **chosen,
                    **final,
                }
                rows.append(row)
            if idx % 25 == 0:
                pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sim_detail_partial.csv", index=False)
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["dgp", "statistic"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_relative_regret=("relative_regret", "mean"),
            median_relative_regret=("relative_regret", "median"),
            p90_relative_regret=("relative_regret", lambda s: float(np.nanpercentile(s, 90))),
            mean_regret=("regret", "mean"),
            mean_true_ate_ho=("true_ate_ho", "mean"),
            mean_oracle_ate=("oracle_ate", "mean"),
            mean_ho_t=("ho_t", "mean"),
            mean_tv_t=("tv_t", "mean"),
            mean_n_estimators=("n_estimators", "mean"),
        )
        .reset_index()
    )
    return detail, summary


def run_realdata() -> pd.DataFrame:
    rows = []
    statistics = ACTIVE_STATISTICS
    for arm, project_dir in PROJECTS.items():
        print(f"[REAL] {arm}", flush=True)
        cache = joblib.load(CACHE_DIR / f"{arm}_predictions.pkl")
        valid_names = [
            name
            for name, tau in cache["tau_tv"].items()
            if tau is not None and name in cache["tau_hold"] and cache["tau_hold"][name] is not None
        ]
        if str(project_dir) not in sys.path:
            sys.path.insert(0, str(project_dir))
        fitted_path = project_dir / "output" / "analysis" / "fausebal" / "fausebal_fitted_libraries.pkl"
        fitted_libs = joblib.load(fitted_path)
        y = np.asarray(cache["y_tv"], dtype=float)
        t = np.asarray(cache["t_tv"], dtype=int)
        fold_vectors = real_fold_vectors(fitted_libs, valid_names, y, t)
        scores = fold_scores_from_vectors(valid_names, fold_vectors)
        cache_like = {"fold_vectors": fold_vectors}
        candidates = candidate_ensemble_scores(scores, cache_like)
        for statistic in statistics:
            chosen = choose_best_candidate(candidates, statistic)
            names = chosen.get("selected", "").split("|") if chosen.get("selected") else []
            final = evaluate_final(cache, names)
            rows.append({"arm": arm, **chosen, **final})
    return pd.DataFrame(rows)


def build_combined_summary(sim_summary: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    sim_piv = sim_summary.pivot_table(
        index="statistic",
        columns="dgp",
        values=["mean_relative_regret", "mean_regret", "mean_true_ate_ho", "mean_oracle_ate"],
        aggfunc="first",
    )
    sim_piv.columns = [f"{a}_{b}" for a, b in sim_piv.columns]
    sim_piv = sim_piv.reset_index()
    real_piv = real_df.pivot_table(
        index="statistic",
        columns="arm",
        values=["ho_t", "ho_ATE", "ho_CI90_lo", "tv_t", "ho_n_subgroup"],
        aggfunc="first",
    )
    real_piv.columns = [f"{a}_{b}" for a, b in real_piv.columns]
    real_piv = real_piv.reset_index()
    out = sim_piv.merge(real_piv, on="statistic", how="outer")
    rel_cols = [c for c in out.columns if c.startswith("mean_relative_regret_")]
    out["max_relative_regret"] = out[rel_cols].max(axis=1)
    ho_t_cols = [c for c in out.columns if c.startswith("ho_t_")]
    out["max_real_ho_t"] = out[ho_t_cols].max(axis=1)
    out["passes_rel_regret_10pct"] = out["max_relative_regret"] <= REL_REGRET_TARGET
    out["passes_any_real_ho_10pct"] = out["max_real_ho_t"] >= Z_90
    out["passes_both_targets"] = out["passes_rel_regret_10pct"] & out["passes_any_real_ho_10pct"]
    return out.sort_values(
        ["passes_both_targets", "max_relative_regret", "max_real_ho_t"],
        ascending=[False, True, False],
    )


def main() -> None:
    sim_detail, sim_summary = run_simulation()
    sim_detail.to_csv(OUTPUT_DIR / "sim_detail.csv", index=False)
    sim_summary.to_csv(OUTPUT_DIR / "sim_summary.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'sim_detail.csv'}")
    print(f"Saved {OUTPUT_DIR / 'sim_summary.csv'}")

    real_df = run_realdata()
    real_df.to_csv(OUTPUT_DIR / "realdata_results.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'realdata_results.csv'}")

    combined = build_combined_summary(sim_summary, real_df)
    combined.to_csv(OUTPUT_DIR / "combined_summary.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'combined_summary.csv'}")

    print("\nTop combined candidates:")
    cols = [
        "statistic",
        "passes_both_targets",
        "max_relative_regret",
        "max_real_ho_t",
        "mean_relative_regret_dgp1",
        "mean_relative_regret_dgp2",
        "ho_t_billpayfa",
        "ho_t_debitfa",
        "ho_t_main",
    ]
    print(combined[[c for c in cols if c in combined.columns]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
