#!/usr/bin/env python3
"""Large selection-statistic bank for the current cal-slope procedure.

This is the simulation-first version of the current JBES procedure:

  * anchor model is selected by Q5 cal_slope;
  * non-anchor models are ordered by Q5 cal_slope;
  * nested arithmetic-mean ensembles are formed along that order;
  * an intentional 40-statistic development bank chooses one ensemble;
  * the statistic is selected by simulation regret only;
  * real data are evaluated once for the frozen simulation winner.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import selection_metrics as base

warnings.filterwarnings("ignore")

OUTPUT_DIR = base.ROOT / "output" / "large_selection_bank_simfirst_40"

OUTCOME = "fausebal"
Q_SET = "Q5"
Q_VALUES = base.Q_SETS[Q_SET]
ANCHOR_STAT = "cal_slope"
ADD_RANK_STAT = "cal_slope"
EPS = 1e-12

ANCHORS = {
    "dgp1": ["s_rf"],
    "dgp2": ["x_rf"],
}

SELECTION_STATS = [
    "weighted_b",
    "mean_delta",
    "mean_delta_top10",
    "lcb_delta_top10",
    "epm",
    "snr",
    "lcb_delta",
    "top_lcb",
    "rate",
    "qini",
    "cal_slope",
    "cal_r2",
    "cal_monotonicity",
    "neg_cal_rmse",
    "strata_ate_sd",
    "top_bin_is_best",
    "neg_targeting_slope",
    "median_delta_top10",
    "trimmed_mean_delta_top10",
    "score_iqr_neg",
    "score_sd_neg",
    "score_mad_neg",
    "score_p90_p10_neg",
    "score_range_neg",
    "top_tail_ratio",
    "bbar_epm",
    "bbar_delta",
    "weighted_b_epm",
    "positive_cal_epm",
    "positive_lcb_epm",
    "positive_cal_bbar_epm",
    "rank_bbar_epm",
    "rank_bbar_delta",
    "rank_bbar_cal_epm",
    "maximin_bbar_epm",
    "cal_slope_x_delta",
    "smooth_delta_ratio",
    "smooth_lcb_ratio",
    "top_balance_delta",
    "stability_x_lcb",
]


def split_selected(selected: str) -> list[str]:
    return selected.split("|") if isinstance(selected, str) and selected else []


def selected_name(names: tuple[str, ...]) -> str:
    return "|".join(names)


def rank01(s: pd.Series) -> pd.Series:
    clean = s.replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() <= 1:
        return pd.Series(np.nan, index=s.index)
    return clean.rank(method="average", pct=True)


def symmetric_trimmed_mean(values: np.ndarray, proportion: float = 0.10) -> float:
    clean = np.sort(values[np.isfinite(values)])
    if clean.size == 0:
        return np.nan
    trim = int(math.floor(proportion * clean.size))
    if trim == 0:
        return float(np.mean(clean))
    return float(np.mean(clean[trim:-trim]))


def fast_rate_qini_metrics(y: np.ndarray, t: np.ndarray, tau: np.ndarray, val_mask: np.ndarray) -> dict[str, float]:
    idx = np.where(val_mask & np.isfinite(tau))[0]
    if idx.size < 10:
        return {"rate": np.nan, "qini": np.nan}
    ordered = idx[np.argsort(-tau[idx], kind="mergesort")]
    y_ord = y[ordered]
    t_ord = t[ordered].astype(int)
    treated = t_ord == 1
    control = ~treated
    cum_t = np.cumsum(treated)
    cum_c = np.cumsum(control)
    cum_y_t = np.cumsum(np.where(treated, y_ord, 0.0))
    cum_y_c = np.cumsum(np.where(control, y_ord, 0.0))
    ate_all, _ = base.subgroup_ate(y, t, val_mask)
    if not np.isfinite(ate_all):
        return {"rate": np.nan, "qini": np.nan}
    rates = []
    qinis = []
    n = len(ordered)
    for frac in np.linspace(0.02, base.FINAL_FRACTION, 10):
        m = max(2, int(math.ceil(frac * n)))
        pos = min(m, n) - 1
        if cum_t[pos] <= 1 or cum_c[pos] <= 1:
            continue
        ate = float(cum_y_t[pos] / cum_t[pos] - cum_y_c[pos] / cum_c[pos])
        if np.isfinite(ate):
            rates.append(ate)
            qinis.append(ate - ate_all)
    return {
        "rate": float(np.mean(rates)) if rates else np.nan,
        "qini": float(np.mean(qinis)) if qinis else np.nan,
    }


def calibration_extra(y: np.ndarray, t: np.ndarray, tau: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray) -> dict[str, float]:
    bins = 5
    q_values = np.linspace(0.0, 1.0, bins + 1)
    tau_means = []
    ates = []
    train_tau = tau[train_mask]
    if train_tau.size == 0:
        return {"neg_cal_rmse": np.nan, "top_bin_is_best": np.nan}
    cutoffs = np.quantile(train_tau, q_values)
    for idx in range(bins):
        lo = cutoffs[idx]
        hi = cutoffs[idx + 1]
        if idx == 0:
            mask = (tau >= lo) & (tau <= hi) & val_mask
        elif idx == bins - 1:
            mask = (tau > lo) & (tau <= hi) & val_mask
        else:
            mask = (tau > lo) & (tau <= hi) & val_mask
        ate, _ = base.subgroup_ate(y, t, mask)
        if np.isfinite(ate) and mask.sum() > 0:
            tau_means.append(float(np.nanmean(tau[mask])))
            ates.append(float(ate))
    if len(ates) < 2 or np.nanstd(tau_means) <= EPS:
        return {"neg_cal_rmse": np.nan, "top_bin_is_best": np.nan}
    x = np.asarray(tau_means, dtype=float)
    z = np.asarray(ates, dtype=float)
    slope = float(np.cov(x, z, ddof=1)[0, 1] / np.var(x, ddof=1))
    pred = z.mean() + slope * (x - x.mean())
    return {
        "neg_cal_rmse": -float(np.sqrt(np.mean((z - pred) ** 2))),
        "top_bin_is_best": float(z[-1] >= np.max(z)),
    }


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


def ranked_models(score_df: pd.DataFrame, anchor: str) -> list[str]:
    scores = aggregate_estimator_scores(score_df, ADD_RANK_STAT)
    scores = scores.drop(labels=[anchor], errors="ignore")
    ordered = scores.sort_values(ascending=False, kind="mergesort")
    return [str(name) for name in ordered.index]


def nested_candidates(estimator_names: list[str], anchor: str, score_df: pd.DataFrame) -> list[tuple[str, ...]]:
    names = sorted(name for name in estimator_names if name)
    if anchor not in names:
        return []
    out = [(anchor,)]
    current = [anchor]
    for name in ranked_models(score_df, anchor):
        if name in names and name not in current:
            current.append(name)
            out.append(tuple(current))
    return out


def candidate_scores(
    names: tuple[str, ...],
    fold_vectors: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]],
) -> dict[str, float]:
    first = fold_vectors.get(names[0], [])
    fold_rows = []
    q_rows = []
    for fold_idx in range(len(first)):
        tau_stack = []
        y = t = train_mask = None
        for name in names:
            current = fold_vectors.get(name, [])
            if fold_idx >= len(current):
                continue
            tau, y, t, train_mask = current[fold_idx]
            tau_stack.append(tau)
        if len(tau_stack) != len(names) or y is None or t is None or train_mask is None:
            continue
        tau_mean = np.nanmean(np.vstack(tau_stack), axis=0)
        val_mask = ~train_mask
        val_tau = tau_mean[val_mask & np.isfinite(tau_mean)]
        if val_tau.size == 0:
            continue

        for q in Q_VALUES:
            top = base.top_mask_from_train_quantile(tau_mean, train_mask, val_mask, q)
            metrics = base.top_vs_rest_metrics(y, t, tau_mean, train_mask, val_mask, q)
            top_ate, top_se = base.subgroup_ate(y, t, top)
            cal = base.calibration_metrics(y, t, tau_mean, train_mask, val_mask)
            cal.update(calibration_extra(y, t, tau_mean, train_mask, val_mask))
            rq = fast_rate_qini_metrics(y, t, tau_mean, val_mask)
            q_rows.append(
                {
                    "q": q,
                    "delta": metrics["delta"],
                    "b": metrics["b"],
                    "top_lcb": top_ate - base.Z_90 * top_se if np.isfinite(top_ate) and np.isfinite(top_se) else np.nan,
                    "rate": rq["rate"],
                    "qini": rq["qini"],
                    **cal,
                }
            )

        top10 = base.top_mask_from_train_quantile(tau_mean, train_mask, val_mask, base.FINAL_FRACTION)
        fold_rows.append(
            {
                "score_iqr": float(np.nanpercentile(val_tau, 75) - np.nanpercentile(val_tau, 25)),
                "score_sd": float(np.nanstd(val_tau, ddof=1)) if val_tau.size > 1 else np.nan,
                "score_mad": float(np.nanmedian(np.abs(val_tau - np.nanmedian(val_tau)))),
                "score_p90_p10": float(np.nanpercentile(val_tau, 90) - np.nanpercentile(val_tau, 10)),
                "score_range": float(np.nanmax(val_tau) - np.nanmin(val_tau)),
                "top_tail_ratio": float(np.nanmean(tau_mean[top10]) / max(abs(np.nanmean(val_tau)), EPS)) if np.any(top10) else np.nan,
            }
        )

    q_df = pd.DataFrame(q_rows)
    fold_df = pd.DataFrame(fold_rows)
    if q_df.empty:
        return {}

    d = q_df["delta"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    top10_df = q_df[q_df["q"] == base.FINAL_FRACTION]
    d10 = top10_df["delta"].to_numpy(dtype=float)
    d10 = d10[np.isfinite(d10)]
    mean_delta = float(np.mean(d)) if d.size else np.nan
    sd_delta = float(np.std(d, ddof=1)) if d.size > 1 else np.nan
    mean_delta10 = float(np.mean(d10)) if d10.size else np.nan
    sd_delta10 = float(np.std(d10, ddof=1)) if d10.size > 1 else np.nan
    abs_sum = float(np.sum(np.abs(d))) if d.size else np.nan
    bbar_component = float(np.nanmean(q_df["b"]))
    stability_component = float(max(bbar_component, 1.0 - bbar_component))

    out = {
        "_bbar_component": bbar_component,
        "weighted_b": float(np.sum(np.maximum(d, 0.0)) / abs_sum) if np.isfinite(abs_sum) and abs_sum > 0 else np.nan,
        "mean_delta": mean_delta,
        "epm": float(np.mean(np.maximum(d, 0.0))) if d.size else np.nan,
        "snr": mean_delta / sd_delta if d.size > 1 and np.isfinite(sd_delta) and sd_delta > 0 else np.nan,
        "lcb_delta": mean_delta - base.LCB_Z * (sd_delta / math.sqrt(d.size)) if d.size > 1 and np.isfinite(sd_delta) else np.nan,
        "top_lcb": float(np.nanmean(q_df["top_lcb"])),
        "rate": float(np.nanmean(q_df["rate"])),
        "qini": float(np.nanmean(q_df["qini"])),
        "cal_slope": float(np.nanmean(q_df["cal_slope"])),
        "cal_r2": float(np.nanmean(q_df["cal_r2"])),
        "cal_monotonicity": float(np.nanmean(q_df["cal_monotonicity"])),
        "neg_cal_rmse": float(np.nanmean(q_df["neg_cal_rmse"])),
        "strata_ate_sd": float(np.nanmean(q_df["strata_ate_sd"])),
        "top_bin_is_best": float(np.nanmean(q_df["top_bin_is_best"])),
        "_stability_component": stability_component,
        "mean_delta_top10": mean_delta10,
        "median_delta_top10": float(np.median(d10)) if d10.size else np.nan,
        "trimmed_mean_delta_top10": symmetric_trimmed_mean(d10),
        "lcb_delta_top10": (
            mean_delta10 - base.LCB_Z * (sd_delta10 / math.sqrt(d10.size))
            if d10.size > 1 and np.isfinite(sd_delta10)
            else np.nan
        ),
    }
    for col in ["score_iqr", "score_sd", "score_mad", "score_p90_p10", "score_range", "top_tail_ratio"]:
        out[col] = float(np.nanmean(fold_df[col])) if not fold_df.empty else np.nan
        out[f"{col}_neg"] = -out[col] if np.isfinite(out[col]) else np.nan

    delta_by_q = q_df.groupby("q")["delta"].mean().sort_index()
    if len(delta_by_q.dropna()) >= 2:
        x = delta_by_q.dropna().index.to_numpy(dtype=float)
        z = delta_by_q.dropna().to_numpy(dtype=float)
        out["neg_targeting_slope"] = -float(np.cov(x, z, ddof=1)[0, 1] / np.var(x, ddof=1))
    else:
        out["neg_targeting_slope"] = np.nan

    return add_composites(out)


def add_composites(row: dict[str, float]) -> dict[str, float]:
    bbar = row.get("_bbar_component", np.nan)
    epm = row.get("epm", np.nan)
    delta = row.get("mean_delta", np.nan)
    weighted_b = row.get("weighted_b", np.nan)
    cal_pos = max(row.get("cal_slope", np.nan), 0.0) if np.isfinite(row.get("cal_slope", np.nan)) else np.nan
    lcb_pos = max(row.get("lcb_delta", np.nan), 0.0) if np.isfinite(row.get("lcb_delta", np.nan)) else np.nan
    stability = row.get("_stability_component", np.nan)
    smooth = 1.0 / (abs(row.get("score_iqr", np.nan)) + EPS) if np.isfinite(row.get("score_iqr", np.nan)) else np.nan
    row["bbar_epm"] = bbar * epm
    row["bbar_delta"] = bbar * delta
    row["weighted_b_epm"] = weighted_b * epm
    row["positive_cal_epm"] = cal_pos * epm
    row["positive_lcb_epm"] = lcb_pos * epm
    row["positive_cal_bbar_epm"] = cal_pos * bbar * epm
    row["cal_slope_x_delta"] = cal_pos * delta
    row["smooth_delta_ratio"] = delta * smooth
    row["smooth_lcb_ratio"] = lcb_pos * smooth
    row["top_balance_delta"] = stability * delta
    row["stability_x_lcb"] = stability * lcb_pos
    return row


def score_nested_candidates(
    estimator_names: list[str],
    fold_vectors: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]],
    score_df: pd.DataFrame,
    anchor: str,
) -> pd.DataFrame:
    rows = []
    for names in nested_candidates(estimator_names, anchor, score_df):
        row = {
            "selected": selected_name(names),
            "n_estimators": len(names),
            "anchor_model": anchor,
            "add_k": len(names) - 1,
            "added_models": "|".join([name for name in names if name != anchor]),
        }
        row.update(candidate_scores(names, fold_vectors))
        rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates["rank_bbar_epm"] = (rank01(candidates["_bbar_component"]) + rank01(candidates["epm"])) / 2.0
    candidates["rank_bbar_delta"] = (rank01(candidates["_bbar_component"]) + rank01(candidates["mean_delta"])) / 2.0
    candidates["rank_bbar_cal_epm"] = (
        rank01(candidates["_bbar_component"]) + rank01(candidates["cal_slope"]) + rank01(candidates["epm"])
    ) / 3.0
    candidates["maximin_bbar_epm"] = pd.concat(
        [rank01(candidates["_bbar_component"]), rank01(candidates["epm"])],
        axis=1,
    ).min(axis=1)
    return candidates


def choose_candidate(candidates: pd.DataFrame, statistic: str) -> dict:
    if candidates.empty or statistic not in candidates:
        return {"selection_stat": statistic, "selected": "", "n_estimators": 0, "selection_score": np.nan}
    pool = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=[statistic]).copy()
    if pool.empty:
        return {"selection_stat": statistic, "selected": "", "n_estimators": 0, "selection_score": np.nan}
    ordered = pool.sort_values(
        [statistic, "n_estimators", "selected"],
        ascending=[False, True, True],
        kind="mergesort",
        na_position="last",
    )
    row = ordered.iloc[0].to_dict()
    row["selection_stat"] = statistic
    row["selection_score"] = row.get(statistic, np.nan)
    return row


def procedure_label(selection_stat: str) -> str:
    return f"{Q_SET}__{ANCHOR_STAT}_anchor__rank_{ADD_RANK_STAT}__select_{selection_stat}"


def run_simulation(max_seeds: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for dgp, cache_dir in [
        ("dgp1", base.ROOT / "output" / "dgp1" / "monte_carlo"),
        ("dgp2", base.ROOT / "output" / "dgp2" / "monte_carlo"),
    ]:
        seeds = base.discover_seeds(cache_dir)
        if max_seeds is not None:
            seeds = seeds[:max_seeds]
        print(f"[SIM] {dgp}: {len(seeds)} seeds", flush=True)
        for idx, (seed, path) in enumerate(seeds, start=1):
            if idx % 10 == 1:
                print(f"  seed {idx}/{len(seeds)}", flush=True)
            cache = joblib.load(path)
            fold_vectors = base.simulation_fold_vectors(cache)
            estimator_names = list(cache["estimator_names"])
            scores = base.fold_scores_from_vectors(estimator_names, fold_vectors)
            anchor_model, anchor_score = select_anchor(scores)
            candidates = score_nested_candidates(estimator_names, fold_vectors, scores, anchor_model)
            benchmark = base.evaluate_final(cache, ANCHORS[dgp]).get("true_ate_ho", np.nan)
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
                        "procedure": procedure_label(statistic),
                        "anchor_stat": ANCHOR_STAT,
                        "anchor_score": anchor_score,
                        "add_rank_stat": ADD_RANK_STAT,
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
            if idx % 10 == 0:
                pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sim_detail_partial.csv", index=False)
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["procedure", "selection_stat", "dgp"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_anchor_relative_regret=("anchor_relative_regret", "mean"),
            median_anchor_relative_regret=("anchor_relative_regret", "median"),
            p90_anchor_relative_regret=("anchor_relative_regret", lambda s: float(np.nanpercentile(s, 90))),
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
    winners = freeze_simulation_winner(summary)
    return detail, summary, winners


def freeze_simulation_winner(sim_summary: pd.DataFrame) -> pd.DataFrame:
    piv = sim_summary.pivot_table(
        index=["procedure", "selection_stat"],
        columns="dgp",
        values=[
            "mean_anchor_relative_regret",
            "mean_anchor_regret",
            "mean_true_ate_ho",
            "mean_benchmark_true_ate",
            "mean_true_oracle_relative_regret",
            "mean_n_estimators",
        ],
        aggfunc="first",
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    out = piv.reset_index()
    rel_cols = [c for c in out.columns if c.startswith("mean_anchor_relative_regret_")]
    reg_cols = [c for c in out.columns if c.startswith("mean_anchor_regret_")]
    n_cols = [c for c in out.columns if c.startswith("mean_n_estimators_")]
    out["avg_anchor_relative_regret"] = out[rel_cols].mean(axis=1)
    out["max_anchor_relative_regret"] = out[rel_cols].max(axis=1)
    out["avg_anchor_regret"] = out[reg_cols].mean(axis=1)
    out["avg_n_estimators"] = out[n_cols].mean(axis=1)
    out["passes_anchor_rel_10pct"] = out["max_anchor_relative_regret"] <= base.REL_REGRET_TARGET
    return out.sort_values(
        [
            "avg_anchor_relative_regret",
            "max_anchor_relative_regret",
            "avg_anchor_regret",
            "avg_n_estimators",
            "procedure",
        ],
        ascending=[True, True, True, True, True],
    )


def run_realdata_for_winner(winner_row: pd.Series) -> pd.DataFrame:
    import screen_anchor_augmented_procedure as current

    selection_stat = str(winner_row["selection_stat"])
    rows = []
    for arm, project_dir in base.PROJECTS.items():
        print(f"[REAL VERIFY] {arm}", flush=True)
        data = current.load_real_project(project_dir)
        fitted_libs = data["fitted_libs"]
        trainval_df = data["trainval_df"]
        holdout_df = data["holdout_df"]
        features = data["features"]
        treatment_var = data["treatment_var"]
        valid_names = list(fitted_libs["pert_none"].keys())
        y = trainval_df[OUTCOME].to_numpy(dtype=float)
        t = current.treatment_array(trainval_df, treatment_var)
        fold_vectors = base.real_fold_vectors(fitted_libs, valid_names, y, t)
        scores = base.fold_scores_from_vectors(valid_names, fold_vectors)
        anchor_model, anchor_score = select_anchor(scores)
        candidates = score_nested_candidates(valid_names, fold_vectors, scores, anchor_model)
        chosen = choose_candidate(candidates, selection_stat)
        final = current.evaluate_real_final(
            project_dir,
            fitted_libs,
            trainval_df,
            holdout_df,
            features,
            treatment_var,
            split_selected(chosen["selected"]),
            {},
        )
        rows.append(
            {
                "arm": arm,
                "procedure": procedure_label(selection_stat),
                "anchor_stat": ANCHOR_STAT,
                "anchor_score": anchor_score,
                "add_rank_stat": ADD_RANK_STAT,
                **chosen,
                **final,
            }
        )
    return pd.DataFrame(rows)


def build_combined_summary(winners: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    top = winners.head(1).copy()
    real_piv = real_df.pivot_table(
        index=["procedure", "selection_stat"],
        columns="arm",
        values=["ho_t", "ho_ATE", "ho_CI90_lo", "tv_t", "ho_n_subgroup", "selected"],
        aggfunc="first",
    )
    real_piv.columns = [f"{a}_{b}" for a, b in real_piv.columns]
    real_piv = real_piv.reset_index()
    out = top.merge(real_piv, on=["procedure", "selection_stat"], how="left")
    ho_cols = [c for c in out.columns if c.startswith("ho_t_")]
    out["max_real_ho_t"] = out[ho_cols].max(axis=1)
    out["passes_any_real_ho_10pct"] = out["max_real_ho_t"] >= base.Z_90
    return out


def parse_args() -> tuple[int | None, Path]:
    global OUTPUT_DIR
    max_seeds = None
    output_dir = OUTPUT_DIR
    for arg in sys.argv[1:]:
        if arg == "--smoke":
            max_seeds = 2
            output_dir = base.ROOT / "output" / "large_selection_bank_smoke"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise SystemExit(f"Unknown argument: {arg}")
    OUTPUT_DIR = output_dir
    return max_seeds, output_dir


def main() -> None:
    max_seeds, output_dir = parse_args()
    sim_detail, sim_summary, winners = run_simulation(max_seeds=max_seeds)
    sim_detail.to_csv(output_dir / "sim_detail.csv", index=False)
    sim_summary.to_csv(output_dir / "sim_summary.csv", index=False)
    winners.to_csv(output_dir / "simulation_winner_ranking.csv", index=False)
    print(f"Saved {output_dir / 'sim_detail.csv'}")
    print(f"Saved {output_dir / 'sim_summary.csv'}")
    print(f"Saved {output_dir / 'simulation_winner_ranking.csv'}")
    cols = [
        "procedure",
        "avg_anchor_relative_regret",
        "max_anchor_relative_regret",
        "mean_anchor_relative_regret_dgp1",
        "mean_anchor_relative_regret_dgp2",
        "avg_n_estimators",
    ]
    print("\nTop simulation-first selection statistics:")
    print(winners[[c for c in cols if c in winners.columns]].head(20).to_string(index=False))
    if winners.empty:
        return
    real_df = run_realdata_for_winner(winners.iloc[0])
    real_df.to_csv(output_dir / "realdata_verification.csv", index=False)
    combined = build_combined_summary(winners, real_df)
    combined.to_csv(output_dir / "combined_frozen_winner.csv", index=False)
    print(f"Saved {output_dir / 'realdata_verification.csv'}")
    print(f"Saved {output_dir / 'combined_frozen_winner.csv'}")
    print("\nFrozen winner real-data verification:")
    print(real_df.to_string(index=False))


if __name__ == "__main__":
    main()
