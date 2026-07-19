#!/usr/bin/env python3
"""Full review of the large selection-statistic bank.

The input simulation bank is produced by screen_large_selection_bank_simfirst.py.
This script re-evaluates every statistic in that bank on the real data, reusing
full-fit predictions within each arm, and writes a merged review table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_proposed_targeting_statistics as base
import screen_anchor_augmented_procedure as current
import screen_large_selection_bank_simfirst as bank

OUTPUT_DIR = base.ROOT / "output" / "large_selection_bank_simfirst_40"
REAL_OUT = OUTPUT_DIR / "realdata_all_stats_refit_audit.csv"
REVIEW_OUT = OUTPUT_DIR / "simulation_rank_with_real_refit_audit.csv"


def run_all_realdata_refit() -> pd.DataFrame:
    rows = []
    for arm, project_dir in base.PROJECTS.items():
        print(f"[REAL ALL] {arm}", flush=True)
        data = current.load_real_project(project_dir)
        fitted_libs = data["fitted_libs"]
        trainval_df = data["trainval_df"]
        holdout_df = data["holdout_df"]
        features = data["features"]
        treatment_var = data["treatment_var"]
        valid_names = list(fitted_libs["pert_none"].keys())

        y = trainval_df[bank.OUTCOME].to_numpy(dtype=float)
        t = current.treatment_array(trainval_df, treatment_var)
        fold_vectors = base.real_fold_vectors(fitted_libs, valid_names, y, t)
        scores = base.fold_scores_from_vectors(valid_names, fold_vectors)
        anchor_model, anchor_score = bank.select_anchor(scores)
        candidates = bank.score_nested_candidates(valid_names, fold_vectors, scores, anchor_model)
        prediction_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for statistic in bank.SELECTION_STATS:
            chosen = bank.choose_candidate(candidates, statistic)
            final = current.evaluate_real_final(
                project_dir,
                fitted_libs,
                trainval_df,
                holdout_df,
                features,
                treatment_var,
                bank.split_selected(chosen["selected"]),
                prediction_cache,
            )
            rows.append(
                {
                    "arm": arm,
                    "procedure": bank.procedure_label(statistic),
                    "anchor_stat": bank.ANCHOR_STAT,
                    "anchor_score": anchor_score,
                    "add_rank_stat": bank.ADD_RANK_STAT,
                    **chosen,
                    **final,
                }
            )
    return pd.DataFrame(rows)


def summarize_real_arms(row: pd.Series) -> str:
    arms = []
    for arm in base.PROJECTS:
        value = row.get(f"ho_t_{arm}", np.nan)
        if np.isfinite(value) and value >= base.Z_90:
            arms.append(arm)
    return "|".join(arms)


def build_review(real_df: pd.DataFrame) -> pd.DataFrame:
    sim = pd.read_csv(OUTPUT_DIR / "simulation_winner_ranking.csv").copy()
    sim["avg_regret_rank"] = sim["avg_anchor_relative_regret"].rank(method="min", ascending=True).astype(int)
    sim["max_regret_rank"] = sim["max_anchor_relative_regret"].rank(method="min", ascending=True).astype(int)
    sim["passes_anchor_rel_10pct"] = sim["max_anchor_relative_regret"] <= base.REL_REGRET_TARGET

    real_piv = real_df.pivot_table(
        index=["procedure", "selection_stat"],
        columns="arm",
        values=[
            "selected",
            "n_estimators",
            "ho_ATE",
            "ho_SE",
            "ho_t",
            "ho_CI90_lo",
            "ho_n_subgroup",
            "tv_t",
        ],
        aggfunc="first",
    )
    real_piv.columns = [f"{metric}_{arm}" for metric, arm in real_piv.columns]
    real_piv = real_piv.reset_index()

    out = sim.merge(real_piv, on=["procedure", "selection_stat"], how="left")
    ho_cols = [f"ho_t_{arm}" for arm in base.PROJECTS if f"ho_t_{arm}" in out.columns]
    out["max_real_ho_t"] = out[ho_cols].max(axis=1)
    out["significant_real_arms_10pct"] = out.apply(summarize_real_arms, axis=1)
    out["passes_any_real_ho_10pct"] = out["significant_real_arms_10pct"].astype(bool)
    out["passes_both_targets"] = out["passes_anchor_rel_10pct"] & out["passes_any_real_ho_10pct"]
    return out.sort_values(
        [
            "passes_both_targets",
            "avg_anchor_relative_regret",
            "max_anchor_relative_regret",
            "selection_stat",
        ],
        ascending=[False, True, True, True],
    )


def print_summary(review: pd.DataFrame) -> None:
    print(f"\nBank size: {len(review)}")
    print(f"Pass simulation max-regret <= 10%: {int(review['passes_anchor_rel_10pct'].sum())}")
    print(f"Pass any positive real holdout 10% test: {int(review['passes_any_real_ho_10pct'].sum())}")
    print(f"Pass both: {int(review['passes_both_targets'].sum())}")

    cols = [
        "selection_stat",
        "avg_regret_rank",
        "max_regret_rank",
        "avg_anchor_relative_regret",
        "max_anchor_relative_regret",
        "mean_anchor_relative_regret_dgp1",
        "mean_anchor_relative_regret_dgp2",
        "max_real_ho_t",
        "significant_real_arms_10pct",
    ]
    print("\nStats passing both targets:")
    both = review[review["passes_both_targets"]]
    print(both[cols].to_string(index=False) if not both.empty else "(none)")

    print("\nTop 15 by average simulation regret:")
    print(
        review.sort_values(["avg_anchor_relative_regret", "max_anchor_relative_regret"])[cols]
        .head(15)
        .to_string(index=False)
    )

    focus = ["mean_delta_top10", "cal_slope", "neg_targeting_slope", "qini", "rate"]
    print("\nFocus statistics:")
    print(review[review["selection_stat"].isin(focus)].sort_values("avg_regret_rank")[cols].to_string(index=False))


def main() -> None:
    real_df = run_all_realdata_refit()
    real_df.to_csv(REAL_OUT, index=False)
    print(f"Saved {REAL_OUT}", flush=True)

    review = build_review(real_df)
    review.to_csv(REVIEW_OUT, index=False)
    print(f"Saved {REVIEW_OUT}", flush=True)
    print_summary(review)


if __name__ == "__main__":
    main()
