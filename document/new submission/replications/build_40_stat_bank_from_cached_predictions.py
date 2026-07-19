#!/usr/bin/env python3
"""Build a 40-rule bank using saved Monte Carlo prediction objects only.

The 39-rule simulation output is treated as immutable. Two algebraically
redundant source labels are replaced by robust top-decile criteria, and
``lcb_delta_top10`` is added. No CATE estimator is fitted here.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import evaluate_proposed_targeting_statistics as base
import screen_large_selection_bank_simfirst as bank


SOURCE_DIR = base.ROOT / "output" / "large_selection_bank_simfirst"
OUTPUT_DIR = base.ROOT / "output" / "large_selection_bank_simfirst_40"
STATISTICS = [
    "lcb_delta_top10",
    "median_delta_top10",
    "trimmed_mean_delta_top10",
]
SOURCE_STATS_REPLACED = {"aut_delta", "stability_x_delta"}


def score_robust_candidates(
    estimator_names: list[str],
    fold_vectors: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]],
    scores: pd.DataFrame,
    anchor_model: str,
) -> pd.DataFrame:
    rows = []
    for names in bank.nested_candidates(estimator_names, anchor_model, scores):
        deltas = []
        first = fold_vectors.get(names[0], [])
        for fold_idx in range(len(first)):
            tau_stack = []
            y = t = train_mask = None
            for name in names:
                tau, y, t, train_mask = fold_vectors[name][fold_idx]
                tau_stack.append(tau)
            tau_mean = np.mean(np.vstack(tau_stack), axis=0)
            val_mask = ~train_mask
            metrics = base.top_vs_rest_metrics(
                y,
                t,
                tau_mean,
                train_mask,
                val_mask,
                base.FINAL_FRACTION,
            )
            if np.isfinite(metrics["delta"]):
                deltas.append(float(metrics["delta"]))
        d = np.asarray(deltas, dtype=float)
        mean_delta = float(np.mean(d)) if d.size else np.nan
        sd_delta = float(np.std(d, ddof=1)) if d.size > 1 else np.nan
        lcb = (
            mean_delta - base.LCB_Z * sd_delta / np.sqrt(d.size)
            if d.size > 1 and np.isfinite(sd_delta)
            else np.nan
        )
        rows.append(
            {
                "selected": bank.selected_name(names),
                "n_estimators": len(names),
                "anchor_model": anchor_model,
                "add_k": len(names) - 1,
                "added_models": "|".join(names[1:]),
                "mean_delta_top10": mean_delta,
                "lcb_delta_top10": lcb,
                "median_delta_top10": float(np.median(d)) if d.size else np.nan,
                "trimmed_mean_delta_top10": bank.symmetric_trimmed_mean(d),
            }
        )
    return pd.DataFrame(rows)


def evaluate_cached_predictions() -> pd.DataFrame:
    rows = []
    for dgp, cache_dir in [
        ("dgp1", base.ROOT / "output" / "dgp1" / "monte_carlo"),
        ("dgp2", base.ROOT / "output" / "dgp2" / "monte_carlo"),
    ]:
        seeds = base.discover_seeds(cache_dir)
        print(f"[CACHED] {dgp}: {len(seeds)} fitted prediction objects", flush=True)
        for idx, (seed, path) in enumerate(seeds, start=1):
            cache = joblib.load(path)
            estimator_names = list(cache["estimator_names"])
            if len(estimator_names) != 23:
                raise ValueError(f"{dgp} seed {seed}: expected 23 estimators, found {len(estimator_names)}")
            fold_vectors = base.simulation_fold_vectors(cache)
            scores = base.fold_scores_from_vectors(estimator_names, fold_vectors)
            anchor_model, anchor_score = bank.select_anchor(scores)
            candidates = score_robust_candidates(
                estimator_names,
                fold_vectors,
                scores,
                anchor_model,
            )
            benchmark = base.evaluate_final(cache, bank.ANCHORS[dgp]).get("true_ate_ho", np.nan)
            for statistic in STATISTICS:
                chosen = bank.choose_candidate(candidates, statistic)
                final = base.evaluate_final(cache, bank.split_selected(chosen["selected"]))
                anchor_regret = (
                    benchmark - final.get("true_ate_ho", np.nan)
                    if np.isfinite(benchmark)
                    else np.nan
                )
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
                        "anchor_regret": anchor_regret,
                        "anchor_relative_regret": (
                            anchor_regret / benchmark
                            if np.isfinite(anchor_regret) and benchmark != 0
                            else np.nan
                        ),
                    }
                )
            if idx % 25 == 0:
                print(f"  processed {idx}/{len(seeds)}", flush=True)
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["procedure", "selection_stat", "dgp"], dropna=False)
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


def main() -> None:
    baseline = pd.read_csv(SOURCE_DIR / "sim_detail.csv")
    source_stats = set(baseline["selection_stat"].dropna())
    if len(source_stats) != 39 or not SOURCE_STATS_REPLACED.issubset(source_stats):
        raise ValueError("The source output is not the expected immutable 39-rule bank")
    expected_rows = 39 * 199
    if len(baseline) != expected_rows:
        raise ValueError(f"Expected {expected_rows} source rows, found {len(baseline)}")

    baseline = baseline[~baseline["selection_stat"].isin(SOURCE_STATS_REPLACED)].copy()
    if baseline["selection_stat"].nunique() != 37:
        raise ValueError("The source-label replacement did not retain exactly 37 rules")

    added = evaluate_cached_predictions()
    if len(added) != 3 * 199 or set(added["selection_stat"]) != set(STATISTICS):
        raise ValueError("The cached-prediction evaluation did not produce 597 rows")

    detail = pd.concat([baseline, added], ignore_index=True, sort=False)
    summary = summarize(detail)
    winners = bank.freeze_simulation_winner(summary)
    if detail["selection_stat"].nunique() != 40 or len(winners) != 40:
        raise ValueError("The merged output does not contain exactly 40 statistics")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUTPUT_DIR / "sim_detail.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "sim_summary.csv", index=False)
    winners.to_csv(OUTPUT_DIR / "simulation_winner_ranking.csv", index=False)

    current_real = pd.read_csv(
        base.ROOT / "output" / "anchor_expanded_procedure" / "realdata_results.csv"
    )
    current_real.to_csv(OUTPUT_DIR / "realdata_fixed_mean_delta_top10.csv", index=False)

    print(f"Saved 40-rule outputs to {OUTPUT_DIR}")
    print(winners.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
