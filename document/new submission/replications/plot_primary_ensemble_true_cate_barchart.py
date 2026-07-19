#!/usr/bin/env python3
"""Plot the primary nested ensemble against the 23 individual estimators."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

import evaluate_proposed_targeting_statistics as base


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parents[1] / "new submission" / "figs"
PRIMARY_DETAIL = ROOT / "output" / "anchor_expanded_procedure" / "sim_detail.csv"
FIG_BASENAME = "mc_true_cate_top10_primary_ensemble_vs_estimators"
HIGHLIGHT_LABEL = "Selected calibration-slope nested ensemble"

ESTIMATOR_LABELS = {
    "s_rf": "S-learner (RF)",
    "s_xgb": "S-learner (XGB)",
    "t_lasso": "T-learner (Lasso)",
    "t_logistic": "T-learner (Logistic)",
    "t_rf": "T-learner (RF)",
    "t_xgb": "T-learner (XGB)",
    "x_lasso": "X-learner (Lasso)",
    "x_logistic": "X-learner (Logistic)",
    "x_rf": "X-learner (RF)",
    "x_xgb": "X-learner (XGB)",
    "r_lassolasso": "R-learner (Lasso/Lasso)",
    "r_lassorf": "R-learner (Lasso/RF)",
    "r_lassoxgb": "R-learner (Lasso/XGB)",
    "r_rflasso": "R-learner (RF/Lasso)",
    "r_rfrf": "R-learner (RF/RF)",
    "r_rfxgb": "R-learner (RF/XGB)",
    "r_xgblasso": "R-learner (XGB/Lasso)",
    "r_xgbrf": "R-learner (XGB/RF)",
    "r_xgbxgb": "R-learner (XGB/XGB)",
    "causal_tree_1": "Causal tree (leaf 500)",
    "causal_tree_2": "Causal tree (leaf 2,000)",
    "causal_forest_1": "Causal forest (leaf 500)",
    "causal_forest_2": "Causal forest (leaf 2,000)",
}


def holdout_true_cate_for_estimator(cache: dict, estimator_index: int) -> float:
    tau_tv = np.asarray(cache["full_fit"]["tau_tv"][estimator_index], dtype=float)
    tau_ho = np.asarray(cache["full_fit"]["tau_ho"][estimator_index], dtype=float)
    true_ho = np.asarray(cache["holdout"]["true_tau"], dtype=float)
    threshold = np.quantile(tau_tv, 1.0 - base.FINAL_FRACTION)
    subgroup = tau_ho >= threshold
    return float(np.mean(true_ho[subgroup])) if subgroup.any() else np.nan


def load_individual_rows(primary_seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seed_sets = {
        dgp: set(group["seed"].astype(int))
        for dgp, group in primary_seeds.groupby("dgp")
    }
    for dgp in ["dgp1", "dgp2"]:
        for seed, path in base.discover_seeds(ROOT / "output" / dgp / "monte_carlo"):
            if seed not in seed_sets[dgp]:
                continue
            cache = joblib.load(path)
            for estimator_index, estimator in enumerate(cache["estimator_names"]):
                rows.append(
                    {
                        "dgp": dgp,
                        "seed": seed,
                        "method": estimator,
                        "display_label": ESTIMATOR_LABELS[estimator],
                        "method_type": "Individual estimator",
                        "true_cate_ho": holdout_true_cate_for_estimator(
                            cache, estimator_index
                        ),
                    }
                )
    return pd.DataFrame(rows)


def load_rows() -> pd.DataFrame:
    primary = pd.read_csv(PRIMARY_DETAIL)[["dgp", "seed", "true_ate_ho"]].copy()
    primary["method"] = "primary_ensemble"
    primary["display_label"] = HIGHLIGHT_LABEL
    primary["method_type"] = "Selected ensemble"

    individual = load_individual_rows(primary[["dgp", "seed"]])

    primary = primary.rename(columns={"true_ate_ho": "true_cate_ho"})
    columns = ["dgp", "seed", "method", "display_label", "method_type", "true_cate_ho"]
    return pd.concat([individual[columns], primary[columns]], ignore_index=True)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["dgp", "method", "display_label", "method_type"], sort=False)
        .agg(
            n=("seed", "nunique"),
            mean_true_cate=("true_cate_ho", "mean"),
            sd_true_cate=("true_cate_ho", "std"),
        )
        .reset_index()
    )
    summary["se_true_cate"] = summary["sd_true_cate"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se_true_cate"]
    summary["rank_within_dgp"] = summary.groupby("dgp")["mean_true_cate"].rank(
        ascending=False, method="min"
    ).astype(int)
    summary["n_methods_ranked"] = summary.groupby("dgp")["method"].transform("size")
    return summary


def plot(summary: pd.DataFrame) -> tuple[Path, Path]:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    fig, axes = plt.subplots(ncols=2, figsize=(12.0, 8.2), sharex=False)

    for ax, dgp in zip(axes, ["dgp1", "dgp2"]):
        plot_df = summary[summary["dgp"].eq(dgp)].sort_values(
            ["mean_true_cate", "display_label"], ascending=[True, True]
        )
        y_pos = np.arange(len(plot_df))
        selected = plot_df["method_type"].eq("Selected ensemble")
        colors = np.where(selected, "crimson", "steelblue")
        ax.barh(
            y_pos,
            plot_df["mean_true_cate"],
            xerr=plot_df["ci95"],
            color=colors,
            edgecolor=colors,
            alpha=0.88,
            height=0.72,
            error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2},
        )
        ax.axvline(0, color="0.25", linewidth=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["display_label"], fontsize=9.5)
        for label, highlight in zip(ax.get_yticklabels(), selected):
            if highlight:
                label.set_fontweight("bold")
                label.set_color("crimson")
        ax.set_title(dgp.upper(), fontsize=12)
        ax.set_xlabel(
            "Mean true CATE in selected holdout top 10%",
            fontsize=10,
        )
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", color="0.86", linewidth=0.8)
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax, left=False, bottom=False)

    fig.suptitle(
        "Selected nested ensemble versus individual CATE estimators",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{FIG_BASENAME}.png"
    pdf = OUTPUT_DIR / f"{FIG_BASENAME}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    summary = summarize(load_rows())
    summary_path = OUTPUT_DIR / f"{FIG_BASENAME}.csv"
    summary.to_csv(summary_path, index=False)
    png, pdf = plot(summary)
    selected = summary[summary["method_type"].eq("Selected ensemble")]
    print(selected[["dgp", "n", "mean_true_cate", "rank_within_dgp"]].to_string(index=False))
    print(f"Saved {summary_path}")
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
