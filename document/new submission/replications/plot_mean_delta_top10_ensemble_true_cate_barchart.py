#!/usr/bin/env python3
"""Plot mean_delta_top10 ensemble true CATE among individual estimators.

The figure compares the Monte Carlo mean ground-truth CATE in the holdout
top-10% subgroup selected by each individual estimator against the ensemble
selected by the mean_delta_top10 statistic. The top-10% cutoff is computed on
the train/validation CATE scores and then applied to holdout CATE scores,
matching the regret calculation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import joblib
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import evaluate_proposed_targeting_statistics as base


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parents[1] / "new submission" / "figs"
SIM_DETAIL = ROOT / "output" / "bbar_first_stage_reproduction" / "simulation_detail.csv"
SELECTION_STAT = "mean_delta_top10"
FIG_BASENAME = "mc_true_cate_top10_mean_delta_top10_ensemble_vs_estimators"
HIGHLIGHT_LABEL = "Calibration-slope-first-stage ensemble"

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


def individual_rows() -> pd.DataFrame:
    rows = []
    for dgp in ["dgp1", "dgp2"]:
        seeds = base.discover_seeds(ROOT / "output" / dgp / "monte_carlo")
        for seed, path in seeds:
            cache = joblib.load(path)
            for idx, estimator in enumerate(cache["estimator_names"]):
                rows.append(
                    {
                        "dgp": dgp,
                        "seed": seed,
                        "method": estimator,
                        "display_label": ESTIMATOR_LABELS[estimator],
                        "method_type": "Individual estimator",
                        "true_cate_ho": holdout_true_cate_for_estimator(cache, idx),
                    }
                )
    return pd.DataFrame(rows)


def ensemble_rows() -> pd.DataFrame:
    detail = pd.read_csv(SIM_DETAIL)
    detail = detail[
        detail["selection_stat"].eq(SELECTION_STAT)
        & detail["first_stage_score"].eq("cal_slope")
    ].copy()
    detail["method"] = f"{SELECTION_STAT} ensemble"
    detail["display_label"] = HIGHLIGHT_LABEL
    detail["method_type"] = "Current ensemble"
    return detail.rename(columns={"true_ate_ho": "true_cate_ho"})[
        ["dgp", "seed", "method", "display_label", "method_type", "true_cate_ho"]
    ]


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = (
        rows.groupby(["dgp", "method", "display_label", "method_type"], sort=False)
        .agg(
            n=("seed", "nunique"),
            mean_true_cate=("true_cate_ho", "mean"),
            sd_true_cate=("true_cate_ho", "std"),
        )
        .reset_index()
    )
    out["se_true_cate"] = out["sd_true_cate"] / np.sqrt(out["n"])
    out["ci95"] = 1.96 * out["se_true_cate"]
    return out


def plot(summary: pd.DataFrame) -> tuple[Path, Path]:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    fig, axes = plt.subplots(ncols=2, figsize=(12.0, 8.2), sharex=False)

    dgp_titles = {
        "dgp1": "DGP1",
        "dgp2": "DGP2",
    }
    for ax, dgp in zip(axes, ["dgp1", "dgp2"]):
        plot_df = summary[summary["dgp"] == dgp].copy()
        plot_df = plot_df.sort_values(
            ["mean_true_cate", "display_label"], ascending=[True, True]
        )
        y_pos = np.arange(len(plot_df))
        is_ensemble = plot_df["method_type"].eq("Current ensemble")
        colors = np.where(is_ensemble, "crimson", "steelblue")
        edgecolors = np.where(is_ensemble, "crimson", "steelblue")

        ax.barh(
            y_pos,
            plot_df["mean_true_cate"],
            xerr=plot_df["ci95"],
            color=colors,
            edgecolor=edgecolors,
            alpha=0.88,
            height=0.72,
            error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2},
        )
        ax.axvline(0, color="0.25", linewidth=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["display_label"], fontsize=9.5)
        for label, highlight in zip(ax.get_yticklabels(), is_ensemble):
            if highlight:
                label.set_fontweight("bold")
        ax.set_title(dgp_titles[dgp], fontsize=12)
        ax.set_xlabel(
            "Mean ground-truth CATE in selected top-10% holdout subgroup",
            fontsize=10,
        )
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", color="0.86", linewidth=0.8)
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax, left=False, bottom=False)

    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    fig.suptitle(
        "Calibration-slope-first-stage ensemble versus individual CATE estimators",
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
    rows = pd.concat([individual_rows(), ensemble_rows()], ignore_index=True)
    summary = summarize(rows)
    summary_path = OUTPUT_DIR / f"{FIG_BASENAME}.csv"
    summary.to_csv(summary_path, index=False)
    png, pdf = plot(summary)
    print(f"Saved {summary_path}")
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
