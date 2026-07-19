#!/usr/bin/env python3
"""Plot the B_bar_0.9-first-stage ensemble against all 23 estimators."""

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


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parents[1] / "new submission" / "figs"
REPRODUCTION_DETAIL = (
    ROOT / "output" / "bbar_first_stage_reproduction" / "simulation_detail.csv"
)
INDIVIDUAL_SUMMARY = (
    OUTPUT_DIR / "mc_true_cate_top10_mean_delta_top10_ensemble_vs_estimators.csv"
)
FIG_BASENAME = "mc_true_cate_top10_bbar_first_stage_ensemble_vs_estimators"
HIGHLIGHT_LABEL = r"$\bar{B}_{0.9}$-first-stage ensemble"
DGP_ORDER = ["dgp1", "dgp2"]

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


def individual_summaries() -> pd.DataFrame:
    summary = pd.read_csv(INDIVIDUAL_SUMMARY)
    rows = summary[summary["method_type"].eq("Individual estimator")].copy()
    for dgp in DGP_ORDER:
        observed = set(rows.loc[rows["dgp"].eq(dgp), "method"])
        if observed != set(ESTIMATOR_LABELS):
            raise ValueError(f"{dgp} does not contain the expected 23 estimators")
    rows["display_label"] = rows["method"].map(ESTIMATOR_LABELS)
    return rows[
        [
            "dgp",
            "method",
            "display_label",
            "method_type",
            "n",
            "mean_true_cate",
            "sd_true_cate",
        ]
    ]


def bbar_ensemble_summaries() -> pd.DataFrame:
    detail = pd.read_csv(REPRODUCTION_DETAIL)
    detail = detail[detail["first_stage_score"].eq("B_bar_0.9")].copy()
    if detail.duplicated(["dgp", "seed"]).any():
        raise ValueError("The B_bar_0.9 reproduction contains duplicate DGP/seed rows")
    out = (
        detail.groupby("dgp", sort=False)
        .agg(
            n=("seed", "nunique"),
            mean_true_cate=("true_ate_ho", "mean"),
            sd_true_cate=("true_ate_ho", "std"),
        )
        .reset_index()
    )
    out["method"] = "bbar_first_stage_ensemble"
    out["display_label"] = HIGHLIGHT_LABEL
    out["method_type"] = "Bbar first-stage ensemble"
    return out[
        [
            "dgp",
            "method",
            "display_label",
            "method_type",
            "n",
            "mean_true_cate",
            "sd_true_cate",
        ]
    ]


def build_summary() -> pd.DataFrame:
    summary = pd.concat(
        [individual_summaries(), bbar_ensemble_summaries()], ignore_index=True
    )
    summary["n"] = summary["n"].astype(int)
    summary["se_true_cate"] = summary["sd_true_cate"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se_true_cate"]
    summary["ci95_lower"] = summary["mean_true_cate"] - summary["ci95"]
    summary["ci95_upper"] = summary["mean_true_cate"] + summary["ci95"]
    summary["rank_within_dgp"] = (
        summary.groupby("dgp")["mean_true_cate"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    summary["n_methods_ranked"] = summary.groupby("dgp")["method"].transform("size")
    for dgp in DGP_ORDER:
        current = summary[summary["dgp"].eq(dgp)]
        if len(current) != 24:
            raise ValueError(f"{dgp} must contain 23 estimators and one ensemble")
    return summary.sort_values(["dgp", "rank_within_dgp", "display_label"])


def plot(summary: pd.DataFrame) -> tuple[Path, Path]:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    fig, axes = plt.subplots(ncols=2, figsize=(12.0, 8.2), sharex=False)
    for ax, dgp in zip(axes, DGP_ORDER):
        plot_df = summary[summary["dgp"].eq(dgp)].copy()
        plot_df = plot_df.sort_values(
            ["mean_true_cate", "display_label"], ascending=[True, True]
        )
        y_pos = np.arange(len(plot_df))
        highlighted = plot_df["method_type"].eq("Bbar first-stage ensemble")
        colors = np.where(highlighted, "crimson", "steelblue")
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
        for label, is_highlighted in zip(ax.get_yticklabels(), highlighted):
            if is_highlighted:
                label.set_fontweight("bold")
                label.set_color("crimson")
        ax.set_xlim(0, 1.055 * plot_df["ci95_upper"].max())
        ax.set_title(dgp.upper(), fontsize=11)
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
        r"$\bar{B}_{0.9}$-first-stage ensemble versus individual CATE estimators",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{FIG_BASENAME}.png"
    pdf = OUTPUT_DIR / f"{FIG_BASENAME}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    summary = build_summary()
    summary_path = OUTPUT_DIR / f"{FIG_BASENAME}.csv"
    summary.to_csv(summary_path, index=False)
    png, pdf = plot(summary)
    highlighted = summary[summary["method_type"].eq("Bbar first-stage ensemble")]
    print(highlighted[["dgp", "mean_true_cate", "rank_within_dgp"]].to_string(index=False))
    print(f"Saved {summary_path}")
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
