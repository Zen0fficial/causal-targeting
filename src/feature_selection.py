import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.utils import check_random_state


@dataclass
class SelectionReportRow:
    feature: str
    kept: bool
    reason: str
    variance: Optional[float] = None
    missing_fraction: Optional[float] = None
    univariate_score: Optional[float] = None
    double_selected: Optional[bool] = None
    stability_frequency: Optional[float] = None


class TransparentFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transparent feature selection transformer that supports:
    - Low-variance and high-missingness filtering
    - Correlation pruning with deterministic tie-breaks
    - Univariate screening (F-test or mutual information)
    - Double selection (Lasso for y|X and L1-Logit for t|X; union of supports)
    - Stability selection (subsampled Lasso selection frequency threshold)

    Notes
    -----
    - Fit expects X to be a pandas DataFrame with column names.
    - For supervised steps (univariate, double selection, stability), provide y and t.
    - All steps are applied in the order listed in the strategy string.
    """

    def __init__(
        self,
        strategy: str = "filters+correlation+univariate",
        params: Optional[Dict] = None,
        report_path: Optional[str] = None,
        random_state: int = 0,
    ) -> None:
        self.strategy = strategy
        self.params = params or {}
        self.report_path = report_path
        self.random_state = random_state

        # Learned during fit
        self.selected_columns_: List[str] = []
        self.report_: List[SelectionReportRow] = []
        self.fitted_: bool = False

    # ------------------------ Public API ------------------------
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None):
        self._validate_input(X)
        rng = check_random_state(self.random_state)
        steps = [s.strip() for s in self.strategy.split("+") if s.strip()]

        current_cols = list(X.columns)
        report_rows: Dict[str, SelectionReportRow] = {
            c: SelectionReportRow(feature=c, kept=True, reason="initial") for c in current_cols
        }

        # Step 1: Missingness and variance filters
        if any(s in steps for s in ["filters", "filter", "missingness", "lowvar"]):
            max_missing = float(self.params.get("max_missing", 0.3))
            min_variance = float(self.params.get("min_variance", 1e-8))
            missing_frac = X[current_cols].isna().mean()
            variances = X[current_cols].var(ddof=0)

            keep_mask = (missing_frac <= max_missing) & (variances.fillna(0.0) >= min_variance)
            for c in current_cols:
                rr = report_rows[c]
                rr.missing_fraction = float(missing_frac.get(c, np.nan))
                rr.variance = float(variances.get(c, np.nan))
                if not keep_mask.get(c, True):
                    rr.kept = False
                    if missing_frac.get(c, 0.0) > max_missing:
                        rr.reason = f"removed: missing>{max_missing}"
                    elif float(variances.get(c, 0.0)) < min_variance:
                        rr.reason = f"removed: variance<{min_variance}"
            current_cols = [c for c in current_cols if report_rows[c].kept]

        # Step 2: Correlation pruning
        if "correlation" in steps or "corr" in steps:
            threshold = float(self.params.get("corr_threshold", 0.95))
            if len(current_cols) > 1:
                corr = X[current_cols].corr().abs()
                # Tie-break: higher variance wins; if y is provided, higher |corr(y,X)| wins
                variances = X[current_cols].var(ddof=0).fillna(0.0)
                y_assoc = None
                if y is not None:
                    try:
                        y_assoc = pd.Series(np.abs(np.corrcoef(X[current_cols].fillna(0.0).values.T, y)[-1, :-1]), index=current_cols)
                    except Exception:
                        y_assoc = None
                selected = []
                removed = set()
                for c in sorted(current_cols, key=lambda k: (y_assoc.get(k, 0.0) if y_assoc is not None else 0.0, variances.get(k, 0.0)), reverse=True):
                    if c in removed:
                        continue
                    selected.append(c)
                    to_remove = [d for d in current_cols if d != c and d not in removed and corr.loc[c, d] >= threshold]
                    for d in to_remove:
                        removed.add(d)
                        rr = report_rows[d]
                        rr.kept = False
                        rr.reason = f"removed: corr>{threshold} with {c}"
                current_cols = selected

        # Step 3: Univariate screening
        if "univariate" in steps and len(current_cols) > 0 and y is not None:
            uni_cfg = self.params.get("univariate", {})
            k = int(uni_cfg.get("k", min(len(current_cols), 200)))
            kind = str(uni_cfg.get("kind", "mutual_info_regression"))
            X_sub = X[current_cols].fillna(0.0).values
            if kind == "f_regression":
                scores, _ = f_regression(X_sub, y)
            else:
                scores = mutual_info_regression(X_sub, y, random_state=rng)
            scores = pd.Series(scores, index=current_cols).fillna(0.0)
            topk = scores.sort_values(ascending=False).head(k).index.tolist()
            for c in current_cols:
                report_rows[c].univariate_score = float(scores.get(c, np.nan))
                if c not in topk:
                    report_rows[c].kept = False
                    report_rows[c].reason = f"removed: univariate_not_top_{k}"
            current_cols = topk

        # Step 4: Double selection
        if ("double_selection" in steps or "double" in steps) and len(current_cols) > 0 and y is not None and t is not None:
            ds_cfg = self.params.get("double_selection", {})
            alpha = ds_cfg.get("alpha", "cv")
            max_iter = int(ds_cfg.get("max_iter", 2000))
            X_sub = X[current_cols].fillna(0.0).values
            # y|X via LassoCV
            if alpha == "cv":
                lasso = LassoCV(cv=5, random_state=rng, n_jobs=None, max_iter=max_iter)
            else:
                lasso = LassoCV(alphas=[float(alpha)], cv=5, random_state=rng, n_jobs=None, max_iter=max_iter)
            lasso.fit(X_sub, y)
            support_y = np.abs(lasso.coef_) > 1e-12
            # t|X via L1 logistic
            logit = LogisticRegression(penalty="l1", solver="liblinear", max_iter=max_iter, random_state=self.random_state)
            logit.fit(X_sub, t)
            support_t = np.abs(logit.coef_).ravel() > 1e-12
            keep_mask = support_y | support_t
            kept_cols = [c for c, k in zip(current_cols, keep_mask) if k]
            for c, k, sy, st in zip(current_cols, keep_mask, support_y, support_t):
                report_rows[c].double_selected = bool(sy or st)
                if not k:
                    report_rows[c].kept = False
                    report_rows[c].reason = "removed: double_selection_support_false"
                else:
                    # If previously removed, keep the earlier reason; otherwise set
                    if report_rows[c].reason in ("initial", "removed: univariate_not_top_{}".format(self.params.get("univariate", {}).get("k", ""))):
                        report_rows[c].reason = "kept: double_selection_support"
            current_cols = kept_cols

        # Step 5: Stability selection
        if ("stability" in steps or "stability_selection" in steps) and len(current_cols) > 0 and y is not None:
            st_cfg = self.params.get("stability", {})
            n_subsamples = int(st_cfg.get("n_subsamples", 50))
            sample_frac = float(st_cfg.get("sample_frac", 0.5))
            threshold = float(st_cfg.get("threshold", 0.6))
            max_iter = int(st_cfg.get("max_iter", 2000))

            X_sub_all = X[current_cols].fillna(0.0).values
            n = X_sub_all.shape[0]
            freq = np.zeros(len(current_cols), dtype=float)
            for i in range(n_subsamples):
                idx = rng.choice(n, size=max(1, int(sample_frac * n)), replace=False)
                X_sub = X_sub_all[idx]
                y_sub = y[idx]
                lasso = LassoCV(cv=5, random_state=rng, n_jobs=None, max_iter=max_iter)
                lasso.fit(X_sub, y_sub)
                freq += (np.abs(lasso.coef_) > 1e-12).astype(float)
            freq /= float(n_subsamples)
            kept_cols = [c for c, f in zip(current_cols, freq) if f >= threshold]
            for c, f in zip(current_cols, freq):
                report_rows[c].stability_frequency = float(f)
                if c not in kept_cols:
                    report_rows[c].kept = False
                    report_rows[c].reason = f"removed: stability_freq<{threshold}"
            current_cols = kept_cols

        # Finalize selections
        self.selected_columns_ = current_cols
        # Update reasons for kept features if they still have "initial"
        for c in self.selected_columns_:
            if report_rows[c].reason == "initial":
                report_rows[c].reason = "kept"
        self.report_ = [report_rows[c] for c in report_rows]
        self._maybe_write_report()
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        self._validate_input(X)
        # Select missing columns robustly by intersecting
        cols = [c for c in self.selected_columns_ if c in X.columns]
        return X[cols]

    # ------------------------ Internals ------------------------
    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("TransparentFeatureSelector expects X as a pandas DataFrame with column names.")
        if X.columns.has_duplicates:
            raise ValueError("Duplicate column names are not supported.")

    def _ensure_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("TransparentFeatureSelector must be fitted before transform().")

    def _maybe_write_report(self) -> None:
        if self.report_path is None:
            # default to timestamped file under output/params/feature_selection/
            out_dir = os.path.join("output", "params", "feature_selection")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.report_path = os.path.join(out_dir, f"selector_report_{ts}.csv")
        # Write report CSV
        rows = [
            {
                "feature": r.feature,
                "kept": r.kept,
                "reason": r.reason,
                "variance": r.variance,
                "missing_fraction": r.missing_fraction,
                "univariate_score": r.univariate_score,
                "double_selected": r.double_selected,
                "stability_frequency": r.stability_frequency,
            }
            for r in self.report_
        ]
        pd.DataFrame(rows).to_csv(self.report_path, index=False)
        # Save params alongside
        meta_path = self.report_path.replace(".csv", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "strategy": self.strategy,
                "params": self.params,
                "random_state": self.random_state,
                "selected_columns": self.selected_columns_,
            }, f, ensure_ascii=False, indent=2)


