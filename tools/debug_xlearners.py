import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import joblib

# Ensure project root is importable so that pickled classes under `methods.*` resolve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional: import wrappers for typing/introspection
try:
    from methods.cate_estimator_wrappers import XLearnerWrapper  # type: ignore
except Exception:
    XLearnerWrapper = object  # fallback for isinstance checks


def format_pct(n: int, d: int) -> str:
    if d == 0:
        return "0.00%"
    return f"{(100.0 * n / d):.2f}%"


def check_inputs(est) -> Dict[str, Tuple[int, int]]:
    X = getattr(est, "X", None)
    t = getattr(est, "t", None)
    y = getattr(est, "y", None)
    stats: Dict[str, Tuple[int, int]] = {}
    if X is not None:
        X = np.asarray(X)
        stats["X_nan"] = (int(np.isnan(X).sum()), int(X.size))
        stats["X_inf"] = (int(np.isinf(X).sum()), int(X.size))
    if t is not None:
        t = np.asarray(t)
        stats["t_nan"] = (int(np.isnan(t).sum()), int(t.size))
        stats["t_not_binary"] = (int(np.any(~np.isin(np.unique(t), [0, 1]))), int(t.size))
    if y is not None:
        y = np.asarray(y)
        stats["y_nan"] = (int(np.isnan(y).sum()), int(y.size))
        stats["y_inf"] = (int(np.isinf(y).sum()), int(y.size))
    return stats


def compute_propensity(est) -> np.ndarray:
    # Reproduce the wrapper's logic used in _predict_tau
    X = est.X
    T = est.t
    pl = getattr(est, "propensity_learner", None)
    if pl is not None:
        # Fit a fresh copy to avoid mutating the stored one
        import copy as _copy
        pl_copy = _copy.deepcopy(pl)
        pl_copy.fit(X, T)
        if hasattr(pl_copy, "predict_proba"):
            p_all = pl_copy.predict_proba(X)[:, 1]
        elif hasattr(pl_copy, "decision_function"):
            scores = pl_copy.decision_function(X)
            p_all = 1.0 / (1.0 + np.exp(-scores))
        else:
            # Fallback to mean(T)
            p_all = np.mean(T) * np.ones(X.shape[0], dtype=float)
    else:
        p_all = np.mean(T) * np.ones(X.shape[0], dtype=float)
    return p_all


def summarize_propensity(p: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.nanmin(p)),
        "p1": float(np.nanpercentile(p, 1)),
        "median": float(np.nanmedian(p)),
        "p99": float(np.nanpercentile(p, 99)),
        "max": float(np.nanmax(p)),
        "num_nan": int(np.isnan(p).sum()),
        "num_zero": int(np.sum(np.isclose(p, 0.0))),
        "num_one": int(np.sum(np.isclose(p, 1.0))),
    }


def safe_predict(model, X) -> np.ndarray:
    try:
        return np.asarray(model.predict(X))
    except Exception:
        return np.full((X.shape[0],), np.nan)


def diagnose_xlearner(est) -> Dict[str, object]:
    diag: Dict[str, object] = {}
    # 1) Inputs
    diag["input_stats"] = check_inputs(est)

    # 2) Propensity
    try:
        p = compute_propensity(est)
        diag["propensity_summary"] = summarize_propensity(p)
    except Exception as e:
        diag["propensity_error"] = str(e)
        p = None

    # 3) NaNs in meta predictions by fold
    fold_nan: List[Tuple[int, int, int]] = []  # (fold, num_nan, total)
    nan_indices_any: np.ndarray | None = None
    for fold, res in getattr(est, "results", {}).items():
        tau = np.asarray(getattr(res, "tau", np.array([])))
        if tau.size == 0:
            fold_nan.append((fold, 0, 0))
            continue
        num_nan = int(np.isnan(tau).sum())
        fold_nan.append((fold, num_nan, int(tau.size)))
        if num_nan > 0 and nan_indices_any is None:
            nan_indices_any = np.isnan(tau)
    diag["meta_tau_nan_by_fold"] = fold_nan

    # 4) If NaNs detected, try component learners on those rows
    if nan_indices_any is not None and np.any(nan_indices_any):
        X = est.X
        # Outcome learners
        to_pred = [
            ("treat_outcome", getattr(est, "treatment_outcome_learner", None)),
            ("control_outcome", getattr(est, "control_outcome_learner", None)),
            ("treat_effect", getattr(est, "treatment_effect_learner", None)),
            ("control_effect", getattr(est, "control_effect_learner", None)),
        ]
        comp_nan: Dict[str, Tuple[int, int]] = {}
        for label, model in to_pred:
            if model is None:
                continue
            yhat_all = safe_predict(model, X)
            comp_nan[label] = (int(np.isnan(yhat_all[nan_indices_any]).sum()), int(np.sum(nan_indices_any)))
        diag["component_pred_nan_on_bad_rows"] = comp_nan

        # 5) Recompute meta prediction explicitly to confirm source
        try:
            if p is None:
                p = compute_propensity(est)
            if hasattr(est.meta_learner, "predict"):
                tau2 = est.meta_learner.predict(X, p=p) if "p" in est.meta_learner.predict.__code__.co_varnames else est.meta_learner.predict(X)
                tau2 = np.asarray(tau2)
                diag["meta_predict_recomputed_nan"] = int(np.isnan(tau2).sum())
        except Exception as e:
            diag["meta_predict_recomputed_error"] = str(e)

    return diag


def main():
    parser = argparse.ArgumentParser(description="Diagnose NaNs in X-Learner estimators from fitted libraries.")
    parser.add_argument("pickle_path", type=str, help="Path to <outcome>_fitted_libraries.pkl")
    parser.add_argument("--only", type=str, default="x_", help="Substring filter for estimator names (default: x_)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of X-learners to inspect (0 = all)")
    args = parser.parse_args()

    p = Path(args.pickle_path)
    if not p.exists():
        print("File not found:", p)
        sys.exit(1)

    print("Loading:", p.resolve())
    print("Note: this may be large and memory-intensive.")
    # Ensure 'methods' is importable for unpickling
    try:
        fitted_libraries = joblib.load(p)
    except ModuleNotFoundError as e:
        print("Import error while loading pickle:", e)
        print("Ensure the project root is on PYTHONPATH and run from project root.")
        sys.exit(2)

    lib_names = list(fitted_libraries.keys())
    print("Libraries:", lib_names)

    inspected = 0
    for lib_name, lib in fitted_libraries.items():
        # Collect X-learner names
        x_names = [name for name in lib.keys() if args.only in name]
        print(f"\n=== Library: {lib_name} | X-Learners: {len(x_names)} ===")
        for name in x_names:
            est = lib[name]
            is_x = isinstance(est, XLearnerWrapper) or name.startswith("x_")
            if not is_x:
                continue
            inspected += 1
            print(f"\nEstimator: {name}")
            diag = diagnose_xlearner(est)

            # Print compact summary
            inp = diag.get("input_stats", {})
            print("  Inputs:", {k: f"{v[0]}/{v[1]} ({format_pct(v[0], v[1])})" for k, v in inp.items()})
            prop = diag.get("propensity_summary")
            if prop:
                print("  Propensity:", {k: prop[k] for k in ["min","p1","median","p99","max","num_nan","num_zero","num_one"]})
            elif "propensity_error" in diag:
                print("  Propensity error:", diag["propensity_error"]) 

            by_fold = diag.get("meta_tau_nan_by_fold", [])
            total_nan = sum(n for _, n, _ in by_fold)
            total_vals = sum(d for _, _, d in by_fold)
            print("  Meta tau NaNs by fold:", [(f, n, d, format_pct(n, d)) for f, n, d in by_fold])
            print("  Meta tau total:", f"{total_nan}/{total_vals} ({format_pct(total_nan, total_vals)})")

            comp = diag.get("component_pred_nan_on_bad_rows")
            if comp:
                print("  Component preds NaN on NaN-tau rows:", {k: f"{v[0]}/{v[1]} ({format_pct(v[0], v[1])})" for k, v in comp.items()})
            if "meta_predict_recomputed_nan" in diag:
                print("  Meta predict recomputed NaNs:", diag["meta_predict_recomputed_nan"])
            if "meta_predict_recomputed_error" in diag:
                print("  Meta predict recompute error:", diag["meta_predict_recomputed_error"])

            if args.limit and inspected >= args.limit:
                break
        if args.limit and inspected >= args.limit:
            break

    print("\nDone. Inspected:", inspected)


if __name__ == "__main__":
    main()
