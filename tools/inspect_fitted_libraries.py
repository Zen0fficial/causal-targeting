import sys
import joblib
import numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python tools/inspect_fitted_libraries.py <path_to_pickle>")
    sys.exit(1)

p = Path(sys.argv[1])
print("File:", p.resolve())
print("Exists:", p.exists(), "Size(MB):", round(p.stat().st_size/1e6, 1) if p.exists() else None)

fitted = joblib.load(p)
print("Top-level libraries:", list(fitted.keys()))

for lib_name, lib in fitted.items():
    print("\n=== LIB:", lib_name, "(estimators:", len(lib), ") ===")
    est_nan = {}
    for est_name, est in lib.items():
        nans = 0
        vals = 0
        # results is a dict of folds -> CATEEstimatorResults
        if hasattr(est, 'results') and isinstance(est.results, dict):
            for fold_idx, res in est.results.items():
                if hasattr(res, 'tau'):
                    arr = np.asarray(res.tau)
                    if arr.size > 0:
                        nans += int(np.isnan(arr).sum())
                        vals += int(arr.size)
        est_nan[est_name] = (nans, vals)
    total_nans = sum(n for n, _ in est_nan.values())
    total_vals = sum(v for _, v in est_nan.values())
    print("Total NaNs:", total_nans, "over", total_vals)
    worst = sorted(est_nan.items(), key=lambda kv: kv[1][0], reverse=True)[:10]
    for name, (n, v) in worst:
        pct = (100.0*n/v) if v else 0.0
        print(f"  - {name}: NaNs={n} / {v} ({pct:.2f}%)")

