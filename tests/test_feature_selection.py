import numpy as np
import pandas as pd

from src.feature_selection import TransparentFeatureSelector


def test_selector_shapes_and_report(tmp_path):
    rng = np.random.RandomState(0)
    n, p = 200, 50
    X = pd.DataFrame(rng.randn(n, p), columns=[f"x_{i}" for i in range(p)])
    # Create correlated duplicates to exercise correlation pruning
    X["x_dup"] = X["x_0"] + 1e-4 * rng.randn(n)
    # Outcome depends on a few features
    y = (2.0 * X["x_0"].values - 1.5 * X["x_3"].values + 0.5 * rng.randn(n))
    t = (rng.rand(n) < 0.4).astype(int)

    selector = TransparentFeatureSelector(
        strategy="filters+correlation+univariate+double_selection+stability",
        params={
            "min_variance": 1e-8,
            "max_missing": 0.5,
            "corr_threshold": 0.9,
            "univariate": {"k": 20, "kind": "mutual_info_regression"},
            "double_selection": {"alpha": "cv", "max_iter": 2000},
            "stability": {"n_subsamples": 10, "sample_frac": 0.7, "threshold": 0.3, "max_iter": 500},
        },
        report_path=str(tmp_path / "selector_report.csv"),
        random_state=0,
    )
    selector.fit(X, y, t)
    X_sel = selector.transform(X)

    assert X_sel.shape[0] == n
    # Should select at least one feature
    assert X_sel.shape[1] >= 1
    # Report should exist
    assert (tmp_path / "selector_report.csv").exists()
    assert (tmp_path / "selector_report_meta.json").exists()


