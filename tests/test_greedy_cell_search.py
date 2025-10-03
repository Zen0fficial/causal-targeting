import numpy as np
import pandas as pd

from methods.greedy_cell_search import greedy_get_cell_search_results


class _FakeFoldResult:
    def __init__(self, ite: np.ndarray):
        self._ite = ite

    def get_subgroup_indicator(self, q_bot, q_top, which):
        # which is "all" in our usage; return indicator for ite in [q_bot, q_top] quantiles
        # Convert quantile thresholds to absolute thresholds on ite
        q = q_top  # expecting q_bot=0 or 1-q; use q_top for simplicity
        if q_bot == 0:
            threshold = np.quantile(self._ite, q)
            return self._ite <= threshold
        else:
            threshold = np.quantile(self._ite, q_bot)
            return self._ite >= threshold


class _FakeEstimator:
    def __init__(self, X: np.ndarray, ite: np.ndarray, n_splits: int = 1):
        self.X = X
        self.y = np.zeros(X.shape[0])
        self.t = np.zeros(X.shape[0])
        self.n_splits = n_splits
        # mimic dict of fold->result
        self.results = {fold: _FakeFoldResult(ite) for fold in range(n_splits)}


def test_greedy_covers_target_and_shape():
    # Build a toy dataset with binary features and a clear pattern
    # Features: A, B, C with values {0,1}
    rng = np.random.default_rng(42)
    n = 200
    A = rng.integers(0, 2, size=n)
    B = rng.integers(0, 2, size=n)
    C = rng.integers(0, 2, size=n)
    X = np.c_[A, B, C]

    # Define ITE so that target is primarily A==1 & B==1
    ite = -2 * (A == 1) - 1 * (B == 1) + 0.1 * rng.standard_normal(n)

    estimator = _FakeEstimator(X=X, ite=ite, n_splits=1)
    all_features = ["A", "B", "C"]
    top_features = all_features
    q_values = [0.2]

    # Print input summaries
    print("[test] n=", n)
    print("[test] feature distributions:")
    print("  A:", pd.Series(A).value_counts().to_dict())
    print("  B:", pd.Series(B).value_counts().to_dict())
    print("  C:", pd.Series(C).value_counts().to_dict())
    print("[test] q_values=", q_values)

    # Run greedy search
    res = greedy_get_cell_search_results(
        estimator,
        all_features=all_features,
        top_features=top_features,
        q_values=q_values,
        dir_neg=True,
        n_reps=1,
        verbose=True,
    )

    # Basic shape checks
    assert isinstance(res, pd.DataFrame)
    # Columns should include the run tag
    assert any(col.startswith("q=0.2/") for col in res.columns)
    print("[test] result shape:", res.shape)
    print("[test] result columns:", list(res.columns))
    if res.shape[0] > 0:
        print("[test] first selected cell:", list(res.index)[0])

    # Build target indicator the same way and check coverage
    fold = 0
    q = 0.2
    target_indicator = estimator.results[fold].get_subgroup_indicator(0, q, "all")
    thr = np.quantile(ite, q)
    print(f"[test] target threshold (q={q}): {thr:.4f}")
    print("[test] target size:", int(target_indicator.sum()))

    # Reconstruct coverage from selected cells without importing mlxtend-dependent code
    data_df = pd.DataFrame(X, columns=all_features)[top_features]
    selected_cells = list(res.index)
    df_full = pd.DataFrame({
        "A": A,
        "B": B,
        "C": C,
        "ite": ite,
        "target": target_indicator.astype(int),
    })
    mask = np.zeros(n, dtype=bool)

    def _indicator_from_cell(cell, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(df.shape[0], dtype=bool)
        for token in cell:
            feature, value = token.split("_", 1)
            m &= (df[feature].astype(str).values == value)
        return m

    for i, cell in enumerate(selected_cells):
        ind = _indicator_from_cell(cell, data_df)
        df_full[f"cell{i}_member"] = ind.astype(int)
        mask |= ind
    df_full["covered"] = mask.astype(int)
    print("[test] full input + indicators:\n" + df_full.to_string(index=False))

    # Greedy should fully cover the target set (allowing for ties/noise, ensure at least 95%)
    covered_tp = (mask & target_indicator).sum()
    total_tp = target_indicator.sum()
    print(f"[test] coverage: {covered_tp}/{total_tp} ({(covered_tp/total_tp if total_tp>0 else 0):.1%})")
    if total_tp > 0:
        assert covered_tp / total_tp >= 0.95
    else:
        assert covered_tp == 0


