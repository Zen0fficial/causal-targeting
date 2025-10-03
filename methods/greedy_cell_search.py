import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Iterable


def _get_top_quantile_indicator(CATE_estimator,
                                fold: int,
                                q: float,
                                dir_neg: bool) -> np.ndarray:
    """
    Return boolean indicator for the target subgroup for a given fold and q.
    Matches the convention used in methods.cell_search.CellSearch.
    """
    if dir_neg:
        q_bot = 0
        q_top = q
    else:
        q_bot = 1 - q
        q_top = 1
    return CATE_estimator.results[fold].get_subgroup_indicator(q_bot, q_top, "all")


def _build_distinct_cells(te_df: pd.DataFrame) -> Tuple[List[frozenset], List[np.ndarray]]:
    """
    Build the set of distinct cells from a transaction-encoded dataframe.

    Each row corresponds to an individual's itemset; a distinct cell is the
    (frozenset) of items present in that row. We collapse duplicate rows and
    return:
      - a list of unique frozenset itemsets (cells)
      - a parallel list of boolean index arrays indicating membership in each cell
    """
    # Represent each row by the tuple of active column indices to avoid large strings
    active_cols_per_row: List[Tuple[int, ...]] = []
    active_cols_cache = te_df.columns.to_numpy()
    values = te_df.values
    n_rows = values.shape[0]

    for i in range(n_rows):
        active_idx = np.flatnonzero(values[i])
        active_cols_per_row.append(tuple(active_idx.tolist()))

    # Map from tuple of active indices to row indices
    pattern_to_indices: Dict[Tuple[int, ...], List[int]] = {}
    for row_idx, pattern in enumerate(active_cols_per_row):
        if pattern not in pattern_to_indices:
            pattern_to_indices[pattern] = []
        pattern_to_indices[pattern].append(row_idx)

    # Build outputs
    cells: List[frozenset] = []
    indicators: List[np.ndarray] = []
    n = te_df.shape[0]
    for pattern, row_indices in pattern_to_indices.items():
        cell_items = frozenset(active_cols_cache[list(pattern)].tolist())
        cell_indicator = np.zeros(n, dtype=bool)
        cell_indicator[np.array(row_indices, dtype=int)] = True
        cells.append(cell_items)
        indicators.append(cell_indicator)

    return cells, indicators


def _recode_samples_for_transactions(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode the input data frame into a transaction-encoded wide binary matrix
    without requiring external libraries. Each binary/categorical feature x is
    split into columns like x_0, x_1 corresponding to its observed values.
    """
    # Build tokenized itemsets per row
    tokens_per_row: List[List[str]] = []
    all_tokens: List[str] = []
    cols = list(input_df.columns)
    for _, row in input_df.iterrows():
        tokens = [f"{feature}_{str(value)}" for feature, value in zip(cols, row)]
        tokens_per_row.append(tokens)
        all_tokens.extend(tokens)
    unique_tokens = sorted(set(all_tokens))

    n = input_df.shape[0]
    m = len(unique_tokens)
    mat = np.zeros((n, m), dtype=bool)
    token_to_idx = {tok: j for j, tok in enumerate(unique_tokens)}
    for i, tokens in enumerate(tokens_per_row):
        idxs = [token_to_idx[t] for t in tokens]
        mat[i, idxs] = True
    te_df = pd.DataFrame(mat, columns=unique_tokens)
    return te_df


def _greedy_cover_from_cells(cells: List[frozenset],
                             indicators: List[np.ndarray],
                             target_indicator: np.ndarray,
                             verbose: bool = False) -> List[frozenset]:
    """
    Greedily select cells until the target set is fully covered.

    At each step, score a cell by:
      score = TP_remaining / |Target_remaining| - FP_remaining / |NonTarget_remaining|
    Break ties by larger TP_remaining, then smaller FP_remaining, then larger cell size.
    Cells with TP_remaining == 0 are ignored.
    """
    n = target_indicator.shape[0]
    remaining_target = target_indicator.copy()
    remaining_non_target = ~target_indicator.copy()

    selected: List[frozenset] = []
    already_covered = np.zeros(n, dtype=bool)

    # Precompute per-cell masks intersected with remaining masks at runtime
    # Loop until all target units are covered or no progress can be made
    step = 0
    while True:
        if remaining_target.sum() == 0:
            break

        denom_tp = remaining_target.sum()
        denom_fp = remaining_non_target.sum()
        best_idx = -1
        best_score = -np.inf
        best_tp = -1
        best_fp = np.inf
        best_size = -1

        for idx, cell_mask in enumerate(indicators):
            # Only consider units not already covered? We allow overlapping but compute marginal gains
            effective_mask = cell_mask & ~already_covered
            if not effective_mask.any():
                continue
            tp = int((effective_mask & remaining_target).sum())
            if tp == 0:
                continue
            fp = int((effective_mask & remaining_non_target).sum())
            # Guard denominators; if denom_fp==0 treat FP% as 0
            tp_pct = tp / denom_tp if denom_tp > 0 else 0.0
            fp_pct = fp / denom_fp if denom_fp > 0 else 0.0
            score = tp_pct - fp_pct

            size = int(effective_mask.sum())
            # Tie-breakers: larger tp, smaller fp, larger size
            if (score > best_score or
                (score == best_score and (tp > best_tp or
                                          (tp == best_tp and (fp < best_fp or
                                                              (fp == best_fp and size > best_size)))))):
                best_score = score
                best_idx = idx
                best_tp = tp
                best_fp = fp
                best_size = size

        if best_idx == -1:
            # No cell can make progress
            if verbose:
                print("[greedy] No further progress; stopping.")
            break

        chosen_mask = indicators[best_idx] & ~already_covered
        selected.append(cells[best_idx])
        already_covered |= chosen_mask
        # Update remaining sets
        remaining_target &= ~chosen_mask
        remaining_non_target &= ~chosen_mask

        if verbose:
            step += 1
            covered_tp = (target_indicator & already_covered).sum()
            total_tp = target_indicator.sum()
            preview = list(cells[best_idx])[:5]
            more = "..." if len(cells[best_idx]) > 5 else ""
            print(f"[greedy][step {step}] chose cell with {best_tp} TP, {best_fp} FP, size {best_size}, score {best_score:.4f}")
            print(f"  cell: {preview}{more}")
            if total_tp > 0:
                print(f"  coverage: {covered_tp}/{total_tp} TP covered ({covered_tp/total_tp:.1%})")

    return selected


def greedy_get_cell_search_results(CATE_estimator,
                                   all_features: List[str],
                                   top_features: List[str],
                                   q_values: Iterable[float],
                                   dir_neg: bool = True,
                                   n_reps: int = 1,
                                   verbose: bool = False) -> pd.DataFrame:
    """
    Fast greedy cell search that considers only distinct unit cells (unique
    itemsets per row) rather than ranking all possible feature combinations.

    For each fold and q, we:
      1) Build transaction-encoded data for the selected top_features
      2) Collapse rows into distinct cells
      3) Greedily select cells by the difference between percentages of
         true positives and false positives until the target set is covered

    Output format matches the original get_cell_search_results contract, with
    a "cells" index of selected frozenset itemsets and columns named
    "q={q}/fold={fold}/b={b}".
    """
    search_results_df = pd.DataFrame({"cells": []})

    for fold in range(CATE_estimator.n_splits):
        # Prepare data and target indicator
        data_df = pd.DataFrame(CATE_estimator.X, columns=all_features)[top_features]
        te_df = _recode_samples_for_transactions(data_df)
        for q in q_values:
            target_indicator = _get_top_quantile_indicator(CATE_estimator, fold, q, dir_neg)
            if verbose:
                total_tp = int(target_indicator.sum())
                print(f"[greedy] fold={fold}, q={q}: target size {total_tp} of {len(target_indicator)}")

            # Restrict to active set initially equals all units
            # Build distinct cells once per (fold, q) on the full set
            cells, indicators = _build_distinct_cells(te_df)

            # Run greedy selection once per repetition (repetitions identical unless ties differ)
            for b in range(n_reps):
                if verbose and n_reps > 1:
                    print(f"[greedy] repetition b={b}")
                selected_cells = _greedy_cover_from_cells(cells, indicators, target_indicator, verbose=verbose)
                if len(selected_cells) == 0:
                    # No cells found (empty target or degenerate case)
                    continue
                run_results = pd.DataFrame({
                    "cells": selected_cells,
                    f"q={q}/fold={fold}/b={b}": 1,
                })
                search_results_df = search_results_df.merge(run_results, how="outer", on="cells")

    if search_results_df.shape[0] == 0:
        return pd.DataFrame(index=pd.Index([], name="cells"))

    search_results_df = search_results_df.set_index("cells").fillna(0).astype(int)
    return search_results_df


__all__ = [
    "greedy_get_cell_search_results",
]


