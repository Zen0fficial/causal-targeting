# 04a/04b Notebook Separation Plan

## Current Responsibilities

| Notebook | Current Role | Key Outputs |
|----------|--------------|-------------|
| **04a_rank_estimators.ipynb** | Compute t-statistics, rank by Q set, create ensembles | `ensemble_table.pkl`, `t_cv_Q_results.pkl` |
| **04b_validate_ensembles.ipynb** | Compute B̄ and Δτ̄ metrics, select optimal (Q*,k*) | `optimal_config.pkl`, `validation_results.csv` |

---

## Dependency

04b **depends on** 04a:
```python
# In 04b (line 146-147):
ensemble_table = joblib.load(INTERMEDIATE_PATH / f"{OUTCOME_NAME}_ensemble_table.pkl")
```

**Current issue:** Both notebooks define Q_SETS independently → must stay synchronized.

---

## Proposed Clean Separation

### [MODIFY] [04a_rank_estimators.ipynb](file:///Users/zenofficial/Documents/statistics/pcs/document/projects/causal-targeting-main/04a_rank_estimators.ipynb)

**Role: "Find Candidate Ensembles"**

1. Compute t-statistics for all (estimator, CV, fold, q)
2. Average t-stats over Q sets
3. Rank estimators per CV
4. Select ensembles via intersection criterion (top-k in ALL CVs)
5. **Remove Q1 from Q_SETS**

**Outputs:**
- `{outcome}_all_t_statistics.csv`
- `{outcome}_t_cv_Q_results.pkl`
- `{outcome}_ensemble_table.pkl`

---

### [MODIFY] [04b_validate_ensembles.ipynb](file:///Users/zenofficial/Documents/statistics/pcs/document/projects/causal-targeting-main/04b_validate_ensembles.ipynb)

**Role: "Rank Ensembles by Validation Metrics"**

1. Load `ensemble_table.pkl` from 04a
2. Compute B̄_0.8 and Δτ̄_0.8 for each (Q, k) configuration
3. Select optimal (Q*, k*) by maximizing B̄_0.8, tiebreak by Δτ̄_0.8
4. **Remove duplicate Q_SETS definition** — load from config or 04a output

**Outputs:**
- `{outcome}_estimator_B_metrics.csv`
- `{outcome}_validation_results.csv`
- `{outcome}_optimal_config.pkl`

---

## Key Changes Summary

| Change | File | Description |
|--------|------|-------------|
| Remove Q1 | 04a, 04b | Keep only Q2-Q5 |
| Update Q_STAR | 04b | 0.9 → 0.8 |
| Centralize Q_SETS | Both | Define once, load in 04b |
| Update metric names | 04b | `*_0.9` → `*_0.8` |

---

## Execution Order

```
04a_rank_estimators.ipynb  →  04b_validate_ensembles.ipynb  →  05b_GI_rebuild_top_groups.ipynb
```
