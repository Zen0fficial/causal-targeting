"""
Configuration file for the simulated causal-targeting project.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output"
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"

# Data directories
ANALYSIS_DATA_DIR = DATA_ROOT / "analysis"

# Output directories
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"

# Local synthetic data source
ORIGINAL_DATA_ROOT = DATA_ROOT
ORIGINAL_DATA_DIR = ANALYSIS_DATA_DIR
ORIGINAL_CODEBOOKS_DIR = DATA_ROOT / "codebooks"

# File paths retained for compatibility with helper scripts
DATASETS = {
    'analysis': ANALYSIS_DATA_DIR / "analysis_df.csv",
}

# Analysis configuration
STUDY_TITLE = "Simulated causal targeting benchmark"
STUDY_DATE = "Synthetic"
REPLICATION_VERSION = "Python Simulation Clone"

# Table configuration - mapping Stata scripts to table names
TABLE_CONFIG = {
    't1_orthogonality': {
        'title': 'Table 1: Orthogonality Checks',
        'sheet_name': 'T1_orth'
    },
    't2_direct1': {
        'title': 'Table 2: Effects of First Two Overdraft Messages on Overdraft Usage, During Experiment',
        'sheet_name': 'T2_dir1'
    },
    't2a_direct1_quantiles': {
        'title': 'Table 2a: Effects by Quantile',
        'sheet_name': 'T2a_dir1_quant'
    },
    't3_debbp': {
        'title': 'Table 3: Effects of Auto-Debit and Debit Card Messages',
        'sheet_name': 'T3_bpdeb'
    },
    't4_direct_long': {
        'title': 'Table 4: Effects After Experiment',
        'sheet_name': 'T4_dir_long'
    },
    't4a_direct_long_dyn': {
        'title': 'Table 4a: Monthly Effects After Experiment',
        'sheet_name': 'T4a_dir_long_dyn'
    },
    't5_hte_prior': {
        'title': 'Table 5: Heterogeneous Treatment Effects by Prior Use',
        'sheet_name': 'T5_hte_prior'
    },
    't6_hte_prebalance': {
        'title': 'Table 6: Heterogeneous Treatment Effects by Baseline Balance',
        'sheet_name': 'T6_hte_basebal'
    }
}

# Ensure directories exist
for directory in [ANALYSIS_DATA_DIR, TABLES_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default configuration for transparent feature selection
FEATURE_SELECTION_DEFAULT = {
    "enabled": False,
    "strategy": "filters+correlation+univariate",
    "params": {
        "min_variance": 1e-8,
        "max_missing": 0.3,
        "corr_threshold": 0.95,
        "univariate": {"k": 200, "kind": "mutual_info_regression"},
        "double_selection": {"alpha": "cv", "max_iter": 2000},
        "stability": {"n_subsamples": 50, "sample_frac": 0.5, "threshold": 0.6, "max_iter": 2000},
    },
    "random_state": 0,
    "report_path": None,
}
