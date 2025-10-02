"""
Configuration file for Turkey Bank Overdraft Analysis
Recreating "Unshrouding: Evidence from Bank Overdrafts in Turkey" in Python
"""

import os
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

# Original data source - pointing directly to Turkey Dataverse Files
ORIGINAL_DATA_ROOT = Path("/Users/zenofficial/Documents/statistics/pcs/Turkey Dataverse Files/Turkey Dataverse Files/Data")
ORIGINAL_DATA_DIR = ORIGINAL_DATA_ROOT
ORIGINAL_CODEBOOKS_DIR = Path("/Users/zenofficial/Documents/statistics/pcs/Turkey Dataverse Files/Turkey Dataverse Files/Codebooks")

# File paths for main datasets - pointing to original files
DATASETS = {
    'all': ORIGINAL_DATA_DIR / "bankdata" / "clean" / "all.dta",
    'dailydata': ORIGINAL_DATA_DIR / "dailydata" / "dailydata.dta",
    'campaign_and_postcampaign': ORIGINAL_DATA_DIR / "analysis" / "campaign_and_postcampaign_and_dailydata.dta",
    'campaign_and_postcampaign_controlmerged': ORIGINAL_DATA_DIR / "analysis" / "campaign_and_postcampaign_and_dailydata_controlmerged.dta",
    'orthogonality': ORIGINAL_DATA_DIR / "analysis" / "orthogonality.dta",
    'data_group': ORIGINAL_DATA_DIR / "analysis" / "data_group.dta",
    'data_group_interactions': ORIGINAL_DATA_DIR / "analysis" / "data_group_interactions.dta"
}

# Analysis configuration
STUDY_TITLE = "Unshrouding: Evidence from Bank Overdrafts in Turkey"
STUDY_DATE = "June 2017"
REPLICATION_VERSION = "Python Replication"

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
