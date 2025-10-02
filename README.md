# Turkey Bank Overdraft Analysis - Python Replication

This is a Python/Jupyter notebook replication of the study:

**"Unshrouding: Evidence from Bank Overdrafts in Turkey"** (June 2017)

Originally implemented in Stata, this version recreates the entire analysis pipeline using modern Python tools.

## Project Structure

```
turkey_python_analysis/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config/
│   └── config.py             # Configuration and paths
├── src/
│   ├── data_utils.py         # Data processing utilities
│   └── analysis_utils.py     # Statistical analysis utilities
├── data/
│   └── analysis/             # Analysis datasets (created by notebooks)
├── notebooks/
│   ├── 00_master_analysis.ipynb           # Master notebook (equivalent to 0_master_all.do)
│   ├── data_cleaning/
│   │   ├── 02_orthogonality_data.ipynb   # Equivalent to data7_orthogonality.do
│   │   └── 03_main_tables_data.ipynb     # Equivalent to data8_maintables.do
│   ├── analysis/
│   │   ├── 04_table1_orthogonality.ipynb      # t1_orthogonality.do
│   │   ├── 05_table2_direct_effects.ipynb     # t2_direct1.do
│   │   ├── 06_table2a_quantiles.ipynb         # t2a_direct1_quantiles.do
│   │   ├── 07_table3_debit_effects.ipynb      # t3_debbp.do
│   │   ├── 08_table4_longterm_effects.ipynb   # t4_direct_long.do
│   │   ├── 09_table4a_monthly_effects.ipynb   # t4a_direct_long_dyn.do
│   │   ├── 10_table5_hte_prior.ipynb          # t5_hte_prior.do
│   │   └── 11_table6_hte_balance.ipynb        # t6_hte_prebalance.do
│   └── figures/
│       ├── 12_descriptive_figures.ipynb       # Describe_Control.do
│       └── 13_appendix_figures.ipynb          # Additional figures
└── output/
    ├── tables/               # Generated tables (Excel format)
    └── figures/             # Generated figures (PNG/PDF)
```

## Setup Instructions

### 1. Environment Setup

```bash
# Activate the existing virtual environment
source /Users/zenofficial/Documents/statistics/pcs/venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Running the Analysis

The analysis follows a sequential workflow:

1. **Data Loading**: Load data directly from original Stata files
2. **Data Cleaning**: Recreate the data cleaning pipeline
3. **Analysis**: Generate all tables and figures

Start with the master notebook:
```bash
jupyter lab notebooks/00_master_analysis.ipynb
```

### 3. Original vs Python Mapping

| Original Stata File | Python Notebook | Purpose |
|---------------------|-----------------|---------|
| `0_master_all.do` | `00_master_analysis.ipynb` | Master control script |
| `data7_orthogonality.do` | `02_orthogonality_data.ipynb` | Create balance check dataset |
| `data8_maintables.do` | `03_main_tables_data.ipynb` | Create main analysis dataset |
| `t1_orthogonality.do` | `04_table1_orthogonality.ipynb` | Table 1: Balance checks |
| `t2_direct1.do` | `05_table2_direct_effects.ipynb` | Table 2: Direct effects |
| `t3_debbp.do` | `07_table3_debit_effects.ipynb` | Table 3: Debit effects |
| And so on... | | |

## Key Features

- **Exact Replication**: Recreates all original tables and figures
- **Modern Tools**: Uses pandas, statsmodels, and scikit-learn
- **Interactive**: Jupyter notebooks for exploration and documentation
- **Reproducible**: Clear dependency management and configuration
- **Extensible**: Modular structure for easy modification and extension

## Study Overview

This study examines how different types of messages affect customer overdraft usage at a Turkish bank:

- **Randomized Controlled Trial** with multiple treatment arms
- **Main Outcomes**: Overdraft usage rates and amounts
- **Key Findings**: Messages can significantly influence financial behavior
- **Time Period**: September 2011 - December 2012

## Data Sources

The analysis uses the following main datasets:
- `all.dta`: Main customer-level dataset
- `dailydata.dta`: Daily transaction data
- `campaign_and_postcampaign_and_dailydata.dta`: Treatment period data
- Various processed datasets for specific analyses

All data is de-identified and available through the original study's dataverse.


