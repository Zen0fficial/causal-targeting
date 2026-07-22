"""
Descriptive Statistics Table for Lasso-Selected Covariates
===========================================================

Generates a professional LaTeX table with:
- Variable abbreviation and description
- Distribution in Control vs Treatment groups
- Grouped sections (Demographics, Financial History, etc.)

Style: booktabs with italicized variable names
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Variable info: (abbreviation, full description, group)
VARIABLE_INFO = {
    # Randomization Strata
    'strat_5': ('strat\\_5', 'male, married, low OD limit, other city, unknown balance', 'Randomization Strata'),
    'strat_6': ('strat\\_6', 'male, married, low OD limit, other city, N/A balance', 'Randomization Strata'),
    'strat_8': ('strat\\_8', 'male, married, low OD limit, other city, below-median balance', 'Randomization Strata'),
    'strat_61': ('strat\\_61', 'male, unmarried, low OD limit, unknown city, N/A balance', 'Randomization Strata'),
    'strat_62': ('strat\\_62', 'male, unmarried, low OD limit, unknown city, zero balance', 'Randomization Strata'),
    'strat_66': ('strat\\_66', 'male, unmarried, low OD limit, other city, zero balance', 'Randomization Strata'),
    'strat_70': ('strat\\_70', 'male, unmarried, low OD limit, Istanbul, zero balance', 'Randomization Strata'),
    'strat_162': ('strat\\_162', 'female, married, low OD limit, unknown city, below-median balance', 'Randomization Strata'),
    
    # Prior Usage  
    'htefa': ('htefa', 'used overdraft in 6 months pre-treatment', 'Prior Usage'),
    'htebal_missing': ('htebal\\_missing', 'baseline account balance missing', 'Prior Usage'),
    
    # Financial Characteristics
    'assets': ('assets', 'avg monthly assets, Sept 2011--Aug 2012', 'Financial Characteristics'),
    'debt': ('debt', 'avg monthly debt, Sept 2011--Aug 2012', 'Financial Characteristics'),
    'minbal': ('minbal', 'minimum deposit balance pre-treatment', 'Financial Characteristics'),
    'creditcard': ('creditcard', 'has credit card pre-randomization', 'Financial Characteristics'),
    
    # Outcome
    'fausebal': ('fausebal', 'overdraft usage (outcome)', 'Outcome Variable'),
}

# Define group order
GROUP_ORDER = ['Outcome Variable', 'Randomization Strata', 'Prior Usage', 'Financial Characteristics']


def format_binary(series, n_total):
    """Format binary variable as 'count (percentage)'."""
    count = int(series.sum())
    pct = 100 * count / n_total if n_total > 0 else 0
    return f"{count} ({pct:.1f})"


def format_continuous(series):
    """Format continuous variable as mean (SD)."""
    return f"{series.mean():.1f} ({series.std():.1f})"


def generate_latex_table(df, feature_cols, treatment_col='TREATED'):
    """Generate professional LaTeX table with booktabs."""
    
    df_control = df[df[treatment_col] == 0]
    df_treat = df[df[treatment_col] == 1]
    n_control = len(df_control)
    n_treat = len(df_treat)
    
    # Build table content by group
    grouped_rows = {g: [] for g in GROUP_ORDER}
    
    for col in feature_cols:
        if col not in df.columns or col not in VARIABLE_INFO:
            continue
        
        abbr, desc, group = VARIABLE_INFO[col]
        unique_vals = df[col].dropna().unique()
        is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})
        
        if is_binary:
            ctrl_str = format_binary(df_control[col], n_control)
            treat_str = format_binary(df_treat[col], n_treat)
            var_text = f"Whether \\textit{{{desc}}} ({abbr}=1)"
            grouped_rows[group].append(f"\\quad {var_text} & {ctrl_str} & {treat_str} \\\\")
        else:
            # Format continuous variable as mean (SD)
            ctrl_str = format_continuous(df_control[col])
            treat_str = format_continuous(df_treat[col])
            
            var_text = f"\\textit{{{desc}}} ({abbr})"
            grouped_rows[group].append(f"\\quad {var_text} & {ctrl_str} & {treat_str} \\\\")
    
    # Build LaTeX
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Descriptive Statistics for Lasso-Selected Covariates}")
    latex.append(r"\label{tab:descriptive_lasso}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Covariate (ABBRV)} & \textbf{Control No. (\%)} & \textbf{Treatment No. (\%)} \\")
    latex.append(r"\midrule")
    
    # Overall population
    ctrl_pct = 100 * n_control / (n_control + n_treat)
    treat_pct = 100 * n_treat / (n_control + n_treat)
    latex.append(f"Overall population & {n_control} ({ctrl_pct:.1f}) & {n_treat} ({treat_pct:.1f}) \\\\")
    
    # Add grouped sections
    for group in GROUP_ORDER:
        if grouped_rows[group]:
            latex.append(f"\\textbf{{{group}}} & & \\\\")
            for row in grouped_rows[group]:
                latex.append(row)
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\begin{tablenotes}")
    latex.append(r"\small")
    latex.append(r"\item Note: For binary variables, values show count (\% of group). For continuous variables, values show mean (SD).")
    latex.append(r"\item Low OD limit = overdraft limit $<$ 1/2 minimum wage.")
    latex.append(r"\end{tablenotes}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)


def generate_csv_table(df, feature_cols, treatment_col='TREATED'):
    """Generate CSV table for reference."""
    df_control = df[df[treatment_col] == 0]
    df_treat = df[df[treatment_col] == 1]
    n_control = len(df_control)
    n_treat = len(df_treat)
    
    rows = [{'Group': '', 'Variable': 'Overall population', 'Abbreviation': '', 
             'Control': f"{n_control} ({100*n_control/(n_control+n_treat):.1f}%)",
             'Treatment': f"{n_treat} ({100*n_treat/(n_control+n_treat):.1f}%)"}]
    
    for col in feature_cols:
        if col not in df.columns or col not in VARIABLE_INFO:
            continue
        
        abbr, desc, group = VARIABLE_INFO[col]
        unique_vals = df[col].dropna().unique()
        is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})
        
        if is_binary:
            ctrl_str = format_binary(df_control[col], n_control) + "%"
            treat_str = format_binary(df_treat[col], n_treat) + "%"
        else:
            ctrl_str = format_continuous(df_control[col])
            treat_str = format_continuous(df_treat[col])
        
        rows.append({'Group': group, 'Variable': desc, 'Abbreviation': abbr,
                     'Control': ctrl_str, 'Treatment': treat_str})
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent.parent / "output" / "analysis" / "fausebal"
    TRAINVAL_PATH = DATA_DIR / "trainval_data.csv"
    HOLDOUT_PATH = DATA_DIR / "holdout_data.csv"
    
    print(f"Loading training data from {TRAINVAL_PATH}...")
    df_train = pd.read_csv(TRAINVAL_PATH)
    print(f"Loaded {len(df_train)} training rows")
    
    
    # Use only trainval data for consistency with ATE analysis
    df = df_train
    feature_cols = [c for c in df.columns if c not in ['TREATED']]
    print(f"Using trainval only: {len(df)} rows, {len(feature_cols)} features")
    
    # Generate and save LaTeX
    latex_content = generate_latex_table(df, feature_cols)
    latex_path = Path(__file__).parent.parent / "output" / "tables" / "descriptive_table_lasso.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"\n✓ Saved LaTeX to {latex_path}")
    
    # Generate and save CSV
    csv_df = generate_csv_table(df, feature_cols)
    csv_path = latex_path.with_suffix('.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV to {csv_path}")
    
    # Print LaTeX for preview
    print("\n" + "=" * 80)
    print("LATEX OUTPUT:")
    print("=" * 80)
    print(latex_content)
