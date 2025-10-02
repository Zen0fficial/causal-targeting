"""
Statistical analysis utilities for Turkey Bank Overdraft Analysis
Functions for regressions, table generation, and statistical tests
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from typing import Dict, List, Optional, Union, Tuple
import warnings

def run_ols_regression(data: pd.DataFrame, 
                      dependent_var: str,
                      independent_vars: List[str],
                      robust: bool = True,
                      cluster_var: Optional[str] = None) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression with optional robust/clustered standard errors
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    dependent_var : str
        Name of dependent variable
    independent_vars : list
        List of independent variable names
    robust : bool
        Whether to use robust standard errors
    cluster_var : str, optional
        Variable to cluster standard errors on
        
    Returns:
    --------
    statsmodels regression results
    """
    # Remove missing values
    vars_to_use = [dependent_var] + independent_vars
    if cluster_var:
        vars_to_use.append(cluster_var)
        
    data_clean = data[vars_to_use].dropna()
    
    # Prepare variables
    y = data_clean[dependent_var]
    X = data_clean[independent_vars]
    X = sm.add_constant(X)  # Add intercept
    
    # Fit model
    model = sm.OLS(y, X)
    
    if cluster_var and cluster_var in data_clean.columns:
        # Clustered standard errors
        results = model.fit(cov_type='cluster', cov_kwds={'groups': data_clean[cluster_var]})
    elif robust:
        # Robust standard errors
        results = model.fit(cov_type='HC1')
    else:
        # Standard errors
        results = model.fit()
    
    return results

def create_regression_table(results_list: List[sm.regression.linear_model.RegressionResultsWrapper],
                           model_names: Optional[List[str]] = None,
                           decimal_places: int = 3) -> pd.DataFrame:
    """
    Create a formatted regression table from multiple regression results
    
    Parameters:
    -----------
    results_list : list
        List of statsmodels regression results
    model_names : list, optional
        Names for each model column
    decimal_places : int
        Number of decimal places to display
        
    Returns:
    --------
    pd.DataFrame
        Formatted regression table
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(results_list))]
    
    # Initialize table
    table_data = {}
    
    for i, (results, name) in enumerate(zip(results_list, model_names)):
        col_data = {}
        
        # Extract coefficients and standard errors
        for var in results.params.index:
            if var == 'const':
                var_name = 'Constant'
            else:
                var_name = var
                
            # Coefficient
            coef = results.params[var]
            se = results.bse[var]
            pval = results.pvalues[var]
            
            # Add significance stars
            stars = ''
            if pval < 0.01:
                stars = '***'
            elif pval < 0.05:
                stars = '**'
            elif pval < 0.1:
                stars = '*'
            
            col_data[var_name] = f"{coef:.{decimal_places}f}{stars}"
            col_data[f"{var_name}_se"] = f"({se:.{decimal_places}f})"
        
        # Add model statistics
        col_data['N'] = int(results.nobs)
        col_data['R-squared'] = f"{results.rsquared:.{decimal_places}f}"
        
        table_data[name] = col_data
    
    # Convert to DataFrame
    table_df = pd.DataFrame(table_data)
    
    return table_df

def balance_test(data: pd.DataFrame,
                treatment_var: str,
                outcome_vars: List[str],
                control_vars: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Perform balance tests (orthogonality checks) for RCT
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    treatment_var : str
        Treatment variable name
    outcome_vars : list
        List of variables to test balance on
    control_vars : list, optional
        Control variables to include in regression
        
    Returns:
    --------
    pd.DataFrame
        Balance test results table
    """
    results = []
    
    for var in outcome_vars:
        if var not in data.columns:
            continue
            
        # Prepare regression formula
        if control_vars:
            formula = f"{var} ~ {treatment_var} + " + " + ".join(control_vars)
        else:
            formula = f"{var} ~ {treatment_var}"
        
        try:
            # Run regression
            model = smf.ols(formula, data=data).fit(cov_type='HC1')
            
            # Extract treatment coefficient
            treatment_coef = model.params[treatment_var]
            treatment_se = model.bse[treatment_var]
            treatment_pval = model.pvalues[treatment_var]
            
            # Calculate control mean
            control_mean = data[data[treatment_var] == 0][var].mean()
            
            results.append({
                'Variable': var,
                'Control_Mean': control_mean,
                'Treatment_Effect': treatment_coef,
                'Std_Error': treatment_se,
                'P_Value': treatment_pval,
                'N': int(model.nobs)
            })
        except Exception as e:
            print(f"Error in balance test for {var}: {e}")
            continue
    
    return pd.DataFrame(results)

def summary_statistics(data: pd.DataFrame,
                      variables: List[str],
                      by_group: Optional[str] = None) -> pd.DataFrame:
    """
    Generate summary statistics table
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    variables : list
        Variables to summarize
    by_group : str, optional
        Variable to group by
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics table
    """
    if by_group:
        summary = data.groupby(by_group)[variables].agg(['count', 'mean', 'std', 'min', 'max'])
    else:
        summary = data[variables].agg(['count', 'mean', 'std', 'min', 'max'])
    
    return summary.round(3)

def export_to_excel(tables_dict: Dict[str, pd.DataFrame],
                   filename: str,
                   output_dir: str) -> None:
    """
    Export multiple tables to Excel with separate sheets
    
    Parameters:
    -----------
    tables_dict : dict
        Dictionary with sheet names as keys and DataFrames as values
    filename : str
        Output filename
    output_dir : str
        Output directory
    """
    from pathlib import Path
    
    output_path = Path(output_dir) / filename
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in tables_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=True)
    
    print(f"✓ Exported {len(tables_dict)} tables to: {output_path}")

def heterogeneous_effects_analysis(data: pd.DataFrame,
                                 dependent_var: str,
                                 treatment_var: str,
                                 heterogeneity_var: str,
                                 control_vars: Optional[List[str]] = None) -> Dict:
    """
    Analyze heterogeneous treatment effects
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    dependent_var : str
        Outcome variable
    treatment_var : str
        Treatment variable
    heterogeneity_var : str
        Variable to interact with treatment
    control_vars : list, optional
        Control variables
        
    Returns:
    --------
    dict
        Results including main effects and interaction
    """
    # Create interaction term
    interaction_var = f"{treatment_var}_x_{heterogeneity_var}"
    data[interaction_var] = data[treatment_var] * data[heterogeneity_var]
    
    # Prepare formula
    formula_vars = [treatment_var, heterogeneity_var, interaction_var]
    if control_vars:
        formula_vars.extend(control_vars)
    
    formula = f"{dependent_var} ~ " + " + ".join(formula_vars)
    
    # Run regression
    try:
        model = smf.ols(formula, data=data).fit(cov_type='HC1')
        
        return {
            'model': model,
            'treatment_effect': model.params[treatment_var],
            'interaction_effect': model.params[interaction_var],
            'treatment_pval': model.pvalues[treatment_var],
            'interaction_pval': model.pvalues[interaction_var]
        }
    except Exception as e:
        print(f"Error in heterogeneous effects analysis: {e}")
        return None

def winsorize_variables(data: pd.DataFrame,
                       variables: List[str],
                       percentile: float = 0.01) -> pd.DataFrame:
    """
    Winsorize variables at specified percentiles
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    variables : list
        Variables to winsorize
    percentile : float
        Percentile to winsorize at (e.g., 0.01 for 1% and 99%)
        
    Returns:
    --------
    pd.DataFrame
        Dataset with winsorized variables
    """
    data_winsorized = data.copy()
    
    for var in variables:
        if var in data.columns:
            lower_bound = data[var].quantile(percentile)
            upper_bound = data[var].quantile(1 - percentile)
            
            data_winsorized[var] = data[var].clip(lower=lower_bound, upper=upper_bound)
            print(f"✓ Winsorized {var}: bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return data_winsorized


