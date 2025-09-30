"""
Data utilities for Turkey Bank Overdraft Analysis
Simplified utilities for data cleaning and processing (no conversion needed)
"""

import pandas as pd
import numpy as np
from typing import List, Optional

def clean_numeric_variables(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """
    Clean numeric variables (replaces "(null)" with NaN, converts to numeric)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    variables : list
        List of variable names to clean
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    for var in variables:
        if var in df_clean.columns:
            # Replace "(null)" strings with NaN
            df_clean[var] = df_clean[var].replace("(null)", np.nan)
            
            # Convert to numeric
            df_clean[var] = pd.to_numeric(df_clean[var], errors='coerce')
            
            # Replace NaN with 0 for specific variables (following Stata code)
            df_clean[var] = df_clean[var].fillna(0)
    
    return df_clean

def create_summary_statistics(df: pd.DataFrame, 
                            variables: List[str],
                            by_group: Optional[str] = None) -> pd.DataFrame:
    """
    Create summary statistics table
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
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
        summary = df.groupby(by_group)[variables].agg(['count', 'mean', 'std', 'min', 'max'])
    else:
        summary = df[variables].agg(['count', 'mean', 'std', 'min', 'max'])
    
    return summary.round(3)