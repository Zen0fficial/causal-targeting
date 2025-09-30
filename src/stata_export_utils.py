"""
Stata Export Utilities
Handles data type conversion for proper Stata file export
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings

def prepare_dataframe_for_stata(df: pd.DataFrame, 
                              date_columns: List[str] = None,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Prepare a DataFrame for Stata export by handling problematic data types.
    
    Args:
        df: DataFrame to prepare
        date_columns: List of column names that contain date data
        verbose: Whether to print conversion details
        
    Returns:
        DataFrame ready for Stata export
    """
    df_prepared = df.copy()
    
    if verbose:
        print("🔧 Preparing DataFrame for Stata export...")
        print(f"Original shape: {df_prepared.shape}")
    
    # Handle datetime columns
    if date_columns is None:
        date_columns = []
    
    # Auto-detect datetime columns
    datetime_cols = []
    for col in df_prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(df_prepared[col]):
            datetime_cols.append(col)
    
    all_date_cols = list(set(date_columns + datetime_cols))
    
    if all_date_cols and verbose:
        print(f"Converting datetime columns: {all_date_cols}")
    
    for col in all_date_cols:
        if col in df_prepared.columns:
            # Convert datetime to numeric (days since 1960-01-01, Stata's epoch)
            if pd.api.types.is_datetime64_any_dtype(df_prepared[col]):
                # Convert to Stata date format (days since 1960-01-01)
                stata_epoch = pd.Timestamp('1960-01-01')
                df_prepared[col] = (df_prepared[col] - stata_epoch).dt.days
                if verbose:
                    print(f"  ✓ {col}: datetime → numeric (days since 1960-01-01)")
    
    # Handle object columns
    object_cols = df_prepared.select_dtypes(include=['object']).columns
    
    if len(object_cols) > 0 and verbose:
        print(f"Checking object columns: {list(object_cols)}")
    
    for col in object_cols:
        col_data = df_prepared[col]
        
        # Check if column contains only nulls
        if col_data.isna().all():
            df_prepared[col] = 0.0  # Convert all-null columns to numeric
            if verbose:
                print(f"  ✓ {col}: all-null object → numeric (0.0)")
            continue
        
        # Try to convert to numeric
        try:
            numeric_version = pd.to_numeric(col_data, errors='coerce')
            if not numeric_version.isna().all():
                df_prepared[col] = numeric_version.fillna(0)
                if verbose:
                    print(f"  ✓ {col}: object → numeric")
                continue
        except:
            pass
        
        # Convert mixed object columns to string
        try:
            # Convert to string, handling None values
            df_prepared[col] = col_data.astype(str).replace('None', '')
            # Check if all values are empty strings or 'nan'
            if df_prepared[col].isin(['', 'nan', 'NaN']).all():
                df_prepared[col] = 0.0
                if verbose:
                    print(f"  ✓ {col}: empty strings → numeric (0.0)")
            else:
                if verbose:
                    print(f"  ✓ {col}: object → string")
        except Exception as e:
            if verbose:
                print(f"  ⚠️ {col}: Could not convert ({e}), dropping column")
            df_prepared = df_prepared.drop(columns=[col])
    
    # Handle any remaining problematic columns
    for col in df_prepared.columns:
        dtype = df_prepared[col].dtype
        
        # Handle categorical columns
        if pd.api.types.is_categorical_dtype(dtype):
            df_prepared[col] = df_prepared[col].astype(str)
            if verbose:
                print(f"  ✓ {col}: categorical → string")
        
        # Handle complex numbers (not supported by Stata)
        elif pd.api.types.is_complex_dtype(dtype):
            df_prepared[col] = df_prepared[col].real  # Take real part
            if verbose:
                print(f"  ✓ {col}: complex → real")
    
    # Final check - ensure no problematic dtypes remain
    final_dtypes = df_prepared.dtypes
    problematic = []
    
    for col, dtype in final_dtypes.items():
        if dtype == 'object':
            # Check if this object column can be safely exported
            sample_values = df_prepared[col].dropna().head(10)
            if len(sample_values) > 0:
                # Check if all values are strings
                if not all(isinstance(x, (str, type(None))) for x in sample_values):
                    problematic.append(col)
    
    if problematic:
        if verbose:
            print(f"⚠️ Still problematic columns: {problematic}")
            print("Converting to string as last resort...")
        for col in problematic:
            df_prepared[col] = df_prepared[col].astype(str).replace('nan', '')
    
    if verbose:
        print(f"✓ Final shape: {df_prepared.shape}")
        print(f"✓ Data types: {dict(df_prepared.dtypes.value_counts())}")
    
    return df_prepared


def safe_stata_export(df: pd.DataFrame, 
                     output_path: str,
                     date_columns: List[str] = None,
                     version: int = 117,
                     verbose: bool = True) -> bool:
    """
    Safely export DataFrame to Stata format with proper data type handling.
    
    Args:
        df: DataFrame to export
        output_path: Path to save the .dta file
        date_columns: List of column names containing date data
        version: Stata version format (default 117)
        verbose: Whether to print export details
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Prepare DataFrame
        df_prepared = prepare_dataframe_for_stata(df, date_columns, verbose)
        
        # Attempt export
        if verbose:
            print(f"💾 Exporting to: {output_path}")
        
        df_prepared.to_stata(
            output_path, 
            write_index=False, 
            version=version,
            convert_dates=None  # Don't auto-convert dates
        )
        
        if verbose:
            print("✅ Stata export successful!")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ Stata export failed: {e}")
        
        # Fallback: save as CSV
        try:
            csv_path = output_path.replace('.dta', '.csv')
            df_prepared.to_csv(csv_path, index=False)
            if verbose:
                print(f"💾 Saved as CSV instead: {csv_path}")
            return False
        except Exception as csv_error:
            if verbose:
                print(f"❌ CSV fallback also failed: {csv_error}")
            return False


def diagnose_stata_export_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Diagnose potential issues with Stata export.
    
    Args:
        df: DataFrame to diagnose
        
    Returns:
        Dictionary with diagnostic information
    """
    issues = {
        'datetime_columns': [],
        'object_columns': [],
        'null_columns': [],
        'complex_columns': [],
        'categorical_columns': [],
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'recommendations': []
    }
    
    # Check each column
    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_datetime64_any_dtype(dtype):
            issues['datetime_columns'].append(col)
            
        elif dtype == 'object':
            issues['object_columns'].append(col)
            
        elif pd.api.types.is_categorical_dtype(dtype):
            issues['categorical_columns'].append(col)
            
        elif pd.api.types.is_complex_dtype(dtype):
            issues['complex_columns'].append(col)
        
        # Check for all-null columns
        if df[col].isna().all():
            issues['null_columns'].append(col)
    
    # Generate recommendations
    if issues['datetime_columns']:
        issues['recommendations'].append(
            f"Convert datetime columns to numeric: {issues['datetime_columns']}"
        )
    
    if issues['object_columns']:
        issues['recommendations'].append(
            f"Handle object columns: {issues['object_columns']}"
        )
    
    if issues['null_columns']:
        issues['recommendations'].append(
            f"Handle all-null columns: {issues['null_columns']}"
        )
    
    if issues['complex_columns']:
        issues['recommendations'].append(
            f"Convert complex columns: {issues['complex_columns']}"
        )
    
    return issues

