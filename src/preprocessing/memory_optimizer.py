"""
Memory Optimization Utilities for Large Datasets.

This module provides functions to dramatically reduce memory usage of pandas DataFrames
by optimizing data types. Critical for handling 50GB+ datasets.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def reduce_mem_usage(df: pd.DataFrame,
                     verbose: bool = True,
                     exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Reduce memory usage of DataFrame by downcasting numeric types.

    This function can reduce memory usage by 50-80% for typical datasets.

    Args:
        df: Input DataFrame
        verbose: Print memory reduction info
        exclude_cols: Columns to exclude from optimization

    Returns:
        Memory-optimized DataFrame

    Example:
        >>> df = pd.DataFrame({'col1': [1, 2, 3] * 1000000})
        >>> df = reduce_mem_usage(df)
        Memory usage decreased to 0.38 MB (75.0% reduction)
    """
    if exclude_cols is None:
        exclude_cols = []

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        if col in exclude_cols:
            continue

        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            # Integer optimization
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            # Float optimization
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Try to convert object to category
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
                   f'({reduction:.1f}% reduction)')

    return df


def optimize_dtypes(df: pd.DataFrame,
                    cat_features: Optional[List[str]] = None,
                    date_features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Optimize data types for AmEx dataset specifically.

    Args:
        df: Input DataFrame
        cat_features: Categorical feature columns
        date_features: Date feature columns

    Returns:
        Optimized DataFrame
    """
    logger.info("Optimizing data types...")

    # Categorical features
    if cat_features:
        for col in cat_features:
            if col in df.columns:
                df[col] = df[col].astype('category')

    # Date features
    if date_features:
        for col in date_features:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

    # Reduce memory for all other columns
    df = reduce_mem_usage(df, exclude_cols=(cat_features or []) + (date_features or []))

    return df


def get_optimal_dtypes() -> Dict[str, str]:
    """
    Get optimal data types for AmEx dataset columns.

    This can be used with pd.read_csv(dtype=...) to reduce memory during loading.

    Returns:
        Dictionary mapping column names to optimal dtypes
    """
    # Based on AmEx dataset analysis
    dtypes = {
        'customer_ID': 'object',
        'S_2': 'object',  # Will convert to datetime later

        # Categorical features (known from competition)
        'B_30': 'category',
        'B_38': 'category',
        'D_63': 'category',
        'D_64': 'category',
        'D_66': 'category',
        'D_68': 'category',
        'D_114': 'category',
        'D_116': 'category',
        'D_117': 'category',
        'D_120': 'category',
        'D_126': 'category',
    }

    # Most numeric features can be float32
    # This will be updated dynamically based on actual ranges

    return dtypes


class DataTypeOptimizer:
    """
    Class to manage data type optimization across pipeline.

    Tracks optimal dtypes discovered during preprocessing for reuse.
    """

    def __init__(self):
        self.optimal_dtypes = {}

    def fit(self, df: pd.DataFrame) -> 'DataTypeOptimizer':
        """Learn optimal dtypes from DataFrame."""
        self.optimal_dtypes = {}

        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's actually categorical
                num_unique = df[col].nunique()
                if num_unique / len(df) < 0.5:
                    self.optimal_dtypes[col] = 'category'
                else:
                    self.optimal_dtypes[col] = 'object'

            elif pd.api.types.is_integer_dtype(df[col]):
                c_min, c_max = df[col].min(), df[col].max()
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    self.optimal_dtypes[col] = 'int8'
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    self.optimal_dtypes[col] = 'int16'
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    self.optimal_dtypes[col] = 'int32'
                else:
                    self.optimal_dtypes[col] = 'int64'

            elif pd.api.types.is_float_dtype(df[col]):
                c_min, c_max = df[col].min(), df[col].max()
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    self.optimal_dtypes[col] = 'float16'
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    self.optimal_dtypes[col] = 'float32'
                else:
                    self.optimal_dtypes[col] = 'float64'

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned dtypes to DataFrame."""
        for col, dtype in self.optimal_dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert {col} to {dtype}: {e}")

        return df

    def save(self, filepath: str) -> None:
        """Save learned dtypes to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.optimal_dtypes, f)

    def load(self, filepath: str) -> 'DataTypeOptimizer':
        """Load learned dtypes from file."""
        import json
        with open(filepath, 'r') as f:
            self.optimal_dtypes = json.load(f)
        return self


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create sample data
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 1000000),
        'float_col': np.random.randn(1000000),
        'cat_col': np.random.choice(['A', 'B', 'C'], 1000000)
    })

    print(f"Original memory: {df.memory_usage().sum() / 1024**2:.2f} MB")

    # Optimize
    df_opt = reduce_mem_usage(df)

    print(f"Optimized memory: {df_opt.memory_usage().sum() / 1024**2:.2f} MB")
