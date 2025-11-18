"""
Feature Aggregation Module.

Aggregates time-series customer data into static features using various statistics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """Aggregate time-series features per customer."""

    def __init__(self,
                 cat_features: List[str],
                 num_agg_stats: Optional[List[str]] = None,
                 cat_agg_stats: Optional[List[str]] = None,
                 advanced_stats: bool = True,
                 n_jobs: int = -1):
        """
        Initialize feature aggregator.

        Args:
            cat_features: List of categorical feature names
            num_agg_stats: Aggregation statistics for numerical features
            cat_agg_stats: Aggregation statistics for categorical features
            advanced_stats: Whether to compute advanced statistics (skew, kurtosis, etc.)
            n_jobs: Number of parallel jobs
        """
        self.cat_features = cat_features
        self.advanced_stats = advanced_stats
        self.n_jobs = n_jobs

        # Default aggregation statistics
        if num_agg_stats is None:
            self.num_agg_stats = ['mean', 'std', 'min', 'max', 'sum', 'last', 'first']
            if advanced_stats:
                self.num_agg_stats.extend(['median', 'skew', 'kurt'])
        else:
            self.num_agg_stats = num_agg_stats

        if cat_agg_stats is None:
            self.cat_agg_stats = ['mean', 'std', 'sum', 'last', 'first', 'nunique', 'count']
        else:
            self.cat_agg_stats = cat_agg_stats

    def _get_numerical_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of numerical features."""
        exclude_cols = ['customer_ID', 'S_2', 'target'] + self.cat_features
        num_features = [col for col in df.columns
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        return num_features

    def _aggregate_numerical(self, df: pd.DataFrame, customer_id_col: str = 'customer_ID') -> pd.DataFrame:
        """
        Aggregate numerical features.

        Args:
            df: Input dataframe with time-series data
            customer_id_col: Customer ID column name

        Returns:
            Aggregated numerical features
        """
        logger.info("Aggregating numerical features...")

        num_features = self._get_numerical_features(df)

        # Define aggregation functions
        agg_dict = {}
        for col in num_features:
            agg_dict[col] = []
            for stat in self.num_agg_stats:
                if stat == 'last':
                    agg_dict[col].append(lambda x: x.iloc[-1] if len(x) > 0 else np.nan)
                elif stat == 'first':
                    agg_dict[col].append(lambda x: x.iloc[0] if len(x) > 0 else np.nan)
                elif stat == 'skew':
                    agg_dict[col].append(pd.Series.skew)
                elif stat == 'kurt':
                    agg_dict[col].append(pd.Series.kurt)
                else:
                    agg_dict[col].append(stat)

        # Aggregate
        agg_df = df.groupby(customer_id_col)[num_features].agg(self.num_agg_stats)

        # Flatten column names
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()

        logger.info(f"Created {len(agg_df.columns)-1} numerical aggregation features")
        return agg_df

    def _aggregate_categorical(self, df: pd.DataFrame, customer_id_col: str = 'customer_ID') -> pd.DataFrame:
        """
        Aggregate categorical features.

        Args:
            df: Input dataframe with time-series data
            customer_id_col: Customer ID column name

        Returns:
            Aggregated categorical features
        """
        logger.info("Aggregating categorical features...")

        # One-hot encode categorical features
        cat_dummies = pd.get_dummies(df[self.cat_features], prefix='cat_onehot')

        # Add customer ID for grouping
        cat_dummies[customer_id_col] = df[customer_id_col].values

        # Aggregate one-hot encoded features
        agg_dict = {col: self.cat_agg_stats for col in cat_dummies.columns if col != customer_id_col}
        agg_cat_df = cat_dummies.groupby(customer_id_col).agg(agg_dict)

        # Flatten column names
        agg_cat_df.columns = ['_'.join(col).strip() for col in agg_cat_df.columns.values]
        agg_cat_df = agg_cat_df.reset_index()

        # Also aggregate original categorical features
        orig_cat_agg = df.groupby(customer_id_col)[self.cat_features].agg(['last', 'nunique', 'count'])
        orig_cat_agg.columns = ['_'.join(col).strip() for col in orig_cat_agg.columns.values]
        orig_cat_agg = orig_cat_agg.reset_index()

        # Merge
        result_df = pd.merge(orig_cat_agg, agg_cat_df, on=customer_id_col, how='left')

        logger.info(f"Created {len(result_df.columns)-1} categorical aggregation features")
        return result_df

    def _aggregate_differences(self, df: pd.DataFrame, customer_id_col: str = 'customer_ID') -> pd.DataFrame:
        """
        Compute and aggregate period-over-period differences.

        Args:
            df: Input dataframe with time-series data
            customer_id_col: Customer ID column name

        Returns:
            Aggregated difference features
        """
        logger.info("Computing difference features...")

        num_features = self._get_numerical_features(df)

        # Compute differences
        diff_df = df.groupby(customer_id_col)[num_features].diff()
        diff_df[customer_id_col] = df[customer_id_col].values

        # Aggregate differences
        agg_dict = {col: self.num_agg_stats for col in num_features}
        agg_diff_df = diff_df.groupby(customer_id_col).agg(agg_dict)

        # Flatten column names
        agg_diff_df.columns = ['diff_' + '_'.join(col).strip() for col in agg_diff_df.columns.values]
        agg_diff_df = agg_diff_df.reset_index()

        logger.info(f"Created {len(agg_diff_df.columns)-1} difference features")
        return agg_diff_df

    def _aggregate_lastk(self, df: pd.DataFrame, k: int, customer_id_col: str = 'customer_ID') -> pd.DataFrame:
        """
        Aggregate features using only last k records per customer.

        Args:
            df: Input dataframe with time-series data
            k: Number of last records to use
            customer_id_col: Customer ID column name

        Returns:
            Aggregated features from last k records
        """
        logger.info(f"Aggregating last {k} records...")

        # Get last k records per customer
        df_lastk = df.groupby(customer_id_col).tail(k)

        # Aggregate numerical features
        num_features = self._get_numerical_features(df_lastk)
        agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in num_features}
        agg_df = df_lastk.groupby(customer_id_col).agg(agg_dict)

        # Flatten column names
        agg_df.columns = [f'last{k}_' + '_'.join(col).strip() for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()

        logger.info(f"Created {len(agg_df.columns)-1} last-{k} features")
        return agg_df

    def fit_transform(self, df: pd.DataFrame,
                      customer_id_col: str = 'customer_ID',
                      compute_diff: bool = True,
                      lastk_variants: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Compute all aggregation features.

        Args:
            df: Input dataframe with time-series data
            customer_id_col: Customer ID column name
            compute_diff: Whether to compute difference features
            lastk_variants: List of k values for last-k aggregations

        Returns:
            Dataframe with all aggregated features
        """
        logger.info("Starting feature aggregation...")

        # Base aggregations
        num_agg = self._aggregate_numerical(df, customer_id_col)
        cat_agg = self._aggregate_categorical(df, customer_id_col)

        # Merge base aggregations
        result_df = pd.merge(num_agg, cat_agg, on=customer_id_col, how='left')

        # Difference features
        if compute_diff:
            diff_agg = self._aggregate_differences(df, customer_id_col)
            result_df = pd.merge(result_df, diff_agg, on=customer_id_col, how='left')

        # Last-k variants
        if lastk_variants is not None:
            for k in lastk_variants:
                lastk_agg = self._aggregate_lastk(df, k, customer_id_col)
                result_df = pd.merge(result_df, lastk_agg, on=customer_id_col, how='left')

        logger.info(f"Total aggregated features: {len(result_df.columns)-1}")
        return result_df

    def transform(self, df: pd.DataFrame,
                  customer_id_col: str = 'customer_ID',
                  compute_diff: bool = True,
                  lastk_variants: Optional[List[int]] = None) -> pd.DataFrame:
        """Transform method (same as fit_transform for this class)."""
        return self.fit_transform(df, customer_id_col, compute_diff, lastk_variants)
