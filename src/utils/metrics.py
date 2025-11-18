"""
Evaluation Metrics for AmEx Default Prediction Competition.

The primary metric is a combination of normalized Gini coefficient and
top-4% default capture rate with weighted samples.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.metrics import roc_auc_score


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray,
                     sample_weight: Optional[np.ndarray] = None) -> float:
    """
    Calculate the Gini coefficient.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        sample_weight: Sample weights

    Returns:
        Gini coefficient (2 * AUC - 1)
    """
    auc = roc_auc_score(y_true, y_pred, sample_weight=sample_weight)
    gini = 2 * auc - 1
    return gini


def top4_capture_rate(y_true: np.ndarray, y_pred: np.ndarray,
                      sample_weight: Optional[np.ndarray] = None) -> float:
    """
    Calculate the default rate captured at top 4% of predictions.

    This represents a Sensitivity/Recall statistic for the highest-risk 4% of customers.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        sample_weight: Sample weights

    Returns:
        Proportion of defaults captured in top 4%
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))

    # Create dataframe for easier manipulation
    df = pd.DataFrame({
        'target': y_true,
        'prediction': y_pred,
        'weight': sample_weight
    })

    # Sort by prediction (descending)
    df = df.sort_values('prediction', ascending=False).reset_index(drop=True)

    # Calculate weighted cumulative sums
    df['weight_cumsum'] = df['weight'].cumsum()
    df['target_weight'] = df['target'] * df['weight']
    df['target_weight_cumsum'] = df['target_weight'].cumsum()

    # Find 4% threshold
    total_weight = df['weight'].sum()
    threshold_weight = 0.04 * total_weight

    # Find the index where cumulative weight exceeds 4%
    top4_idx = (df['weight_cumsum'] <= threshold_weight).sum()

    # Calculate capture rate
    if top4_idx == 0:
        return 0.0

    captured_defaults = df.loc[:top4_idx-1, 'target_weight'].sum()
    total_defaults = df['target_weight'].sum()

    if total_defaults == 0:
        return 0.0

    capture_rate = captured_defaults / total_defaults
    return capture_rate


def amex_metric(y_true: np.ndarray, y_pred: np.ndarray,
                return_components: bool = False) -> float:
    """
    Calculate the AmEx competition metric.

    M = 0.5 * (G + D)

    Where:
    - G is the normalized Gini coefficient with weighted samples
    - D is the default rate captured at 4% with weighted samples
    - Negative class receives 20x weight (due to 5% downsampling)

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities
        return_components: If True, return (metric, gini, capture_rate)

    Returns:
        AmEx metric score (or tuple if return_components=True)
    """
    # Apply class weights: 20x for negative class, 1x for positive
    sample_weight = np.where(y_true == 0, 20, 1)

    # Calculate components
    gini = gini_coefficient(y_true, y_pred, sample_weight)
    capture = top4_capture_rate(y_true, y_pred, sample_weight)

    # Final metric
    metric = 0.5 * (gini + capture)

    if return_components:
        return metric, gini, capture
    return metric


def amex_metric_mod(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Modified AmEx metric (backward compatibility with original implementation).

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities

    Returns:
        AmEx metric score
    """
    return amex_metric(y_true, y_pred)


def normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate normalized Gini coefficient.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities

    Returns:
        Normalized Gini coefficient
    """
    # Apply class weights
    sample_weight = np.where(y_true == 0, 20, 1)
    return gini_coefficient(y_true, y_pred, sample_weight)


# For LightGBM callback
def lgb_amex_metric(y_pred: np.ndarray, y_true) -> Tuple[str, float, bool]:
    """
    LightGBM-compatible metric function.

    Args:
        y_pred: Predicted probabilities
        y_true: LightGBM Dataset object

    Returns:
        Tuple of (metric_name, metric_value, is_higher_better)
    """
    y_true_labels = y_true.get_label()
    metric_value = amex_metric(y_true_labels, y_pred)
    return 'amex', metric_value, True


# For XGBoost callback
def xgb_amex_metric(y_pred: np.ndarray, y_true) -> Tuple[str, float]:
    """
    XGBoost-compatible metric function.

    Args:
        y_pred: Predicted probabilities
        y_true: XGBoost DMatrix object

    Returns:
        Tuple of (metric_name, metric_value)
    """
    y_true_labels = y_true.get_label()
    metric_value = amex_metric(y_true_labels, y_pred)
    return 'amex', metric_value


# For CatBoost callback
class CatBoostAmexMetric:
    """CatBoost-compatible metric class."""

    def get_final_error(self, error: float, weight: float) -> float:
        return error

    def is_max_optimal(self) -> bool:
        return True

    def evaluate(self, approxes, target, weight):
        """Evaluate the metric."""
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        y_pred = np.array(approxes[0])
        y_true = np.array(target)

        # Sigmoid activation for CatBoost logits
        y_pred = 1 / (1 + np.exp(-y_pred))

        metric_value = amex_metric(y_true, y_pred)
        return metric_value, 1.0


if __name__ == "__main__":
    # Test the metrics
    np.random.seed(42)

    # Generate synthetic data
    n_samples = 10000
    y_true = np.random.binomial(1, 0.2, n_samples)  # 20% default rate
    y_pred = np.random.beta(2, 5, n_samples)  # Predictions skewed toward lower values

    # Add some signal
    y_pred = np.where(y_true == 1, y_pred + 0.2, y_pred - 0.1)
    y_pred = np.clip(y_pred, 0, 1)

    # Calculate metrics
    metric, gini, capture = amex_metric(y_true, y_pred, return_components=True)

    print(f"AmEx Metric: {metric:.4f}")
    print(f"Gini Coefficient: {gini:.4f}")
    print(f"Top-4% Capture Rate: {capture:.4f}")
