#!/usr/bin/env python
"""
Main training script for AmEx Default Prediction.

This script orchestrates the entire pipeline:
1. Data preprocessing
2. Feature engineering
3. Model training
4. Ensemble creation
5. Submission generation

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --models lightgbm xgboost
    python train.py --config configs/config.yaml --skip-preprocessing
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.config import load_config, save_config
from src.utils.logger import setup_logging, get_logger
from src.utils.seed import seed_everything
from src.utils.timer import Timer

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AmEx Default Prediction Training Pipeline')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Specific models to train (e.g., lightgbm xgboost catboost)'
    )

    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing step'
    )

    parser.add_argument(
        '--skip-feature-engineering',
        action='store_true',
        help='Skip feature engineering step'
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step'
    )

    parser.add_argument(
        '--skip-ensemble',
        action='store_true',
        help='Skip ensemble creation step'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode with smaller data'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed'
    )

    return parser.parse_args()


def preprocess_data(config):
    """Run data preprocessing pipeline."""
    logger.info("=" * 80)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 80)

    # Import here to avoid loading unnecessary modules
    from src.preprocessing.denoise import denoise_data

    with Timer("Data Preprocessing"):
        train_df, test_df = denoise_data(config)

    logger.info(f"Train shape after preprocessing: {train_df.shape}")
    logger.info(f"Test shape after preprocessing: {test_df.shape}")

    return train_df, test_df


def engineer_features(config, train_df, test_df):
    """Run feature engineering pipeline."""
    logger.info("=" * 80)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 80)

    from src.features import FeatureAggregator, TemporalFeatureEngineer

    # Initialize feature engineers
    aggregator = FeatureAggregator(
        cat_features=config.data.cat_features,
        advanced_stats=config.feature_engineering.get('advanced_stats', True)
    )

    with Timer("Feature Aggregation"):
        train_features = aggregator.fit_transform(
            train_df,
            compute_diff=config.feature_engineering.temporal.use_diff_features,
            lastk_variants=config.feature_engineering.temporal.lastk_variants
        )

        test_features = aggregator.transform(
            test_df,
            compute_diff=config.feature_engineering.temporal.use_diff_features,
            lastk_variants=config.feature_engineering.temporal.lastk_variants
        )

    logger.info(f"Train features shape: {train_features.shape}")
    logger.info(f"Test features shape: {test_features.shape}")

    # Save features
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features.to_feather(output_dir / "train_features.feather")
    test_features.to_feather(output_dir / "test_features.feather")

    logger.info("Features saved to disk")

    return train_features, test_features


def train_models(config, train_features, train_labels, test_features, model_names: Optional[List[str]] = None):
    """Train models."""
    logger.info("=" * 80)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("=" * 80)

    from src.training.trainer import ModelTrainer

    # Initialize trainer
    trainer = ModelTrainer(config)

    # Determine which models to train
    if model_names is None:
        model_names = [name for name, cfg in config.models.items() if cfg.get('enabled', False)]

    logger.info(f"Training models: {model_names}")

    # Train models
    results = {}
    for model_name in model_names:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*80}\n")

        with Timer(f"{model_name} Training"):
            result = trainer.train_model(
                model_name=model_name,
                train_features=train_features,
                train_labels=train_labels,
                test_features=test_features
            )

        results[model_name] = result

        logger.info(f"\n{model_name} CV Score: {result['cv_score']:.6f}")
        logger.info(f"{model_name} OOF saved to: {result['oof_path']}")
        logger.info(f"{model_name} Predictions saved to: {result['pred_path']}")

    return results


def create_ensemble(config, model_results):
    """Create ensemble predictions."""
    logger.info("=" * 80)
    logger.info("STEP 4: ENSEMBLE CREATION")
    logger.info("=" * 80)

    from src.ensemble.weighted_ensemble import WeightedEnsemble

    ensemble = WeightedEnsemble(config)

    with Timer("Ensemble Creation"):
        submission = ensemble.create_ensemble(model_results)

    # Save submission
    output_dir = Path(config.paths.output_dir)
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    logger.info(f"Final submission saved to: {submission_path}")

    return submission


def main():
    """Main training pipeline."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line args
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.seed:
        config.general.seed = args.seed
    if args.debug:
        config.general.debug = True

    # Setup logging
    log_dir = Path(config.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=str(log_dir), level=logging.DEBUG if args.debug else logging.INFO)

    logger.info("="*80)
    logger.info("AMEX DEFAULT PREDICTION - TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {config.paths.output_dir}")
    logger.info(f"Random seed: {config.general.seed}")
    logger.info("="*80)

    # Set random seed
    seed_everything(config.general.seed)

    # Save configuration
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(output_dir / "config.yaml"))

    try:
        # Step 1: Preprocessing
        if not args.skip_preprocessing:
            train_df, test_df = preprocess_data(config)
        else:
            logger.info("Skipping preprocessing (loading from disk)")
            import pandas as pd
            train_df = pd.read_feather(config.paths.data_dir + "/train.feather")
            test_df = pd.read_feather(config.paths.data_dir + "/test.feather")

        # Step 2: Feature Engineering
        if not args.skip_feature_engineering:
            train_features, test_features = engineer_features(config, train_df, test_df)
        else:
            logger.info("Skipping feature engineering (loading from disk)")
            import pandas as pd
            train_features = pd.read_feather(output_dir / "train_features.feather")
            test_features = pd.read_feather(output_dir / "test_features.feather")

        # Load labels
        import pandas as pd
        train_labels = pd.read_csv(config.paths.train_labels)

        # Step 3: Model Training
        if not args.skip_training:
            model_results = train_models(config, train_features, train_labels, test_features, args.models)
        else:
            logger.info("Skipping model training")
            model_results = {}

        # Step 4: Ensemble
        if not args.skip_ensemble and model_results:
            submission = create_ensemble(config, model_results)

        logger.info("="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
