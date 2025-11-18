"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = OmegaConf.create(config_dict)
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object
        save_path: Path to save configuration
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)


def merge_configs(base_config: DictConfig,
                  override_config: Optional[Dict[str, Any]] = None) -> DictConfig:
    """
    Merge base configuration with overrides.

    Args:
        base_config: Base configuration
        override_config: Override dictionary

    Returns:
        Merged configuration
    """
    if override_config is None:
        return base_config

    override_omega = OmegaConf.create(override_config)
    merged = OmegaConf.merge(base_config, override_omega)
    return merged


def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """
    Get configuration value with dot notation.

    Args:
        config: Configuration object
        key: Key in dot notation (e.g., 'model.lightgbm.params.learning_rate')
        default: Default value if key not found

    Returns:
        Configuration value
    """
    try:
        return OmegaConf.select(config, key, default=default)
    except Exception:
        return default
