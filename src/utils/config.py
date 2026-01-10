"""Configuration loading and management utilities."""
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result


def get_config(
    default_path: str = "configs/default.yaml",
    env_config: Optional[str] = None,
    agent_config: Optional[str] = None,
    **overrides
) -> Dict[str, Any]:
    """
    Load and merge configurations from multiple sources.
    
    Args:
        default_path: Path to default configuration
        env_config: Path to environment configuration
        agent_config: Path to agent configuration
        **overrides: Direct override values
        
    Returns:
        Merged configuration dictionary
    """
    config = load_config(default_path)
    
    if env_config:
        env_cfg = load_config(env_config)
        config = merge_configs(config, {"environment": env_cfg})
        
    if agent_config:
        agent_cfg = load_config(agent_config)
        config = merge_configs(config, {"agent": agent_cfg})
        
    if overrides:
        config = merge_configs(config, overrides)
        
    return config