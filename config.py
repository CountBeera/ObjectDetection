# config.py

import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config.

    Returns:
        dict: Parsed config dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config
