"""Utility functions for training and inference."""

import torch
import json
from pathlib import Path
from typing import Dict, Any


def setup_device(use_cuda: bool = True) -> str:
    """
    Set up device for training (GPU or CPU).
    
    Args:
        use_cuda: Whether to use GPU if available.
    
    Returns:
        Device string ('cuda' or 'cpu').
    """
    if use_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")
    
    return device


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file.
    
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save config to.
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_data_folder_structure(base_path: str):
    """
    Create data folder structure.
    
    Args:
        base_path: Base path for data folders.
    """
    paths = [
        f"{base_path}/raw",
        f"{base_path}/processed",
    ]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
