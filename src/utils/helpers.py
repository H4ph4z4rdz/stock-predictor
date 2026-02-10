"""Utility functions for the stock predictor."""

import os
import yaml
import torch


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
