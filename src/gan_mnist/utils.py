"""
Utility functions for the GAN-MNIST project.
"""

import logging
import os
import random
import sys
from typing import Tuple

# Try to import tensorflow, but don't fail if it's not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device_info() -> str:
    """Get information about available computing devices."""
    if not TF_AVAILABLE:
        return "CPU: TensorFlow not available"
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_details = []
        for gpu in gpus:
            try:
                details = tf.config.experimental.get_device_details(gpu)
                gpu_details.append(f"{gpu.name}: {details}")
            except Exception:
                gpu_details.append(f"{gpu.name}: details unavailable")
        return f"GPUs: {', '.join(gpu_details)}"
    else:
        return "CPU: No GPUs detected"


def create_directories(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def format_tensor_shape(shape: Tuple) -> str:
    """Format tensor shape as a string for logging/display."""
    return f"[{', '.join(map(str, shape))}]"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning a default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator
