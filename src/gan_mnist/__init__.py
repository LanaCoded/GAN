"""
GAN-MNIST: Conditional GAN for MNIST digit generation.
"""

from .config import Config, DataConfig, ModelConfig, TrainConfig, PathsConfig

# Version information
__version__ = "0.1.0"
__author__ = "Contributor"
__email__ = "contributor@example.com"

# Public API - core components that don't require tensorflow import at module level
__all__ = [
    "Config",
    "DataConfig", 
    "ModelConfig",
    "TrainConfig",
    "PathsConfig"
]

# TensorFlow-dependent components are available but imported lazily
def __getattr__(name):
    """Lazily import tensorflow-dependent components."""
    if name == "GANTrainer":
        try:
            from .trainer import GANTrainer
            return GANTrainer
        except ImportError:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}' (TensorFlow not available)")
    elif name == "create_trainer_from_config":
        try:
            from .trainer import create_trainer_from_config
            return create_trainer_from_config
        except ImportError:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}' (TensorFlow not available)")
    elif name == "main":
        try:
            from .cli import main
            return main
        except ImportError:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}' (TensorFlow not available)")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
