"""
Configuration management for the GAN-MNIST project.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    dataset: str = "mnist"
    buffer_size: int = 60000
    batch_size: int = 256
    image_shape: List[int] = field(default_factory=lambda: [28, 28, 1])
    num_classes: int = 10
    normalize_to_neg1_1: bool = True
    
    def __post_init__(self):
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if len(self.image_shape) != 3:
            raise ValueError("image_shape must have 3 dimensions [height, width, channels]")


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    latent_dim: int = 100
    label_embedding_dim: int = 50
    generator_base_filters: List[int] = field(default_factory=lambda: [256, 128, 64])
    discriminator_base_filters: List[int] = field(default_factory=lambda: [64, 128])
    generator_upsample_filters: List[int] = field(default_factory=lambda: [128, 64, 1])
    discriminator_dropout_rate: float = 0.3
    leaky_relu_alpha: float = 0.2
    batch_norm_momentum: float = 0.9
    
    def __post_init__(self):
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.label_embedding_dim <= 0:
            raise ValueError("label_embedding_dim must be positive")
        if not 0 <= self.discriminator_dropout_rate <= 1:
            raise ValueError("discriminator_dropout_rate must be between 0 and 1")
        if not 0 < self.leaky_relu_alpha:
            raise ValueError("leaky_relu_alpha must be positive")
        if not 0 < self.batch_norm_momentum < 1:
            raise ValueError("batch_norm_momentum must be between 0 and 1")


@dataclass
class TrainConfig:
    """Configuration for training parameters."""
    epochs: int = 300
    generator_learning_rate: float = 2e-4
    discriminator_learning_rate: float = 1e-5
    adam_beta_1: float = 0.5
    label_smoothing: float = 0.9
    seed: int = 42
    steps_per_epoch: Optional[int] = None
    
    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.generator_learning_rate <= 0:
            raise ValueError("generator_learning_rate must be positive")
        if self.discriminator_learning_rate <= 0:
            raise ValueError("discriminator_learning_rate must be positive")
        if not 0 <= self.adam_beta_1 <= 1:
            raise ValueError("adam_beta_1 must be between 0 and 1")
        if not 0 < self.label_smoothing <= 1:
            raise ValueError("label_smoothing must be between 0 and 1")


@dataclass
class PathsConfig:
    """Configuration for output paths."""
    checkpoint_dir: str = "outputs/checkpoints"
    samples_dir: str = "outputs/samples"
    figures_dir: str = "outputs/figures"
    logs_dir: str = "outputs/logs"
    reports_dir: str = "outputs/reports"


@dataclass
class Config:
    """Main configuration combining all sub-configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    outputs: PathsConfig = field(default_factory=PathsConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested structure
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        train_config = TrainConfig(**config_dict.get('training', {}))
        outputs_config = PathsConfig(**config_dict.get('outputs', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=train_config,
            outputs=outputs_config
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'outputs': self.outputs.__dict__
        }
        
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(path)
        if dir_name:  # Only create directory if path includes a directory
            os.makedirs(dir_name, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """Validate all configuration parameters."""
        # The __post_init__ methods in each dataclass handle validation
        self.data.__post_init__()
        self.model.__post_init__()
        self.training.__post_init__()
