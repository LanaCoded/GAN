"""
Tests for configuration module.
"""

import pytest
import yaml
import tempfile
import os
from gan_mnist.config import Config, DataConfig, ModelConfig, TrainConfig, PathsConfig


def test_data_config_defaults():
    """Test DataConfig default values."""
    config = DataConfig()
    assert config.dataset == "mnist"
    assert config.buffer_size == 60000
    assert config.batch_size == 256
    assert config.image_shape == [28, 28, 1]
    assert config.num_classes == 10
    assert config.normalize_to_neg1_1 == True


def test_data_config_validation():
    """Test DataConfig validation."""
    # Valid config
    config = DataConfig(buffer_size=1000, batch_size=32, num_classes=5)
    assert config.buffer_size == 1000
    assert config.batch_size == 32
    assert config.num_classes == 5
    
    # Invalid buffer_size
    with pytest.raises(ValueError):
        DataConfig(buffer_size=0)
    
    # Invalid batch_size
    with pytest.raises(ValueError):
        DataConfig(batch_size=-1)
    
    # Invalid num_classes
    with pytest.raises(ValueError):
        DataConfig(num_classes=0)


def test_model_config_defaults():
    """Test ModelConfig default values."""
    config = ModelConfig()
    assert config.latent_dim == 100
    assert config.label_embedding_dim == 50
    assert config.generator_base_filters == [256, 128, 64]
    assert config.discriminator_base_filters == [64, 128]
    assert config.generator_upsample_filters == [128, 64, 1]
    assert config.discriminator_dropout_rate == 0.3
    assert config.leaky_relu_alpha == 0.2
    assert config.batch_norm_momentum == 0.9


def test_model_config_validation():
    """Test ModelConfig validation."""
    # Valid config
    config = ModelConfig(latent_dim=50, label_embedding_dim=25)
    assert config.latent_dim == 50
    assert config.label_embedding_dim == 25
    
    # Invalid latent_dim
    with pytest.raises(ValueError):
        ModelConfig(latent_dim=0)
    
    # Invalid label_embedding_dim
    with pytest.raises(ValueError):
        ModelConfig(label_embedding_dim=-1)
    
    # Invalid dropout rate
    with pytest.raises(ValueError):
        ModelConfig(discriminator_dropout_rate=1.5)
    
    # Invalid leaky_relu_alpha
    with pytest.raises(ValueError):
        ModelConfig(leaky_relu_alpha=0)
    
    # Invalid batch_norm_momentum
    with pytest.raises(ValueError):
        ModelConfig(batch_norm_momentum=1.5)


def test_train_config_defaults():
    """Test TrainConfig default values."""
    config = TrainConfig()
    assert config.epochs == 300
    assert config.generator_learning_rate == 2e-4
    assert config.discriminator_learning_rate == 1e-5
    assert config.adam_beta_1 == 0.5
    assert config.label_smoothing == 0.9
    assert config.seed == 42
    assert config.steps_per_epoch is None


def test_train_config_validation():
    """Test TrainConfig validation."""
    # Valid config
    config = TrainConfig(epochs=100, generator_learning_rate=1e-3)
    assert config.epochs == 100
    assert config.generator_learning_rate == 1e-3
    
    # Invalid epochs
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    
    # Invalid learning rates
    with pytest.raises(ValueError):
        TrainConfig(generator_learning_rate=0)
    
    with pytest.raises(ValueError):
        TrainConfig(discriminator_learning_rate=-1e-5)
    
    # Invalid adam_beta_1
    with pytest.raises(ValueError):
        TrainConfig(adam_beta_1=1.5)
    
    # Invalid label_smoothing
    with pytest.raises(ValueError):
        TrainConfig(label_smoothing=0)
    
    with pytest.raises(ValueError):
        TrainConfig(label_smoothing=1.5)


def test_paths_config_defaults():
    """Test PathsConfig default values."""
    config = PathsConfig()
    assert config.checkpoint_dir == "outputs/checkpoints"
    assert config.samples_dir == "outputs/samples"
    assert config.figures_dir == "outputs/figures"
    assert config.logs_dir == "outputs/logs"
    assert config.reports_dir == "outputs/reports"


def test_config_from_yaml():
    """Test loading configuration from YAML."""
    config_dict = {
        'data': {
            'buffer_size': 1000,
            'batch_size': 64,
            'num_classes': 10
        },
        'model': {
            'latent_dim': 50,
            'label_embedding_dim': 25
        },
        'training': {
            'epochs': 50,
            'generator_learning_rate': 1e-3,
            'discriminator_learning_rate': 5e-6
        },
        'outputs': {
            'checkpoint_dir': 'test/checkpoints',
            'samples_dir': 'test/samples'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        config = Config.from_yaml(config_path)
        assert config.data.buffer_size == 1000
        assert config.data.batch_size == 64
        assert config.model.latent_dim == 50
        assert config.model.label_embedding_dim == 25
        assert config.training.epochs == 50
        assert config.training.generator_learning_rate == 1e-3
        assert config.training.discriminator_learning_rate == 5e-6
        assert config.outputs.checkpoint_dir == 'test/checkpoints'
        assert config.outputs.samples_dir == 'test/samples'
    finally:
        os.unlink(config_path)


def test_config_to_yaml():
    """Test saving configuration to YAML."""
    config = Config(
        data=DataConfig(buffer_size=500),
        model=ModelConfig(latent_dim=50),
        training=TrainConfig(epochs=25),
        outputs=PathsConfig(checkpoint_dir='test/checkpoints')
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_path = f.name
    
    try:
        config.to_yaml(config_path)
        
        # Load it back and verify
        loaded_config = Config.from_yaml(config_path)
        assert loaded_config.data.buffer_size == 500
        assert loaded_config.model.latent_dim == 50
        assert loaded_config.training.epochs == 25
        assert loaded_config.outputs.checkpoint_dir == 'test/checkpoints'
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_config_validation():
    """Test Config validation method."""
    config = Config()
    # This should not raise an exception
    config.validate()
    
    # Test with invalid values
    config.data.buffer_size = -1
    with pytest.raises(ValueError):
        config.validate()
