"""
Fixtures and configuration for pytest.
"""

import pytest
import tempfile
import os
import sys

# Add the src directory to the path so we can import gan_mnist modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import tensorflow, but don't fail if it's not available
try:
    import tensorflow as tf
    import numpy as np
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    np = None

from gan_mnist.config import Config, DataConfig, ModelConfig, TrainConfig, PathsConfig


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "tensorflow: marks tests that require tensorflow")


def pytest_runtest_setup(item):
    """Skip tests that require tensorflow when it's not available."""
    if "tensorflow" in item.keywords and not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def config():
    """Create a basic configuration for testing."""
    return Config(
        data=DataConfig(
            buffer_size=100,
            batch_size=32,
            image_shape=[28, 28, 1],
            num_classes=10
        ),
        model=ModelConfig(
            latent_dim=10,
            label_embedding_dim=5,
            generator_base_filters=[16, 8],
            discriminator_base_filters=[8, 16],
            generator_upsample_filters=[8, 16, 1],
            discriminator_dropout_rate=0.3,
            leaky_relu_alpha=0.2,
            batch_norm_momentum=0.9
        ),
        training=TrainConfig(
            epochs=2,
            generator_learning_rate=2e-4,
            discriminator_learning_rate=1e-5,
            adam_beta_1=0.5,
            label_smoothing=0.9,
            seed=42,
            steps_per_epoch=5
        ),
        outputs=PathsConfig(
            checkpoint_dir="test_checkpoints",
            samples_dir="test_samples",
            figures_dir="test_figures",
            logs_dir="test_logs",
            reports_dir="test_reports"
        )
    )


@pytest.fixture
def small_config():
    """Create a small configuration for fast tests."""
    return Config(
        data=DataConfig(
            buffer_size=50,
            batch_size=16,
            image_shape=[28, 28, 1],
            num_classes=10
        ),
        model=ModelConfig(
            latent_dim=5,
            label_embedding_dim=3,
            generator_base_filters=[8, 4],
            discriminator_base_filters=[4, 8],
            generator_upsample_filters=[4, 8, 1],
            discriminator_dropout_rate=0.3,
            leaky_relu_alpha=0.2,
            batch_norm_momentum=0.9
        ),
        training=TrainConfig(
            epochs=1,
            generator_learning_rate=2e-4,
            discriminator_learning_rate=1e-5,
            adam_beta_1=0.5,
            label_smoothing=0.9,
            seed=42,
            steps_per_epoch=2
        ),
        outputs=PathsConfig(
            checkpoint_dir="test_checkpoints_small",
            samples_dir="test_samples_small",
            figures_dir="test_figures_small",
            logs_dir="test_logs_small",
            reports_dir="test_reports_small"
        )
    )


# Only define model fixtures if tensorflow is available
if TF_AVAILABLE:
    from gan_mnist.models.generator import build_generator
    from gan_mnist.models.discriminator import build_discriminator

    @pytest.fixture
    def generator_model(config):
        """Create a generator model for testing."""
        return build_generator(config.model)

    @pytest.fixture
    def discriminator_model(config):
        """Create a discriminator model for testing."""
        return build_discriminator(config.model)

    @pytest.fixture
    def sample_noise_labels(config):
        """Create sample noise and labels for testing."""
        batch_size = config.training.batch_size if hasattr(config.training, 'batch_size') else 16
        noise = tf.random.normal([batch_size, config.model.latent_dim])
        labels = tf.random.uniform(
            [batch_size, 1], 
            minval=0, 
            maxval=config.data.num_classes, 
            dtype=tf.int32
        )
        return noise, labels

    @pytest.fixture
    def sample_images_labels(config):
        """Create sample images and labels for testing."""
        batch_size = config.training.batch_size if hasattr(config.training, 'batch_size') else 16
        images = tf.random.normal([batch_size, 28, 28, 1])
        labels = tf.random.uniform(
            [batch_size, 1], 
            minval=0, 
            maxval=config.data.num_classes, 
            dtype=tf.int32
        )
        return images, labels
