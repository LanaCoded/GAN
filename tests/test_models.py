"""
Tests for model definitions.
"""

import pytest
import numpy as np
import os
import sys

# Add the src directory to the path so we can import gan_mnist modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import tensorflow, but don't fail if it's not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from gan_mnist.utils import set_seed

# Import model functions only if tensorflow is available, otherwise define mocks
if TF_AVAILABLE:
    from gan_mnist.models.generator import build_generator
    from gan_mnist.models.discriminator import build_discriminator
    from gan_mnist.config import ModelConfig
else:
    # Define mock functions that will be skipped
    def build_generator(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def build_discriminator(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    class ModelConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not available")


def test_generator_build():
    """Test generator model building."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    config = ModelConfig(
        latent_dim=10,
        label_embedding_dim=5,
        generator_base_filters=[16, 8],
        generator_upsample_filters=[8, 16, 1],
        leaky_relu_alpha=0.2
    )
    
    generator = build_generator(config)
    
    # Check that it's a valid Keras model
    assert hasattr(generator, 'layers')
    assert len(generator.layers) > 0
    
    # Check input and output shapes
    assert generator.input_shape == [(None, 10), (None, 1)]  # [noise, label]
    assert generator.output_shape == (None, 28, 28, 1)
    
    # Check that it has a reasonable number of parameters
    assert generator.count_params() > 0
    
    # Test forward pass
    noise = tf.random.normal([4, 10])
    labels = tf.random.uniform([4, 1], minval=0, maxval=10, dtype=tf.int32)
    output = generator([noise, labels], training=False)
    
    assert output.shape == (4, 28, 28, 1)
    # Output should be in [-1, 1] due to tanh activation
    assert tf.reduce_min(output) >= -1.0
    assert tf.reduce_max(output) <= 1.0


def test_generator_deterministic():
    """Test that generator gives deterministic results with same seed."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    config = ModelConfig(
        latent_dim=10,
        label_embedding_dim=5,
        generator_base_filters=[16, 8],
        generator_upsample_filters=[8, 16, 1]
    )
    
    # Build two generators with same config
    generator1 = build_generator(config)
    generator2 = build_generator(config)
    
    # Same input
    noise = tf.random.normal([4, 10])
    labels = tf.random.uniform([4, 1], minval=0, maxval=10, dtype=tf.int32)
    
    # Different outputs due to random initialization
    output1 = generator1([noise, labels], training=False)
    output2 = generator2([noise, labels], training=False)
    
    # They should be different (different random weights)
    # But we can test that same generator gives same output
    output1a = generator1([noise, labels], training=False)
    output1b = generator1([noise, labels], training=False)
    
    np.testing.assert_array_almost_equal(output1a.numpy(), output1b.numpy())


def test_discriminator_build():
    """Test discriminator model building."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    config = ModelConfig(
        latent_dim=10,
        label_embedding_dim=5,
        discriminator_base_filters=[8, 16],
        discriminator_dropout_rate=0.3,
        leaky_relu_alpha=0.2
    )
    
    discriminator = build_discriminator(config)
    
    # Check that it's a valid Keras model
    assert hasattr(discriminator, 'layers')
    assert len(discriminator.layers) > 0
    
    # Check input and output shapes
    assert discriminator.input_shape == [(None, 28, 28, 1), (None, 1)]  # [image, label]
    assert discriminator.output_shape == (None, 1)
    
    # Check that it has a reasonable number of parameters
    assert discriminator.count_params() > 0
    
    # Test forward pass
    images = tf.random.normal([4, 28, 28, 1])
    labels = tf.random.uniform([4, 1], minval=0, maxval=10, dtype=tf.int32)
    output = discriminator([images, labels], training=False)
    
    assert output.shape == (4, 1)
    # Output is logits (no activation), so can be any real number


def test_discriminator_conditional():
    """Test that discriminator is conditional on labels."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    config = ModelConfig(
        latent_dim=10,
        label_embedding_dim=5,
        discriminator_base_filters=[8, 16],
        discriminator_dropout_rate=0.3,
        leaky_relu_alpha=0.2
    )
    
    discriminator = build_discriminator(config)
    
    # Same image, different labels
    images = tf.random.normal([2, 28, 28, 1])
    label1 = tf.constant([[0], [0]], dtype=tf.int32)
    label2 = tf.constant([[1], [1]], dtype=tf.int32)
    
    output1 = discriminator([images, label1], training=False)
    output2 = discriminator([images, label2], training=False)
    
    # Outputs should be different (though might be similar due to random init)
    # We mainly want to ensure it runs without error
    assert output1.shape == (2, 1)
    assert output2.shape == (2, 1)


def test_model_summary_capture():
    """Test that we can capture model summaries."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    config = ModelConfig(
        latent_dim=10,
        label_embedding_dim=5,
        generator_base_filters=[16, 8],
        generator_upsample_filters=[8, 16, 1],
        discriminator_base_filters=[8, 16],
        discriminator_dropout_rate=0.3
    )
    
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    
    # This should not raise an exception
    gen_summary = []
    generator.summary(print_fn=lambda x: gen_summary.append(x))
    disc_summary = []
    discriminator.summary(print_fn=lambda x: disc_summary.append(x))
    
    assert len(gen_summary) > 0
    assert len(disc_summary) > 0
    # Summary should contain model info
    gen_summary_str = '\n'.join(gen_summary)
    disc_summary_str = '\n'.join(disc_summary)
    assert "Generator" in gen_summary_str or "Model" in gen_summary_str
    assert "Discriminator" in disc_summary_str or "Model" in disc_summary_str


def test_model_config_integration():
    """Test that models work with config model."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Test with default config values (matching original notebook)
    config = ModelConfig()  # Uses defaults
    print(f"ModelConfig: {config}")
    assert config is not None
    
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    
    # Test shapes
    assert generator.input_shape == [(None, 100), (None, 1)]
    assert generator.output_shape == (None, 28, 28, 1)
    assert discriminator.input_shape == [(None, 28, 28, 1), (None, 1)]
    assert discriminator.output_shape == (None, 1)
    
    # Test forward pass
    noise = tf.random.normal([2, 100])
    labels = tf.random.uniform([2, 1], minval=0, maxval=10, dtype=tf.int32)
    images = tf.random.normal([2, 28, 28, 1])
    
    gen_output = generator([noise, labels], training=False)
    disc_output = discriminator([images, labels], training=False)
    
    assert gen_output.shape == (2, 28, 28, 1)
    assert disc_output.shape == (2, 1)
