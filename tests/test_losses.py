"""
Tests for loss functions.
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

# Import loss functions only if tensorflow is available, otherwise define mocks
if TF_AVAILABLE:
    from gan_mnist.losses import (
        discriminator_loss, 
        generator_loss,
        compute_gradient_penalty
    )
else:
    # Define mock functions that will be skipped
    def discriminator_loss(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def generator_loss(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def compute_gradient_penalty(*args, **kwargs):
        raise ImportError("TensorFlow not available")


def test_generator_loss():
    """Test generator loss function."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Perfect fake images (discriminator thinks they're real)
    # Output should be large positive numbers (since from_logits=True)
    fake_output_perfect = tf.constant([[10.0], [10.0]])  # Strong "real" prediction
    loss_perfect = generator_loss(fake_output_perfect)
    # Loss should be small when discriminator is fooled
    assert loss_perfect < 0.001  # Very small loss
    
    # Poor fake images (discriminator thinks they're fake)
    fake_output_poor = tf.constant([[-10.0], [-10.0]])  # Strong "fake" prediction
    loss_poor = generator_loss(fake_output_poor)
    # Loss should be large when discriminator is not fooled
    assert loss_poor > 0.9  # Large loss (close to log(2) ~ 0.693, but higher due to extreme values)
    
    # Random outputs
    fake_output_random = tf.random.uniform([4, 1], minval=-2, maxval=2)
    loss_random = generator_loss(fake_output_random)
    assert 0.1 < loss_random < 2.0  # Reasonable range


def test_discriminator_loss():
    """Test discriminator loss function."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
        
    label_smoothing = 0.9
    
    # Perfect discriminator (real=real, fake=fake)
    real_output_perfect = tf.constant([[10.0], [10.0]])  # Strong "real" for real images
    fake_output_perfect = tf.constant([[-10.0], [-10.0]])  # Strong "fake" for fake images
    loss_perfect = discriminator_loss(real_output_perfect, fake_output_perfect, label_smoothing)
    # Loss should be very small
    assert loss_perfect < 0.001
    
    # Worst discriminator (real=fake, fake=real)
    real_output_worst = tf.constant([[-10.0], [-10.0]])  # Strong "fake" for real images
    fake_output_worst = tf.constant([[10.0], [10.0]])   # Strong "real" for fake images
    loss_worst = discriminator_loss(real_output_worst, fake_output_worst, label_smoothing)
    # Loss should be large
    assert loss_worst > 1.0
    
    # Test label smoothing effect
    real_output = tf.constant([[0.0], [0.0]])  # Neutral prediction
    fake_output = tf.constant([[0.0], [0.0]])  # Neutral prediction
    
    # With label smoothing, real labels are 0.9 instead of 1.0
    loss_smoothed = discriminator_loss(real_output, fake_output, label_smoothing=0.9)
    loss_no_smoothing = discriminator_loss(real_output, fake_output, label_smoothing=1.0)
    
    # With smoothing, target is closer to prediction, so loss should be smaller
    assert loss_smoothed < loss_no_smoothing
    
    # Test edge cases
    # Perfect predictions with smoothing
    real_output_perfect = tf.constant([[20.0], [20.0]])  # Very strong "real"
    fake_output_perfect = tf.constant([[-20.0], [-20.0]]) # Very strong "fake"
    loss_perfect = discriminator_loss(real_output_perfect, fake_output_perfect, label_smoothing)
    assert loss_perfect < 0.000001  # Extremely small


def test_loss_functions_shape_handling():
    """Test that loss functions handle various tensor shapes."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
        
    batch_sizes = [1, 2, 4, 8, 16]
    
    for batch_size in batch_sizes:
        real_output = tf.random.normal([batch_size, 1])
        fake_output = tf.random.normal([batch_size, 1])
        
        # These should not raise exceptions
        d_loss = discriminator_loss(real_output, fake_output)
        g_loss = generator_loss(fake_output)
        
        assert d_loss.shape == ()
        assert g_loss.shape == ()
        assert tf.reduce_min(d_loss) >= 0
        assert tf.reduce_min(g_loss) >= 0


def test_loss_functions_dtype():
    """Test that loss functions work with different dtypes."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
        
    for dtype in [tf.float32, tf.float64]:
        real_output = tf.constant([[1.0], [-1.0]], dtype=dtype)
        fake_output = tf.constant([[-1.0], [1.0]], dtype=dtype)
        
        d_loss = discriminator_loss(real_output, fake_output)
        g_loss = generator_loss(fake_output)
        
        assert d_loss.dtype == dtype
        assert g_loss.dtype == dtype


def test_compute_gradient_penalty():
    """Test gradient penalty computation."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
        
    # This is mainly testing that the function runs without error
    # A full test would require known values
    
    # Create simple models for testing
    def simple_discriminator():
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(1)
        ])
    
    discriminator = simple_discriminator()
    
    batch_size = 4
    real_samples = tf.random.normal([batch_size, 28, 28, 1])
    fake_samples = tf.random.normal([batch_size, 28, 28, 1])
    real_labels = tf.random.uniform([batch_size, 1], minval=0, maxval=10, dtype=tf.int32)
    fake_labels = tf.random.uniform([batch_size, 1], minval=0, maxval=10, dtype=tf.int32)
    
    # This should not raise an exception
    gp = compute_gradient_penalty(discriminator, real_samples, fake_samples, real_labels, fake_labels)
    
    assert gp.shape == ()
    assert tf.reduce_min(gp) >= 0  # Gradient penalty should be non-negative
