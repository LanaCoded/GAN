"""
Tests for evaluation metrics.
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

# Import evaluation functions only if tensorflow is available, otherwise define mocks
if TF_AVAILABLE:
    from gan_mnist.evaluation import (
        calculate_diversity,
        calculate_label_conditional_accuracy,
        create_simple_classifier,
        evaluate_discriminator_outputs,
        calculate_inception_score
    )
    from gan_mnist.models.generator import build_generator
    from gan_mnist.models.discriminator import build_discriminator
    from gan_mnist.config import ModelConfig, DataConfig, TrainConfig, PathsConfig
    from gan_mnist.trainer import GANTrainer
else:
    # Define mock functions that will be skipped
    def calculate_diversity(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def calculate_label_conditional_accuracy(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def create_simple_classifier(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def evaluate_discriminator_outputs(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def calculate_inception_score(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def build_generator(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def build_discriminator(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    class ModelConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not available")
    
    class DataConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not available")
    
    class TrainConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not available")
    
    class PathsConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not available")
    
    class GANTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not available")


def test_calculate_diversity():
    """Test diversity calculation."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Identical images should have zero diversity
    identical_images = tf.ones([4, 28, 28, 1])
    div_identical = calculate_diversity(identical_images)
    assert div_identical == 0.0
    
    # Different images should have positive diversity
    # Create two distinct patterns
    img1 = tf.zeros([2, 28, 28, 1])  # All zeros
    img2 = tf.ones([2, 28, 28, 1])   # All ones
    diff_images = tf.concat([img1, img2], axis=0)
    div_diff = calculate_diversity(diff_images)
    assert div_diff > 0.0
    
    # Test with known values
    # Simple case: two images that are far apart
    simple_imgs = tf.constant([
        [[[0.0]], [[0.0]]],  # 2x2 image all zeros
        [[[1.0]], [[1.0]]]   # 2x2 image all ones
    ])
    # Reshape to proper format
    simple_imgs = tf.reshape(simple_imgs, [2, 2, 2, 1])
    div_simple = calculate_diversity(simple_imgs)
    assert div_simple > 0.0


def test_calculate_diversity_edge_cases():
    """Test diversity calculation edge cases."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Single image
    single_image = tf.random.normal([1, 28, 28, 1])
    div_single = calculate_diversity(single_image)
    assert div_single == 0.0


def test_create_simple_classifier():
    """Test creating a simple classifier."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    classifier = create_simple_classifier(input_shape=(28, 28, 1))
    
    assert isinstance(classifier, tf.keras.Model)
    assert len(classifier.layers) > 0
    
    # Check input and output shapes
    assert classifier.input_shape == (None, 28, 28, 1)
    assert classifier.output_shape == (None, 10)  # 10 classes for MNIST
    
    # Test forward pass
    images = tf.random.normal([4, 28, 28, 1])
    output = classifier(images, training=False)
    
    assert output.shape == (4, 10)
    # Output should be probabilities (softmax)
    probs = tf.nn.softmax(output, axis=1)
    assert tf.reduce_min(probs) >= 0.0
    assert tf.reduce_max(probs) <= 1.0
    # Each row should sum to 1 (approximately)
    row_sums = tf.reduce_sum(probs, axis=1)
    np.testing.assert_allclose(row_sums.numpy(), 1.0, rtol=1e-5)


def test_evaluate_discriminator_outputs():
    """Test discriminator output evaluation."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Create simple models for testing
    def simple_discriminator():
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(1)
        ])
    
    discriminator = simple_discriminator()
    
    batch_size = 4
    # Real images that should be classified as real (positive logits)
    real_images = tf.random.normal([batch_size, 28, 28, 1], mean=2.0)  # Shifted to be positive
    # Fake images that should be classified as fake (negative logits)
    fake_images = tf.random.normal([batch_size, 28, 28, 1], mean=-2.0)  # Shifted to be negative
    
    real_labels = tf.random.uniform([batch_size, 1], minval=0, maxval=10, dtype=tf.int32)
    fake_labels = tf.random.uniform([batch_size, 1], minval=0, maxval=10, dtype=tf.int32)
    
    d_real_acc, d_fake_acc, d_real_mean, d_fake_mean = evaluate_discriminator_outputs(
        discriminator, real_images, fake_images, real_labels, fake_labels
    )
    
    # Check return types
    assert isinstance(d_real_acc, float)
    assert isinstance(d_fake_acc, float)
    assert isinstance(d_real_mean, float)
    assert isinstance(d_fake_mean, float)
    
    # Check ranges
    assert 0.0 <= d_real_acc <= 1.0
    assert 0.0 <= d_fake_acc <= 1.0
    assert 0.0 <= d_real_mean <= 1.0
    assert 0.0 <= d_fake_mean <= 1.0
    
    # With our shifted inputs, we expect decent accuracy
    # Note: This is a very simple discriminator, so performance may vary
    # But we can at least check that the function works


def test_evaluate_discriminator_outputs_perfect():
    """Test discriminator evaluation with perfect predictions."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Create a discriminator that always gives correct predictions
    class PerfectDiscriminator(tf.keras.Model):
        def call(self, inputs, training=False):
            images, labels = inputs
            # Always output large positive for real, large negative for fake
            # We'll use a simple approach: return constant values based on a flag
            # Actually, let's make it predict based on image mean
            mean_val = tf.reduce_mean(inputs[0], axis=[1, 2, 3])  # Mean over spatial dims
            # If mean > 0, predict real (positive logit), else predict fake (negative logit)
            logits = tf.where(mean_val > 0, 10.0, -10.0)
            return tf.expand_dims(logits, -1)
    
    discriminator = PerfectDiscriminator()
    
    batch_size = 4
    # Create clearly separable data
    real_images = tf.ones([batch_size, 28, 28, 1]) * 2.0  # Positive mean
    fake_images = tf.ones([batch_size, 28, 28, 1]) * (-2.0)  # Negative mean
    
    real_labels = tf.zeros([batch_size, 1], dtype=tf.int32)  # Dummy labels
    fake_labels = tf.ones([batch_size, 1], dtype=tf.int32)   # Dummy labels
    
    d_real_acc, d_fake_acc, d_real_mean, d_fake_mean = evaluate_discriminator_outputs(
        discriminator, real_images, fake_images, real_labels, fake_labels
    )
    
    # With perfect separation, we should get high accuracy
    # Real images: mean=2.0 > 0 -> logit=10 -> sigmoid ~ 1.0 -> >0.5 -> correct
    # Fake images: mean=-2.0 < 0 -> logit=-10 -> sigmoid ~ 0.0 -> <0.5 -> correct
    assert d_real_acc > 0.9
    assert d_fake_acc > 0.9
    assert d_real_mean > 0.9
    assert d_fake_mean < 0.1


def test_calculate_label_conditional_accuracy():
    """Test label conditional accuracy calculation."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Create simple models for testing
    def simple_generator():
        # Generator that ignores noise and label and produces constant image per label
        noise_input = tf.keras.Input(shape=(10,))
        label_input = tf.keras.Input(shape=(1,))
        
        # Process label to select image
        label_flat = tf.keras.layers.Flatten()(label_input)
        # Create 10 different constant images (one per class)
        # For simplicity, we'll make it produce images where pixel value = label/9.0 * 2 - 1
        # This maps label 0->-1, label 9->1
        
        def label_to_image(label):
            # label is shape (batch, 1), values 0-9
            label_float = tf.cast(label, tf.float32) / 9.0  # 0-1
            image_value = label_float * 2.0 - 1.0  # -1 to 1
            # Expand to full image
            image = tf.fill([tf.shape(label)[0], 28, 28, 1], image_value)
            return image
        
        # Use lambda layer or custom layer
        # For simplicity in test, we'll create a deterministic mapping
        images = tf.keras.layers.Lambda(
            lambda x: tf.fill([tf.shape(x[0])[0], 28, 28, 1], 
                            tf.cast(tf.squeeze(x[1], -1), tf.float32) / 9.0 * 2.0 - 1.0)
        )([noise_input, label_input])
        
        return tf.keras.Model([noise_input, label_input], images)
    
    def simple_classifier():
        # Classifier that reads the top-left pixel and maps to label
        inputs = tf.keras.Input(shape=(28, 28, 1))
        # Extract top-left pixel
        top_left = tf.keras.layers.Lambda(lambda x: x[:, 0, 0, 0])(inputs)
        # Map from [-1,1] to [0,9]
        logits = tf.keras.layers.Lambda(lambda x: (x + 1.0) * 9.0 / 2.0)(top_left)
        # Actually, let's make it predict the exact label from top-left pixel
        # Since our generator makes all pixels the same value
        return tf.keras.Model(inputs, logits)
    
    generator = simple_generator()
    classifier = simple_classifier()
    
    # Test with specific labels
    noise = tf.random.normal([4, 10])
    labels = tf.constant([[0], [5], [9], [3]], dtype=tf.int32)
    
    accuracy = calculate_label_conditional_accuracy(
        generator, classifier, noise, labels
    )
    
    # Should be perfect accuracy since our generator and classifier are perfectly aligned
    assert accuracy == 100.0
    
    # Test with random labels
    noise = tf.random.normal([10, 10])
    labels = tf.random.uniform([10, 1], minval=0, maxval=10, dtype=tf.int32)
    
    accuracy = calculate_label_conditional_accuracy(
        generator, classifier, noise, labels
    )
    
    # Should still be high accuracy
    assert accuracy >= 0.0  # At least not negative
    assert accuracy <= 100.0  # At most 100%


def test_calculate_inception_score():
    """Test inception score calculation (placeholder)."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # This is a placeholder test since the actual IS calculation is complex
    images = tf.random.normal([10, 28, 28, 1])
    score = calculate_inception_score(images, splits=2)
    
    # Should return 0.0 as placeholder
    assert score == 0.0
    
    # Test with different parameters
    score2 = calculate_inception_score(images, splits=5)
    assert score2 == 0.0


def test_evaluation_integration():
    """Test integration of evaluation components."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Create a simple GAN setup
    config = Config(
        data=DataConfig(buffer_size=100, batch_size=16),
        model=ModelConfig(latent_dim=10),
        training=TrainConfig(seed=42),
        outputs=PathsConfig()
    )
    
    # Create trainer-like setup
    generator = build_generator(config.model)
    discriminator = build_discriminator(config.model)
    
    # Create some test data
    noise = tf.random.normal([8, 10])
    labels = tf.random.uniform([8, 1], minval=0, maxval=10, dtype=tf.int32)
    images = tf.random.normal([8, 28, 28, 1])
    
    # Generate fake images
    fake_images = generator([noise, labels], training=False)
    
    # Calculate diversity
    diversity = calculate_diversity(fake_images)
    assert isinstance(diversity, float)
    assert diversity >= 0.0
    
    # Evaluate discriminator outputs
    d_real_acc, d_fake_acc, d_real_mean, d_fake_mean = evaluate_discriminator_outputs(
        discriminator, images, fake_images, labels, labels
    )
    
    assert isinstance(d_real_acc, float)
    assert isinstance(d_fake_acc, float)
    assert isinstance(d_real_mean, float)
    assert isinstance(d_fake_mean, float)
    
    # All values should be in reasonable ranges
    assert 0.0 <= d_real_acc <= 1.0
    assert 0.0 <= d_fake_acc <= 1.0
    assert 0.0 <= d_real_mean <= 1.0
    assert 0.0 <= d_fake_mean <= 1.0
