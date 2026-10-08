"""
Tests for data loading and preprocessing.
"""

import pytest
import numpy as np
import os
import tempfile
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

# Import data functions only if tensorflow is available, otherwise define mocks
if TF_AVAILABLE:
    from gan_mnist.data import load_mnist, get_mnist_info, create_synthetic_dataset
else:
    # Define mock functions that will be skipped
    def load_mnist(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def get_mnist_info(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def create_synthetic_dataset(*args, **kwargs):
        raise ImportError("TensorFlow not available")


def test_get_mnist_info():
    """Test getting MNIST dataset info."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    info = get_mnist_info()
    assert info['train_samples'] == 60000
    assert info['test_samples'] == 10000
    assert info['image_shape'] == (28, 28, 1)
    assert info['num_classes'] == 10
    assert info['label_range'] == (0, 9)


@pytest.mark.tensorflow
def test_load_mnist_defaults():
    """Test loading MNIST with default parameters."""
    set_seed(42)
    dataset = load_mnist(
        buffer_size=1000,
        batch_size=32,
        normalize_to_neg1_1=True
    )
    
    # Check that we can iterate over the dataset
    batch_count = 0
    for images, labels in dataset.take(5):
        batch_count += 1
        # Check shapes
        assert images.shape == (32, 28, 28, 1)
        assert labels.shape == (32, 1)
        # Check dtype
        assert images.dtype == tf.float32
        assert labels.dtype == tf.int32
        # Check normalization range [-1, 1]
        assert tf.reduce_min(images) >= -1.0
        assert tf.reduce_max(images) <= 1.0
    
    assert batch_count == 5


@pytest.mark.tensorflow
def test_load_mnist_subset():
    """Test loading MNIST with subset."""
    set_seed(42)
    dataset = load_mnist(
        buffer_size=500,
        batch_size=16,
        normalize_to_neg1_1=True,
        subset_size=100
    )
    
    batch_count = 0
    total_samples = 0
    for images, labels in dataset:
        batch_count += 1
        total_samples += images.shape[0]
        # We should get exactly 100 samples (with some possibly in last partial batch)
        if total_samples >= 100:
            break
    
    # Should have processed around 100 samples
    assert 80 <= total_samples <= 112  # Allowing for batch size of 16


@pytest.mark.tensorflow
def test_load_mnist_normalization():
    """Test different normalization options."""
    set_seed(42)
    
    # Test [-1, 1] normalization
    dataset1 = load_mnist(
        buffer_size=100,
        batch_size=32,
        normalize_to_neg1_1=True,
        subset_size=100
    )
    
    for images, _ in dataset1.take(1):
        assert tf.reduce_min(images) >= -1.0
        assert tf.reduce_max(images) <= 1.0
        break
    
    # Test [0, 1] normalization
    dataset2 = load_mnist(
        buffer_size=100,
        batch_size=32,
        normalize_to_neg1_1=False,
        subset_size=100
    )
    
    for images, _ in dataset2.take(1):
        assert tf.reduce_min(images) >= 0.0
        assert tf.reduce_max(images) <= 1.0
        break


@pytest.mark.tensorflow
def test_load_mnist_shuffle_reproducibility():
    """Test that shuffling is reproducible with same seed."""
    set_seed(42)
    dataset1 = load_mnist(
        buffer_size=1000,
        batch_size=32,
        subset_size=200
    )
    
    set_seed(42)
    dataset2 = load_mnist(
        buffer_size=1000,
        batch_size=32,
        subset_size=200
    )
    
    # Get first few batches from each dataset
    batches1 = []
    batches2 = []
    
    for images, labels in dataset1.take(3):
        batches1.append((images.numpy(), labels.numpy()))
    
    for images, labels in dataset2.take(3):
        batches2.append((images.numpy(), labels.numpy()))
    
    # Should be identical
    assert len(batches1) == len(batches2) == 3
    for b1, b2 in zip(batches1, batches2):
        np.testing.assert_array_equal(b1[0], b2[0])  # images
        np.testing.assert_array_equal(b1[1], b2[1])  # labels


@pytest.mark.tensorflow
def test_create_synthetic_dataset():
    """Test creating synthetic dataset."""
    set_seed(42)
    dataset = create_synthetic_dataset(
        batch_size=32,
        image_shape=(28, 28, 1),
        num_classes=10,
        num_samples=1000,
        seed=42
    )
    
    # Check that we can iterate over the dataset
    batch_count = 0
    for images, labels in dataset.take(5):
        batch_count += 1
        # Check shapes
        assert images.shape == (32, 28, 28, 1)
        assert labels.shape == (32, 1)
        # Check dtype
        assert images.dtype == tf.float32
        assert labels.dtype == tf.int32
        # Check normalization range [-1, 1] (synthetic data is normalized to this)
        assert tf.reduce_min(images) >= -1.0
        # Note: synthetic data generation might not exactly match [-1,1], but the function should work
        assert tf.reduce_max(images) <= 1.0
    
    assert batch_count == 5


@pytest.mark.tensorflow
def test_create_synthetic_dataset_reproducibility():
    """Test that synthetic dataset is reproducible with same seed."""
    set_seed(42)
    dataset1 = create_synthetic_dataset(
        batch_size=32,
        image_shape=(28, 28, 1),
        num_classes=10,
        num_samples=200,
        seed=42
    )
    
    set_seed(42)
    dataset2 = create_synthetic_dataset(
        batch_size=32,
        image_shape=(28, 28, 1),
        num_classes=10,
        num_samples=200,
        seed=42
    )
    
    # Get first few batches from each dataset
    batches1 = []
    batches2 = []
    
    for images, labels in dataset1.take(3):
        batches1.append((images.numpy(), labels.numpy()))
    
    for images, labels in dataset2.take(3):
        batches2.append((images.numpy(), labels.numpy()))
    
    # Should be identical
    assert len(batches1) == len(batches2) == 3
    for b1, b2 in zip(batches1, batches2):
        np.testing.assert_array_equal(b1[0], b2[0])  # images
        np.testing.assert_array_equal(b1[1], b2[1])  # labels
