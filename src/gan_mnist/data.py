"""
Data loading and preprocessing for MNIST dataset.
"""

import logging
# Try to import tensorflow, but don't fail if it's not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

import numpy as np
from typing import Tuple, Optional
from .config import DataConfig
from .utils import setup_logging

logger = logging.getLogger(__name__)


def load_mnist(
    buffer_size: int = 60000,
    batch_size: int = 256,
    normalize_to_neg1_1: bool = True,
    subset_size: Optional[int] = None,
    seed: int = 42
) -> tf.data.Dataset:
    """
    Load and preprocess the MNIST dataset.
    
    Args:
        buffer_size: Buffer size for shuffling
        batch_size: Batch size for training
        normalize_to_neg1_1: Whether to normalize images to [-1, 1] range
        subset_size: If specified, use only this many samples for faster testing
        seed: Random seed for reproducibility
        
    Returns:
        Preprocessed tf.data.Dataset yielding (images, labels) tuples
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for load_mnist()")
    
    logger.info("Loading MNIST dataset...")
    
    # Load MNIST data
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    
    # Apply subset if specified
    if subset_size is not None:
        logger.info(f"Using subset of {subset_size} samples")
        indices = np.random.RandomState(seed).choice(
            len(train_images), size=subset_size, replace=False
        )
        train_images = train_images[indices]
        train_labels = train_labels[indices]
    
    # Preprocess images: reshape and normalize
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1
    ).astype('float32')
    
    if normalize_to_neg1_1:
        # Normalize to [-1, 1] range
        train_images = (train_images - 127.5) / 127.5
        logger.info("Normalized images to [-1, 1] range")
    else:
        # Normalize to [0, 1] range
        train_images = train_images / 255.0
        logger.info("Normalized images to [0, 1] range")
    
    # Preprocess labels: reshape for embedding layer
    train_labels = train_labels.reshape(-1, 1).astype('int32')
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    
    if subset_size is not None:
        # For subset, we might want to repeat for multiple epochs
        dataset = dataset.shuffle(buffer_size, seed=seed)
    else:
        dataset = dataset.shuffle(buffer_size, seed=seed)
        
    dataset = dataset.batch(batch_size)
    
    logger.info(f"Created dataset with {len(train_images)} samples")
    logger.info(f"Batch size: {batch_size}, Buffer size: {buffer_size}")
    
    return dataset


def get_mnist_info() -> dict:
    """
    Get basic information about the MNIST dataset.
    
    Returns:
        Dictionary containing dataset information
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for get_mnist_info()")
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    return {
        'train_samples': len(train_images),
        'test_samples': len(test_images),
        'image_shape': train_images.shape[1:],
        'num_classes': len(np.unique(train_labels)),
        'label_range': (int(np.min(train_labels)), int(np.max(train_labels)))
    }


def create_synthetic_dataset(
    batch_size: int = 32,
    image_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    num_samples: int = 1000,
    seed: int = 42
) -> tf.data.Dataset:
    """
    Create a synthetic dataset for testing when MNIST is not available.
    
    Args:
        batch_size: Batch size for the dataset
        image_shape: Shape of each image (height, width, channels)
        num_classes: Number of label classes
        num_samples: Total number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Synthetic tf.data.Dataset yielding (images, labels) tuples
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for create_synthetic_dataset()")
    
    logger.info(f"Creating synthetic dataset with {num_samples} samples")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate random images in [0, 1] range
    images = np.random.rand(num_samples, *image_shape).astype('float32')
    
    # Generate random labels
    labels = np.random.randint(0, num_classes, size=(num_samples, 1)).astype('int32')
    
    # Normalize images to [-1, 1] range to match MNIST preprocessing
    images = (images - 0.5) * 2.0  # [0, 1] -> [-1, 1]
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=num_samples, seed=seed)
    dataset = dataset.batch(batch_size)
    
    return dataset
