"""
Evaluation metrics for GAN models.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, List, Optional
from sklearn.metrics import confusion_matrix, accuracy_score
from ..utils import setup_logging

logger = logging.getLogger(__name__)


def calculate_diversity(images: tf.Tensor) -> float:
    """
    Calculate diversity of generated images as average pairwise L2 distance.
    
    Args:
        images: Generated images tensor [batch_size, height, width, channels]
        
    Returns:
        Average pairwise L2 distance
    """
    # Flatten images to vectors
    flat_images = tf.reshape(images, [tf.shape(images)[0], -1])
    
    # Calculate pairwise distances
    # Using a more efficient approach for large batches
    batch_size = tf.shape(flat_images)[0]
    
    if batch_size <= 1:
        return 0.0
    
    # For small batches, compute all pairs
    if batch_size <= 100:
        distances = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                dist = tf.norm(flat_images[i] - flat_images[j])
                distances.append(dist)
        
        if distances:
            return tf.reduce_mean(distances).numpy()
        else:
            return 0.0
    else:
        # For larger batches, use a subset to avoid O(n^2) complexity
        subset_size = min(100, batch_size)
        indices = tf.random.shuffle(tf.range(batch_size))[:subset_size]
        subset = tf.gather(flat_images, indices)
        
        distances = []
        for i in range(subset_size):
            for j in range(i + 1, subset_size):
                dist = tf.norm(subset[i] - subset[j])
                distances.append(dist)
        
        return tf.reduce_mean(distances).numpy()


def calculate_label_conditional_accuracy(
    generator: tf.keras.Model,
    classifier: tf.keras.Model,
    noise: tf.Tensor,
    labels: tf.Tensor
) -> float:
    """
    Calculate how well the generator produces images matching the requested labels.
    
    Args:
        generator: Generator model
        classifier: Classifier model trained on real MNIST
        noise: Noise vectors for generation
        labels: Target labels for generation
        
    Returns:
        Accuracy percentage (0-100)
    """
    # Generate images
    generated_images = generator([noise, labels], training=False)
    
    # Get classifier predictions
    predictions = classifier.predict(generated_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    true_labels = tf.squeeze(labels).numpy()
    accuracy = accuracy_score(true_labels, predicted_labels) * 100
    
    return float(accuracy)


def create_simple_classifier(input_shape: Tuple[int, int, int] = (28, 28, 1)) -> tf.keras.Model:
    """
    Create a simple CNN classifier for evaluating label conditional accuracy.
    
    Args:
        input_shape: Shape of input images
        
    Returns:
        Compiled Keras model for classification
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def evaluate_discriminator_outputs(
    discriminator: tf.keras.Model,
    real_images: tf.Tensor,
    fake_images: tf.Tensor,
    real_labels: tf.Tensor,
    fake_labels: tf.Tensor
) -> Tuple[float, float, float, float]:
    """
    Evaluate discriminator outputs for real and fake images.
    
    Args:
        discriminator: Discriminator model
        real_images: Real image tensors
        fake_images: Fake image tensors
        real_labels: Real image labels
        fake_labels: Fake image labels
        
    Returns:
        Tuple of (d_real_acc, d_fake_acc, d_real_mean, d_fake_mean)
    """
    # Get discriminator outputs (logits)
    real_logits = discriminator([real_images, real_labels], training=False)
    fake_logits = discriminator([fake_images, fake_labels], training=False)
    
    # Convert to probabilities
    real_probs = tf.nn.sigmoid(real_logits)
    fake_probs = tf.nn.sigmoid(fake_logits)
    
    # Calculate accuracy (real should be > 0.5, fake should be < 0.5)
    real_pred = tf.cast(real_probs > 0.5, tf.float32)
    fake_pred = tf.cast(fake_probs < 0.5, tf.float32)
    
    d_real_acc = tf.reduce_mean(real_pred).numpy()
    d_fake_acc = tf.reduce_mean(fake_pred).numpy()
    
    # Calculate mean outputs
    d_real_mean = tf.reduce_mean(real_probs).numpy()
    d_fake_mean = tf.reduce_mean(fake_probs).numpy()
    
    return float(d_real_acc), float(d_fake_acc), float(d_real_mean), float(d_fake_mean)


def calculate_inception_score(images: tf.Tensor, splits: int = 10) -> float:
    """
    Calculate Inception Score for evaluating image quality and diversity.
    Note: This is a simplified version - a full implementation would use 
    an Inception v3 model pretrained on ImageNet.
    
    Args:
        images: Generated images tensor [batch_size, height, width, channels]
        splits: Number of splits for calculating IS
        
    Returns:
        Inception score (higher is better)
    """
    # This is a placeholder implementation
    # A real implementation would:
    # 1. Resize images to 299x299x3
    # 2. Pass through Inception v3 to get class probabilities
    # 3. Calculate KL divergence between p(y|x) and p(y)
    # 4. Return exp(mean(KL divergence))
    
    logger.warning("Inception Score calculation is not fully implemented")
    return 0.0
