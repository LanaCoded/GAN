"""
Loss functions for the GAN-MNIST project.
"""

import tensorflow as tf
from typing import Tuple
from ..utils import setup_logging

logger = logging.getLogger(__name__)


def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor, label_smoothing: float = 0.9) -> tf.Tensor:
    """
    Calculate the discriminator loss.
    
    Args:
        real_output: Discriminator output for real images (logits)
        fake_output: Discriminator output for fake images (logits)
        label_smoothing: Factor for label smoothing (0.9 in original implementation)
        
    Returns:
        Scalar loss value
    """
    # Label smoothing: real labels are smoothed to 0.9 instead of 1.0
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(real_output) * label_smoothing, real_output
    )
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(fake_output), fake_output
    )
    
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """
    Calculate the generator loss.
    
    Args:
        fake_output: Discriminator output for fake images (logits)
        
    Returns:
        Scalar loss value
    """
    # Generator wants discriminator to classify fakes as real (label = 1)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(fake_output), fake_output
    )
    return loss


def compute_gradient_penalty(
    discriminator: tf.keras.Model,
    real_samples: tf.Tensor,
    fake_samples: tf.Tensor,
    real_labels: tf.Tensor,
    fake_labels: tf.Tensor
) -> tf.Tensor:
    """
    Compute gradient penalty for improved training stability (WGAN-GP style).
    Note: This is not used in the original implementation but provided as an option.
    
    Args:
        discriminator: The discriminator model
        real_samples: Real image tensors
        fake_samples: Generated image tensors
        real_labels: Real image labels
        fake_labels: Fake image labels
        
    Returns:
        Gradient penalty loss
    """
    # Random weight term for interpolation
    alpha = tf.random.uniform(shape=[tf.shape(real_samples)[0], 1, 1, 1], minval=0., maxval=1.)
    
    # Interpolate between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        # Get discriminator output for interpolated samples
        pred = discriminator([interpolated, real_labels], training=True)
    
    # Calculate gradients
    grads = tape.gradient(pred, interpolated)
    
    # Calculate gradient norm
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    
    # Calculate gradient penalty
    gradient_penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)
    
    return gradient_penalty
