"""
Callbacks for training monitoring and visualization.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from datetime import datetime
from ..utils import setup_logging

logger = logging.getLogger(__name__)


class GANCallback(tf.keras.callbacks.Callback):
    """
    Base callback class for GAN training.
    """
    
    def __init__(self, save_dir: str):
        """
        Initialize the callback.
        
        Args:
            save_dir: Directory to save callback outputs
        """
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)


class SampleGeneratorCallback(GANCallback):
    """
    Callback to generate and save sample images during training.
    """
    
    def __init__(
        self,
        generator: tf.keras.Model,
        fixed_noise: tf.Tensor,
        fixed_labels: tf.Tensor,
        save_dir: str,
        num_samples: int = 16
    ):
        """
        Initialize the sample generator callback.
        
        Args:
            generator: Generator model
            fixed_noise: Fixed noise vectors for consistent generation
            fixed_labels: Fixed labels for consistent generation
            save_dir: Directory to save samples
            num_samples: Number of samples to generate
        """
        super().__init__(save_dir)
        self.generator = generator
        self.fixed_noise = fixed_noise[:num_samples]
        self.fixed_labels = fixed_labels[:num_samples]
        self.num_samples = num_samples
    
    def on_epoch_end(self, epoch: int, logs: Dict = None) -> None:
        """
        Generate and save sample images at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            logs: Training logs (unused)
        """
        # Generate images
        generated_images = self.generator(
            [self.fixed_noise, self.fixed_labels], 
            training=False
        )
        
        # Rescale from [-1, 1] to [0, 1] for visualization
        generated_images = (generated_images + 1.0) / 2.0
        
        # Create sample grid
        self._save_sample_grid(generated_images, epoch)
    
    def _save_sample_grid(self, images: tf.Tensor, epoch: int) -> None:
        """
        Save images as a grid.
        
        Args:
            images: Generated images tensor [batch_size, 28, 28, 1]
            epoch: Current epoch number
        """
        # Determine grid size
        n = int(np.ceil(np.sqrt(self.num_samples)))
        
        plt.figure(figsize=(n, n))
        for i in range(min(self.num_samples, images.shape[0])):
            plt.subplot(n, n, i + 1)
            plt.imshow(images[i, :, :, 0], cmap='gray')
            plt.axis('off')
        
        plt.suptitle(f'Epoch {epoch + 1}')
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(
            self.save_dir, 
            f'sample_epoch_{epoch+1:04d}.png'
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved sample grid to {save_path}")


class LossHistoryCallback(GANCallback):
    """
    Callback to record and visualize loss history.
    """
    
    def __init__(self, save_dir: str):
        """
        Initialize the loss history callback.
        
        Args:
            save_dir: Directory to save loss history
        """
        super().__init__(save_dir)
        self.history = {
            'generator_loss': [],
            'discriminator_loss': []
        }
    
    def on_epoch_end(self, epoch: int, logs: Dict = None) -> None:
        """
        Record losses at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            logs: Training logs containing loss values
        """
        if logs is not None:
            # Extract loss values from logs
            g_loss = logs.get('generator_loss', 0)
            d_loss = logs.get('discriminator_loss', 0)
            
            self.history['generator_loss'].append(float(g_loss))
            self.history['discriminator_loss'].append(float(d_loss))
            
            # Save history to file
            self._save_history()
    
    def _save_history(self) -> None:
        """Save loss history to a file."""
        import json
        save_path = os.path.join(self.save_dir, 'loss_history.json')
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)


class CheckpointCallback(GANCallback):
    """
    Callback to save model checkpoints periodically.
    """
    
    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        save_dir: str,
        save_freq: int = 50
    ):
        """
        Initialize the checkpoint callback.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            save_dir: Directory to save checkpoints
            save_freq: Frequency (in epochs) to save checkpoints
        """
        super().__init__(save_dir)
        self.generator = generator
        self.discriminator = discriminator
        self.save_freq = save_freq
    
    def on_epoch_end(self, epoch: int, logs: Dict = None) -> None:
        """
        Save checkpoint at specified frequency.
        
        Args:
            epoch: Current epoch number
            logs: Training logs (unused)
        """
        if (epoch + 1) % self.save_freq == 0 or (epoch + 1) == self.params.get('epochs', float('inf')):
            checkpoint_path = os.path.join(
                self.save_dir,
                f'checkpoint_epoch_{epoch+1:04d}'
            )
            
            # Save models
            self.generator.save(f"{checkpoint_path}_generator")
            self.discriminator.save(f"{checkpoint_path}_discriminator")
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")
