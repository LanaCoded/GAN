"""
Training loop and trainer class for GAN-MNIST.
"""

import tensorflow as tf
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from ..config import TrainConfig, ModelConfig, DataConfig, PathsConfig
from ..models.generator import build_generator
from ..models.discriminator import build_discriminator
from ..losses import discriminator_loss, generator_loss
from ..utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


class GANTrainer:
    """
    Trainer class for conditional GAN on MNIST.
    
    Handles the adversarial training loop, checkpointing, and metrics tracking.
    """
    
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        paths_config: PathsConfig
    ):
        """
        Initialize the GAN trainer.
        
        Args:
            data_config: Configuration for data loading
            model_config: Configuration for model architecture
            train_config: Configuration for training parameters
            paths_config: Configuration for output paths
        """
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.paths_config = paths_config
        
        # Set random seeds for reproducibility
        set_seed(train_config.seed)
        
        # Build models
        self.generator = build_generator(model_config)
        self.discriminator = build_discriminator(model_config)
        
        # Create optimizers (matching original notebook exactly)
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=train_config.generator_learning_rate,
            beta_1=train_config.adam_beta_1
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=train_config.discriminator_learning_rate,
            beta_1=train_config.adam_beta_1
        )
        
        # Initialize training history
        self.history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'd_real_acc': [],
            'd_fake_acc': [],
            'd_real_mean': [],
            'd_fake_mean': []
        }
        
        # Fixed seed for consistent sample generation during training
        self.fixed_noise = tf.random.normal([16, model_config.latent_dim])
        self.fixed_labels = tf.constant([[i % 10] for i in range(16)], dtype=tf.int32)
        
        # Create output directories
        self._setup_directories()
        
        logger.info("GANTrainer initialized")
        logger.info(f"Generator parameters: {self.generator.count_params():,}")
        logger.info(f"Discriminator parameters: {self.discriminator.count_params():,}")
    
    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        create_dirs = [
            self.paths_config.checkpoint_dir,
            self.paths_config.samples_dir,
            self.paths_config.figures_dir,
            self.paths_config.logs_dir,
            self.paths_config.reports_dir
        ]
        
        from ..utils import create_directories
        create_directories(*create_dirs)
    
    @tf.function
    def train_step(self, images: tf.Tensor, labels: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Execute one training step (one batch).
        
        Args:
            images: Batch of real images [batch_size, 28, 28, 1]
            labels: Batch of labels [batch_size, 1]
            
        Returns:
            Dictionary containing loss values and metrics
        """
        batch_size = tf.shape(images)[0]
        
        # Generate noise and random labels for generator
        noise = tf.random.normal([batch_size, self.model_config.latent_dim])
        random_labels = tf.random.uniform(
            [batch_size, 1], 
            minval=0, 
            maxval=self.data_config.num_classes, 
            dtype=tf.int32
        )
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images = self.generator([noise, random_labels], training=True)
            
            # Discriminator forward pass
            real_output = self.discriminator([images, labels], training=True)
            fake_output = self.discriminator([generated_images, random_labels], training=True)
            
            # Calculate losses
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(
                real_output, 
                fake_output, 
                label_smoothing=self.train_config.label_smoothing
            )
        
        # Compute gradients
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        
        # Calculate discriminator accuracy and mean outputs
        # Real images should be classified as > 0 (since we use from_logits=True)
        real_pred = tf.sign(real_output)
        fake_pred = tf.sign(fake_output)
        
        d_real_acc = tf.reduce_mean(tf.cast(tf.equal(real_pred, 1), tf.float32))
        d_fake_acc = tf.reduce_mean(tf.cast(tf.equal(fake_pred, -1), tf.float32))
        
        d_real_mean = tf.reduce_mean(tf.nn.sigmoid(real_output))
        d_fake_mean = tf.reduce_mean(tf.nn.sigmoid(fake_output))
        
        return {
            'generator_loss': gen_loss,
            'discriminator_loss': disc_loss,
            'd_real_acc': d_real_acc,
            'd_fake_acc': d_fake_acc,
            'd_real_mean': d_real_mean,
            'd_fake_mean': d_fake_mean
        }
    
    def fit(self, dataset: tf.data.Dataset) -> Dict[str, list]:
        """
        Train the GAN for the specified number of epochs.
        
        Args:
            dataset: tf.data.Dataset yielding (images, labels) tuples
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.train_config.epochs} epochs")
        
        # Calculate steps per epoch if not provided
        if self.train_config.steps_per_epoch is None:
            # Try to get dataset size, fall back to estimation
            try:
                dataset_size = sum(1 for _ in dataset.unbatch())
                steps_per_epoch = max(1, dataset_size // self.data_config.batch_size)
            except:
                # If we can't determine size, use a reasonable default
                steps_per_epoch = 1000
                logger.warning(f"Could not determine dataset size, using {steps_per_epoch} steps/epoch")
        else:
            steps_per_epoch = self.train_config.steps_per_epoch
        
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        
        # Training loop
        for epoch in range(self.train_config.epochs):
            logger.info(f"Epoch {epoch+1}/{self.train_config.epochs}")
            
            # Initialize epoch metrics
            epoch_gen_loss = []
            epoch_disc_loss = []
            epoch_d_real_acc = []
            epoch_d_fake_acc = []
            epoch_d_real_mean = []
            epoch_d_fake_mean = []
            
            # Reset dataset iterator for each epoch
            dataset_iter = iter(dataset.take(steps_per_epoch))
            
            # Train on batches
            for step, (image_batch, label_batch) in enumerate(dataset_iter):
                if step >= steps_per_epoch:
                    break
                    
                # Execute training step
                step_outputs = self.train_step(image_batch, label_batch)
                
                # Collect metrics
                epoch_gen_loss.append(step_outputs['generator_loss'])
                epoch_disc_loss.append(step_outputs['discriminator_loss'])
                epoch_d_real_acc.append(step_outputs['d_real_acc'])
                epoch_d_fake_acc.append(step_outputs['d_fake_acc'])
                epoch_d_real_mean.append(step_outputs['d_real_mean'])
                epoch_d_fake_mean.append(step_outputs['d_fake_mean'])
                
                # Log progress periodically
                if step % 100 == 0:
                    logger.debug(
                        f"Step {step}: G_loss={step_outputs['generator_loss']:.4f}, "
                        f"D_loss={step_outputs['discriminator_loss']:.4f}"
                    )
            
            # Calculate epoch averages
            avg_gen_loss = tf.reduce_mean(epoch_gen_loss).numpy()
            avg_disc_loss = tf.reduce_mean(epoch_disc_loss).numpy()
            avg_d_real_acc = tf.reduce_mean(epoch_d_real_acc).numpy()
            avg_d_fake_acc = tf.reduce_mean(epoch_d_fake_acc).numpy()
            avg_d_real_mean = tf.reduce_mean(epoch_d_real_mean).numpy()
            avg_d_fake_mean = tf.reduce_mean(epoch_d_fake_mean).numpy()
            
            # Update history
            self.history['generator_loss'].append(float(avg_gen_loss))
            self.history['discriminator_loss'].append(float(avg_disc_loss))
            self.history['d_real_acc'].append(float(avg_d_real_acc))
            self.history['d_fake_acc'].append(float(avg_d_fake_acc))
            self.history['d_real_mean'].append(float(avg_d_real_mean))
            self.history['d_fake_mean'].append(float(avg_d_fake_mean))
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch+1} completed: "
                f"G_loss={avg_gen_loss:.4f}, "
                f"D_loss={avg_disc_loss:.4f}, "
                f"D_real_acc={avg_d_real_acc:.4f}, "
                f"D_fake_acc={avg_d_fake_acc:.4f}"
            )
            
            # Generate and save sample images
            self._generate_and_save_samples(epoch + 1)
            
            # Save checkpoint periodically
            if (epoch + 1) % 50 == 0 or (epoch + 1) == self.train_config.epochs:
                self.save_checkpoint(epoch + 1)
        
        logger.info("Training completed!")
        return self.history
    
    def _generate_and_save_samples(self, epoch: int) -> None:
        """
        Generate sample images using fixed noise and labels for consistent visualization.
        
        Args:
            epoch: Current epoch number
        """
        # Generate images using fixed noise and labels
        generated_images = self.generator([self.fixed_noise, self.fixed_labels], training=False)
        
        # Rescale from [-1, 1] to [0, 255] for saving
        generated_images = (generated_images * 127.5) + 127.5
        generated_images = tf.cast(generated_images, tf.uint8)
        
        # Save as individual images or create a grid
        # For now, we'll save the tensor for later visualization
        # In a full implementation, this would create a sample grid image
        
        sample_dir = os.path.join(self.paths_config.samples_dir, f"epoch_{epoch:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save individual samples
        for i in range(generated_images.shape[0]):
            img = generated_images[i, :, :, 0]  # Remove channel dimension
            # In a full implementation, we would save this as PNG
            # For now, we'll just note that samples were generated
            
        logger.debug(f"Generated samples for epoch {epoch}")
    
    def save_checkpoint(self, epoch: int) -> None:
        """
        Save model checkpoints.
        
        Args:
            epoch: Current epoch number
        """
        checkpoint_prefix = os.path.join(
            self.paths_config.checkpoint_dir, 
            f"checkpoint_epoch_{epoch:04d}"
        )
        
        # Save generator
        generator_path = f"{checkpoint_prefix}_generator"
        self.generator.save(generator_path)
        
        # Save discriminator
        discriminator_path = f"{checkpoint_prefix}_discriminator"
        self.discriminator.save(discriminator_path)
        
        # Save optimizer states and training history
        checkpoint_data = {
            'epoch': epoch,
            'generator_optimizer': self.generator_optimizer.get_weights(),
            'discriminator_optimizer': self.discriminator_optimizer.get_weights(),
            'history': self.history
        }
        
        # In a full implementation, we would save this as a pickle or similar
        logger.info(f"Checkpoint saved for epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoints.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
        """
        # Load generator
        generator_path = f"{checkpoint_path}_generator"
        self.generator = tf.keras.models.load_model(generator_path)
        
        # Load discriminator
        discriminator_path = f"{checkpoint_path}_discriminator"
        self.discriminator = tf.keras.models.load_model(discriminator_path)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def generate(
        self, 
        noise: tf.Tensor, 
        labels: tf.Tensor
    ) -> tf.Tensor:
        """
        Generate images from noise and labels.
        
        Args:
            noise: Noise vectors [batch_size, latent_dim]
            labels: Class labels [batch_size, 1]
            
        Returns:
            Generated images [batch_size, 28, 28, 1]
        """
        return self.generator([noise, labels], training=False)
    
    def discriminate(
        self, 
        images: tf.Tensor, 
        labels: tf.Tensor
    ) -> tf.Tensor:
        """
        Get discriminator output for images and labels.
        
        Args:
            images: Image tensors [batch_size, 28, 28, 1]
            labels: Class labels [batch_size, 1]
            
        Returns:
            Discriminator logits [batch_size, 1]
        """
        return self.discriminator([images, labels], training=False)


def create_trainer_from_config(config_path: str) -> GANTrainer:
    """
    Create a GANTrainer instance from a configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Initialized GANTrainer instance
    """
    import yaml
    from ..config import Config
    
    # Load configuration
    config = Config.from_yaml(config_path)
    
    # Create trainer
    trainer = GANTrainer(
        data_config=config.data,
        model_config=config.model,
        train_config=config.training,
        paths_config=config.outputs
    )
    
    return trainer
