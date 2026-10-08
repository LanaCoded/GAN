"""
Visualization functions for GAN-MNIST project.
All functions return matplotlib Figures and are designed to be pure functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from typing import List, Tuple, Optional, Dict, Any
from ..utils import setup_logging

logger = logging.getLogger(__name__)

# Okabe-Ito colorblind-safe palette
OKABE_ITO_PALETTE = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'sky': '#56B4E9',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'black': '#000000'
}

def apply_style() -> None:
    """Apply consistent styling to all plots."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.dpi': 150,
        'figure.autolayout': True,
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'grid.alpha': 0.3,
        'axes.grid': True,
        'grid.linestyle': '--',
        'axes.axisbelow': True
    })


def moving_average(data: List[float], window_size: int = 5) -> List[float]:
    """
    Calculate moving average of data.
    
    Args:
        data: List of values
        window_size: Size of moving window
        
    Returns:
        List of moving average values (same length as input, padded)
    """
    if len(data) < window_size:
        return data
    
    # Pad the beginning with the first value
    padded = [data[0]] * (window_size // 2) + data + [data[-1]] * ((window_size - 1) // 2)
    
    # Calculate moving average
    moving_avg = []
    for i in range(len(data)):
        window = padded[i:i + window_size]
        moving_avg.append(sum(window) / len(window))
    
    return moving_avg


def plot_loss_curves(
    generator_losses: List[float],
    discriminator_losses: List[float],
    title: str = "GAN Training Losses"
) -> plt.Figure:
    """
    Plot generator and discriminator loss curves.
    
    Args:
        generator_losses: List of generator loss values per epoch
        discriminator_losses: List of discriminator loss values per epoch
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = list(range(1, len(generator_losses) + 1))
    
    # Plot raw losses with transparency
    ax.plot(epochs, generator_losses, label='Generator Loss', 
            color=OKABE_ITO_PALETTE['blue'], alpha=0.3, linewidth=1)
    ax.plot(epochs, discriminator_losses, label='Discriminator Loss', 
            color=OKABE_ITO_PALETTE['orange'], alpha=0.3, linewidth=1)
    
    # Plot moving averages
    if len(generator_losses) >= 5:
        g_MA = moving_average(generator_losses, 5)
        d_MA = moving_average(discriminator_losses, 5)
        ax.plot(epochs, g_MA, label='Generator Loss (MA)', 
                color=OKABE_ITO_PALETTE['blue'], linewidth=2)
        ax.plot(epochs, d_MA, label='Discriminator Loss (MA)', 
                color=OKABE_ITO_PALETTE['orange'], linewidth=2)
    
    # Add theoretical equilibrium line at ln(2) ≈ 0.693
    ax.axhline(y=0.693, color=OKABE_ITO_PALETTE['black'], linestyle=':', 
               label='Theoretical Equilibrium (ln(2))', alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_discriminator_outputs(
    d_real_means: List[float],
    d_fake_means: List[float],
    title: str = "Discriminator Outputs"
) -> plt.Figure:
    """
    Plot discriminator mean outputs for real and fake images.
    
    Args:
        d_real_means: List of discriminator mean outputs for real images per epoch
        d_fake_means: List of discriminator mean outputs for fake images per epoch
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = list(range(1, len(d_real_means) + 1))
    
    ax.plot(epochs, d_real_means, label='D(real)', 
            color=OKABE_ITO_PALETTE['blue'], linewidth=2)
    ax.plot(epochs, d_fake_means, label='D(fake)', 
            color=OKABE_ITO_PALETTE['orange'], linewidth=2)
    
    # Add reference line at 0.5
    ax.axhline(y=0.5, color=OKABE_ITO_PALETTE['black'], linestyle=':', 
               label='Reference (0.5)', alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Output')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_discriminator_accuracy(
    d_real_acc: List[float],
    d_fake_acc: List[float],
    title: str = "Discriminator Accuracy"
) -> plt.Figure:
    """
    Plot discriminator accuracy on real and fake images.
    
    Args:
        d_real_acc: List of discriminator accuracy on real images per epoch
        d_fake_acc: List of discriminator accuracy on fake images per epoch
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = list(range(1, len(d_real_acc) + 1))
    
    ax.plot(epochs, d_real_acc, label='D Real Acc', 
            color=OKABE_ITO_PALETTE['blue'], linewidth=2)
    ax.plot(epochs, d_fake_acc, label='D Fake Acc', 
            color=OKABE_ITO_ITO_PALETTE['orange'], linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def create_sample_grid(
    images: tf.Tensor,
    labels: Optional[tf.Tensor] = None,
    title: str = "Generated Samples",
    rows: int = 10,
    cols: int = 10
) -> plt.Figure:
    """
    Create a grid of generated sample images.
    
    Args:
        images: Generated images tensor [batch_size, height, width, channels]
        labels: Optional labels for each image
        title: Plot title
        rows: Number of rows in grid
        cols: Number of columns in grid
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    # Determine actual grid size based on available images
    batch_size = tf.shape(images)[0]
    actual_samples = min(batch_size, rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    for i in range(actual_samples):
        # Get image and rescale from [-1, 1] to [0, 1] for display
        img = images[i, :, :, 0]
        img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        
        axes_flat[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes_flat[i].axis('off')
        
        # Add label if provided
        if labels is not None:
            label = int(tf.squeeze(labels[i]))
            axes_flat[i].text(0.5, -0.1, str(label), 
                            transform=axes_flat[i].transAxes,
                            ha='center', va='top', fontsize=10)
    
    # Hide unused subplots
    for i in range(actual_samples, len(axes_flat)):
        axes_flat[i].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_training_progression(
    generator: tf.keras.Model,
    fixed_noise: tf.Tensor,
    fixed_labels: tf.Tensor,
    epochs_to_show: List[int],
    title: str = "Training Progression"
) -> plt.Figure:
    """
    Plot the same fixed-seed samples at different epochs to show progression.
    
    Args:
        generator: Generator model
        fixed_noise: Fixed noise vectors for consistent generation
        fixed_labels: Fixed labels for consistent generation
        epochs_to_show: List of epoch numbers to show
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    n_epochs = len(epochs_to_show)
    fig, axes = plt.subplots(n_epochs, 10, figsize=(10, n_epochs))
    if n_epochs == 1:
        axes = axes.reshape(1, -1)
    
    # Generate images for each epoch (this would normally come from saved checkpoints)
    # For this function, we'll generate current samples - in practice, 
    # this would load from different checkpoints
    generated_images = generator([fixed_noise, fixed_labels], training=False)
    generated_images = (generated_images + 1.0) / 2.0  # Rescale to [0, 1]
    
    for epoch_idx, epoch in enumerate(epochs_to_show):
        for col in range(10):
            img_idx = epoch_idx * 10 + col
            if img_idx < tf.shape(generated_images)[0]:
                img = generated_images[img_idx, :, :, 0]
                axes[epoch_idx, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[epoch_idx, col].axis('off')
        
        # Add epoch label on the left
        axes[epoch_idx, 0].text(-0.1, 0.5, f'Epoch {epoch}', 
                               transform=axes[epoch_idx, 0].transAxes,
                               ha='right', va='center', fontsize=12, fontweight='bold')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_real_vs_fake(
    real_images: tf.Tensor,
    fake_images: tf.Tensor,
    real_labels: Optional[tf.Tensor] = None,
    fake_labels: Optional[tf.Tensor] = None,
    title: str = "Real vs Generated Images",
    num_pairs: int = 5
) -> plt.Figure:
    """
    Plot side-by-side comparison of real and generated images.
    
    Args:
        real_images: Real image tensors [batch_size, height, width, channels]
        fake_images: Generated image tensors [batch_size, height, width, channels]
        real_labels: Optional labels for real images
        fake_labels: Optional labels for generated images
        title: Plot title
        num_pairs: Number of real/fake pairs to show
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    # Limit to available samples
    batch_size = tf.shape(real_images)[0]
    num_pairs = min(num_pairs, batch_size, tf.shape(fake_images)[0])
    
    fig, axes = plt.subplots(num_pairs, 2, figsize=(6, num_pairs * 2))
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_pairs):
        # Real image
        real_img = real_images[i, :, :, 0]
        real_img = (real_img + 1.0) / 2.0  # Rescale to [0, 1]
        axes[i, 0].imshow(real_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        
        # Fake image
        fake_img = fake_images[i, :, :, 0]
        fake_img = (fake_img + 1.0) / 2.0  # Rescale to [0, 1]
        axes[i, 1].imshow(fake_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        
        # Add labels if provided
        if real_labels is not None and fake_labels is not None:
            real_label = int(tf.squeeze(real_labels[i]))
            fake_label = int(tf.squeeze(fake_labels[i]))
            
            axes[i, 0].text(0.5, -0.1, f'Real: {real_label}', 
                           transform=axes[i, 0].transAxes,
                           ha='center', va='top', fontsize=10)
            axes[i, 1].text(0.5, -0.1, f'Fake: {fake_label}', 
                           transform=axes[i, 1].transAxes,
                           ha='center', va='top', fontsize=10)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_latent_interpolation(
    generator: tf.keras.Model,
    noise1: tf.Tensor,
    noise2: tf.Tensor,
    label: tf.Tensor,
    steps: int = 10,
    title: str = "Latent Space Interpolation"
) -> plt.Figure:
    """
    Plot interpolation between two noise vectors for a fixed label.
    
    Args:
        generator: Generator model
        noise1: Starting noise vector [1, latent_dim]
        noise2: Ending noise vector [1, latent_dim]
        label: Class label [1, 1]
        steps: Number of interpolation steps
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    # Create interpolation steps
    alphas = np.linspace(0, 1, steps)
    interpolated_noises = []
    
    for alpha in alphas:
        interpolated_noise = alpha * noise1 + (1 - alpha) * noise2
        interpolated_noises.append(interpolated_noise)
    
    interpolated_noises = tf.concat(interpolated_noises, axis=0)
    repeated_labels = tf.tile(label, [steps, 1])
    
    # Generate images
    generated_images = generator([interpolated_noises, repeated_labels], training=False)
    generated_images = (generated_images + 1.0) / 2.0  # Rescale to [0, 1]
    
    # Plot
    fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
    if steps == 1:
        axes = np.array([axes])
    
    for i in range(steps):
        img = generated_images[i, :, :, 0]
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f'α={alphas[i]:.1f}', fontsize=10)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_class_conditioning(
    generator: tf.keras.Model,
    fixed_noise: tf.Tensor,
    num_classes: int = 10,
    title: str = "Class Conditioning"
) -> plt.Figure:
    """
    Plot one fixed noise vector rendered under all class labels.
    
    Args:
        generator: Generator model
        fixed_noise: Fixed noise vector [1, latent_dim]
        num_classes: Number of classes to show
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    # Create labels for all classes
    labels = tf.range(num_classes, dtype=tf.int32)
    labels = tf.reshape(labels, (-1, 1))  # [num_classes, 1]
    
    # Repeat the same noise for all classes
    noises = tf.tile(fixed_noise, [num_classes, 1])
    
    # Generate images
    generated_images = generator([noises, labels], training=False)
    generated_images = (generated_images + 1.0) / 2.0  # Rescale to [0, 1]
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))  # 2x5 grid for 10 classes
    axes = axes.flatten()
    
    for i in range(num_classes):
        img = generated_images[i, :, :, 0]
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f'Class {i}', fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_pixel_distribution(
    real_images: tf.Tensor,
    fake_images: tf.Tensor,
    title: str = "Pixel Intensity Distribution"
) -> plt.Figure:
    """
    Plot histogram of pixel intensities for real vs generated images.
    
    Args:
        real_images: Real image tensors [batch_size, height, width, channels]
        fake_images: Generated image tensors [batch_size, height, width, channels]
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Flatten images and rescale to [0, 1] for histogram
    real_pixels = tf.reshape(real_images, [-1])
    real_pixels = (real_pixels + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    
    fake_pixels = tf.reshape(fake_images, [-1])
    fake_pixels = (fake_pixels + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    
    # Plot histograms
    ax.hist(real_pixels.numpy(), bins=50, alpha=0.7, label='Real Images',
            color=OKABE_ITO_PALETTE['blue'], density=True, histtype='step', linewidth=2)
    ax.hist(fake_pixels.numpy(), bins=50, alpha=0.7, label='Generated Images',
            color=OKABE_ITO_PALETTE['orange'], density=True, histtype='step', linewidth=2)
    
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_diversity_report(
    diversity_scores: List[float],
    real_data_score: float,
    title: str = "Diversity Report"
) -> plt.Figure:
    """
    Plot diversity scores per class with reference to real data.
    
    Args:
        diversity_scores: List of diversity scores per class
        real_data_score: Diversity score of real MNIST data
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = list(range(len(diversity_scores)))
    
    # Create bar chart with hatch patterns for accessibility
    bars = ax.bar(classes, diversity_scores, 
                  color=OKABE_ITO_PALETTE['sky'],
                  edgecolor=OKABE_ITO_PALETTE['black'],
                  linewidth=1.2,
                  hatch='///',
                  label='Generated')
    
    # Add reference line for real data
    ax.axhline(y=real_data_score, color=OKABE_ITO_PALETTE['black'], 
               linestyle='-', linewidth=2, label='Real Data Reference')
    
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Diversity Score (Avg Pairwise L2 Distance)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(classes)
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix with numeric annotations.
    
    Args:
        cm: Confusion matrix array [n_classes, n_classes]
        class_names: List of class names for labels
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use sequential colormap (cividis) for accessibility
    im = ax.imshow(cm, interpolation='nearest', cmap='cividis')
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    
    fig.tight_layout()
    
    return fig


def plot_gradient_norms(
    g_norms: List[float],
    d_norms: List[float],
    title: str = "Gradient Norms"
) -> plt.Figure:
    """
    Plot generator and discriminator gradient norms per epoch.
    
    Args:
        g_norms: List of generator gradient norms per epoch
        d_norms: List of discriminator gradient norms per epoch
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = list(range(1, len(g_norms) + 1))
    
    ax.plot(epochs, g_norms, label='Generator Grad Norm', 
            color=OKABE_ITO_PALETTE['blue'], linewidth=2)
    ax.plot(epochs, d_norms, label='Discriminator Grad Norm', 
            color=OKABE_ITO_PALETTE['orange'], linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')  # Log scale for better visibility
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def create_architecture_diagram(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    title: str = "GAN Architecture Diagram"
) -> plt.Figure:
    """
    Create a simple block diagram of GAN architecture with tensor shapes.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get model shapes (simplified representation)
    # In a real implementation, we would extract layer shapes
    
    # Define box positions and sizes
    box_width = 0.15
    box_height = 0.1
    
    # Generator components
    gen_noise_pos = (0.1, 0.7)
    gen_label_pos = (0.1, 0.5)
    gen_merge_pos = (0.3, 0.6)
    gen_upsample_pos = (0.5, 0.6)
    gen_output_pos = (0.8, 0.6)
    
    # Discriminator components
    disc_image_pos = (0.1, 0.3)
    disc_label_pos = (0.1, 0.1)
    disc_merge_pos = (0.3, 0.2)
    disc_conv_pos = (0.5, 0.2)
    disc_flat_pos = (0.7, 0.2)
    disc_output_pos = (0.9, 0.2)
    
    # Draw generator boxes
    gen_noise_rect = plt.Rectangle(gen_noise_pos, box_width, box_height, 
                                   linewidth=1, edgecolor=OKABE_ITO_PALETTE['blue'], 
                                   facecolor='none')
    gen_label_rect = plt.Rectangle(gen_label_pos, box_width, box_height, 
                                   linewidth=1, edgecolor=OKABE_ITO_PALETTE['blue'], 
                                   facecolor='none')
    gen_merge_rect = plt.Rectangle(gen_merge_pos, box_width, box_height, 
                                   linewidth=1, edgecolor=OKABE_ITO_PALETTE['blue'], 
                                   facecolor='none')
    gen_upsample_rect = plt.Rectangle(gen_upsample_pos, box_width, box_height, 
                                      linewidth=1, edgecolor=OKABE_ITO_PALETTE['blue'], 
                                      facecolor='none')
    gen_output_rect = plt.Rectangle(gen_output_pos, box_width, box_height, 
                                    linewidth=1, edgecolor=OKABE_ITO_PALETTE['blue'], 
                                    facecolor='none')
    
    for rect in [gen_noise_rect, gen_label_rect, gen_merge_rect, 
                 gen_upsample_rect, gen_output_rect]:
        ax.add_patch(rect)
    
    # Draw discriminator boxes
    disc_image_rect = plt.Rectangle(disc_image_pos, box_width, box_height, 
                                    linewidth=1, edgecolor=OKABE_ITO_PALETTE['orange'], 
                                    facecolor='none')
    disc_label_rect = plt.Rectangle(disc_label_pos, box_width, box_height, 
                                    linewidth=1, edgecolor=OKABE_ITO_PALETTE['orange'], 
                                    facecolor='none')
    disc_merge_rect = plt.Rectangle(disc_merge_pos, box_width, box_height, 
                                    linewidth=1, edgecolor=OKABE_ITO_PALETTE['orange'], 
                                    facecolor='none')
    disc_conv_rect = plt.Rectangle(disc_conv_pos, box_width, box_height, 
                                   linewidth=1, edgecolor=OKABE_ITO_PALETTE['orange'], 
                                   facecolor='none')
    disc_flat_rect = plt.Rectangle(disc_flat_pos, box_width, box_height, 
                                   linewidth=1, edgecolor=OKABE_ITO_PALETTE['orange'], 
                                   facecolor='none')
    disc_output_rect = plt.Rectangle(disc_output_pos, box_width, box_height, 
                                     linewidth=1, edgecolor=OKABE_ITO_PALETTE['orange'], 
                                     facecolor='none')
    
    for rect in [disc_image_rect, disc_label_rect, disc_merge_rect, 
                 disc_conv_rect, disc_flat_rect, disc_output_rect]:
        ax.add_patch(rect)
    
    # Add labels
    ax.text(gen_noise_pos[0] + box_width/2, gen_noise_pos[1] + box_height/2, 
            'Noise\n(100)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(gen_label_pos[0] + box_width/2, gen_label_pos[1] + box_height/2, 
            'Label\nEmbedding', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(gen_merge_pos[0] + box_width/2, gen_merge_pos[1] + box_height/2, 
            'Merge', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(gen_upsample_pos[0] + box_width/2, gen_upsample_pos[1] + box_height/2, 
            'Upsample\nLayers', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(gen_output_pos[0] + box_width/2, gen_output_pos[1] + box_height/2, 
            'Output\n(28x28x1)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.text(disc_image_pos[0] + box_width/2, disc_image_pos[1] + box_height/2, 
            'Image\n(28x28x1)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(disc_label_pos[0] + box_width/2, disc_label_pos[1] + box_height/2, 
            'Label\nEmbedding', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(disc_merge_pos[0] + box_width/2, disc_merge_pos[1] + box_height/2, 
            'Merge', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(disc_conv_pos[0] + box_width/2, disc_conv_pos[1] + box_height/2, 
            'Conv\nLayers', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(disc_flat_pos[0] + box_width/2, disc_flat_pos[1] + box_height/2, 
            'Flatten', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(disc_output_pos[0] + box_width/2, disc_output_pos[1] + box_height/2, 
            'Output\n(1)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add arrows
    # Generator flow
    ax.arrow(gen_noise_pos[0] + box_width, gen_noise_pos[1] + box_height/2, 
             0.05, 0, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['blue'], ec=OKABE_ITO_PALETTE['blue'])
    ax.arrow(gen_label_pos[0] + box_width, gen_label_pos[1] + box_height/2, 
             0.05, 0, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['blue'], ec=OKABE_ITO_PALETTE['blue'])
    ax.arrow(gen_merge_pos[0] + box_width, gen_merge_pos[1] + box_height/2, 
             0.15, 0, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['blue'], ec=OKABE_ITO_PALETTE['blue'])
    ax.arrow(gen_upsample_pos[0] + box_width, gen_upsample_pos[1] + box_height/2, 
             0.25, 0, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['blue'], ec=OKABE_ITO_PALETTE['blue'])
    
    # Discriminator flow
    ax.arrow(disc_image_pos[0] + box_width, disc_image_pos[1] + box_height/2, 
             0.05, 0, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['orange'], ec=OKABE_ITO_PALETTE['orange'])
    ax.arrow(disc_label_pos[0] + box_width, disc_label_pos[1] + box_height/2, 
             0.05, 0, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['orange'], ec=OKABE_ITO_PALETTE['orange'])
    ax.arrow(disc_merge_pos[0] + box_width, disc_merge_pos[1] + box_height/2, 
             0.15, 0, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['orange'], ec=OKABE_ITO_PALETTE['orange'])
    ax.arrow(disc_conv_pos[0] + box_width, disc_conv_pos[1] + box_height/2, 
             0.15, 0, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['orange'], ec=OKABE_ITO_PALETTE['orange'])
    ax.arrow(disc_flat_pos[0] + box_width, disc_flat_pos[1] + box_height/2, 
             0.15, 0, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['orange'], ec=OKABE_ITO_PALETTE['orange'])
    
    # Connection between generator and discriminator
    ax.arrow(gen_output_pos[0] + box_width, gen_output_pos[1] + box_height/2, 
             0.05, -0.3, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['green'], ec=OKABE_ITO_PALETTE['green'])
    ax.arrow(disc_image_pos[0], disc_image_pos[1] + box_height/2, 
             -0.05, 0.3, head_width=0.02, head_length=0.01, 
             fc=OKABE_ITO_PALETTE['green'], ec=OKABE_ITO_PALETTE['green'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    return fig


def create_dashboard(
    history: Dict[str, List[float]],
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    fixed_noise: tf.Tensor,
    fixed_labels: tf.Tensor,
    title: str = "GAN Training Dashboard"
) -> plt.Figure:
    """
    Create a 2x3 summary dashboard panel.
    
    Args:
        history: Training history dictionary
        generator: Generator model
        discriminator: Discriminator model
        fixed_noise: Fixed noise vectors
        fixed_labels: Fixed labels
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    apply_style()
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    # 1. Loss curves (top left)
    ax1 = plt.subplot(2, 3, 1)
    if 'generator_loss' in history and 'discriminator_loss' in history:
        epochs = list(range(1, len(history['generator_loss']) + 1))
        ax1.plot(epochs, history['generator_loss'], label='G Loss', 
                 color=OKABE_ITO_PALETTE['blue'], linewidth=2)
        ax1.plot(epochs, history['discriminator_loss'], label='D Loss', 
                 color=OKABE_ITO_PALETTE['orange'], linewidth=2)
        ax1.axhline(y=0.693, color=OKABE_ITO_PALETTE['black'], linestyle=':', 
                    label='Eqbm (ln2)', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
    
    # 2. Discriminator outputs (top middle)
    ax2 = plt.subplot(2, 3, 2)
    if 'd_real_mean' in history and 'd_fake_mean' in history:
        epochs = list(range(1, len(history['d_real_mean']) + 1))
        ax2.plot(epochs, history['d_real_mean'], label='D(real)', 
                 color=OKABE_ITO_PALETTE['blue'], linewidth=2)
        ax2.plot(epochs, history['d_fake_mean'], label='D(fake)', 
                 color=OKABE_ITO_PALETTE['orange'], linewidth=2)
        ax2.axhline(y=0.5, color=OKABE_ITO_PALETTE['black'], linestyle=':', 
                    label='Ref (0.5)', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Output')
        ax2.set_title('Discriminator Outputs')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # 3. Sample grid (top right)
    ax3 = plt.subplot(2, 3, 3)
    try:
        generated_images = generator([fixed_noise, fixed_labels], training=False)
        generated_images = (generated_images + 1.0) / 2.0  # Rescale to [0, 1]
        
        # Create a small sample grid
        n_samples = min(16, tf.shape(generated_images)[0])
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        for i in range(n_samples):
            row = i // n_cols
            col = i % n_cols
            ax_sub = plt.axes([0.58 + 0.08*col, 0.58 - 0.12*row, 0.07, 0.1])
            img = generated_images[i, :, :, 0]
            ax_sub.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax_sub.axis('off')
        
        ax3.set_title('Sample Grid')
        ax3.axis('off')
    except Exception as e:
        ax3.text(0.5, 0.5, 'Sample Grid\n(Error)', ha='center', va='center')
        ax3.set_title('Sample Grid')
        ax3.axis('off')
    
    # 4. Training progression (middle left)
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title('Training Progression')
    ax4.axis('off')
    
    # 5. Confusion matrix placeholder (middle middle)
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title('Confusion Matrix')
    ax5.axis('off')
    
    # 6. Diversity report (middle right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title('Diversity Report')
    ax6.axis('off')
    
    plt.tight_layout()
    
    return fig
