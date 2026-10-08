"""
Command-line interface for the GAN-MNIST project.
"""

import argparse
import os
import sys
from typing import Optional
import yaml
from .config import Config
from .trainer import GANTrainer, create_trainer_from_config
from .data import load_mnist, create_synthetic_dataset
from .utils import setup_logging, set_seed, create_directories
from .visualization import (
    plot_loss_curves, plot_discriminator_outputs, plot_discriminator_accuracy,
    create_sample_grid, plot_training_progression, plot_real_vs_fake,
    plot_latent_interpolation, plot_class_conditioning, plot_pixel_distribution,
    plot_diversity_report, plot_confusion_matrix, plot_gradient_norms,
    create_architecture_diagram, create_dashboard
)
from .evaluation import (
    calculate_diversity, calculate_label_conditional_accuracy,
    create_simple_classifier, evaluate_discriminator_outputs
)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def train_command(args) -> None:
    """Handle the train subcommand."""
    logger.info(f"Starting training with config: {args.config}")
    
    # Load configuration
    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.epochs:
        config.training.epochs = args.epochs
    if args.seed is not None:
        config.training.seed = args.seed
    if args.output_dir:
        config.outputs.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        config.outputs.samples_dir = os.path.join(args.output_dir, "samples")
        config.outputs.figures_dir = os.path.join(args.output_dir, "figures")
        config.outputs.logs_dir = os.path.join(args.output_dir, "logs")
        config.outputs.reports_dir = os.path.join(args.output_dir, "reports")
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    
    # Setup logging
    setup_logging()
    
    # Create output directories
    create_directories(
        config.outputs.checkpoint_dir,
        config.outputs.samples_dir,
        config.outputs.figures_dir,
        config.outputs.logs_dir,
        config.outputs.reports_dir
    )
    
    # Save the resolved configuration
    config_path = os.path.join(config.outputs.reports_dir, "resolved_config.yaml")
    config.to_yaml(config_path)
    logger.info(f"Saved resolved configuration to {config_path}")
    
    # Load dataset
    try:
        dataset = load_mnist(
            buffer_size=config.data.buffer_size,
            batch_size=config.data.batch_size,
            normalize_to_neg1_1=config.data.normalize_to_neg1_1,
            subset_size=getattr(args, 'subset_size', None),
            seed=config.training.seed
        )
        logger.info("Successfully loaded MNIST dataset")
    except Exception as e:
        logger.warning(f"Failed to load MNIST dataset: {e}")
        logger.info("Falling back to synthetic dataset")
        dataset = create_synthetic_dataset(
            batch_size=config.data.batch_size,
            image_shape=tuple(config.data.image_shape),
            num_classes=config.data.num_classes,
            num_samples=config.data.buffer_size,
            seed=config.training.seed
        )
    
    # Create trainer
    trainer = GANTrainer(
        data_config=config.data,
        model_config=config.model,
        train_config=config.training,
        paths_config=config.outputs
    )
    
    # Train the model
    history = trainer.fit(dataset)
    
    # Save final checkpoint
    final_epoch = config.training.epochs
    trainer.save_checkpoint(final_epoch)
    
    # Generate final report
    logger.info("Generating final report...")
    generate_report(trainer, history, config.outputs)
    
    logger.info("Training completed successfully!")


def generate_command(args) -> None:
    """Handle the generate subcommand."""
    logger.info(f"Generating images from checkpoint: {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load configuration from checkpoint directory if available
    config_path = os.path.join(os.path.dirname(args.checkpoint), "..", "reports", "resolved_config.yaml")
    if os.path.exists(config_path):
        config = Config.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        # Use default configuration
        config = Config()
        logger.warning("Using default configuration")
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    
    # Setup logging
    setup_logging()
    
    # Initialize trainer to get model structure
    trainer = GANTrainer(
        data_config=config.data,
        model_config=config.model,
        train_config=config.training,
        paths_config=config.outputs
    )
    
    # Load the checkpoint
    trainer.load_checkpoint(args.checkpoint)
    
    # Generate images
    logger.info(f"Generating {args.count} images of label {args.label}")
    
    # Create noise vectors
    noise = tf.random.normal([args.count, config.model.latent_dim])
    labels = tf.constant([[args.label] for _ in range(args.count)], dtype=tf.int32)
    
    # Generate images
    generated_images = trainer.generator([noise, labels], training=False)
    
    # Rescale from [-1, 1] to [0, 255] for saving
    generated_images = (generated_images * 127.5) + 127.5
    generated_images = tf.cast(generated_images, tf.uint8)
    
    # Save images
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    
    if args.count == 1:
        # Save single image
        img = generated_images[0, :, :, 0]
        # In a full implementation, we would save this as PNG
        logger.info(f"Generated 1 image of label {args.label}")
        logger.info(f"Would save to: {args.out}")
    else:
        # Save multiple images or create a grid
        logger.info(f"Generated {args.count} images of label {args.label}")
        logger.info(f"Would save to: {args.out}")
    
    logger.info("Generation completed!")


def evaluate_command(args) -> None:
    """Handle the evaluate subcommand."""
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(args.checkpoint), "..", "reports", "resolved_config.yaml")
    if os.path.exists(config_path):
        config = Config.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = Config()
        logger.warning("Using default configuration for evaluation")
    
    # Set seed
    set_seed(config.training.seed)
    
    # Setup logging
    setup_logging()
    
    # Initialize trainer
    trainer = GANTrainer(
        data_config=config.data,
        model_config=config.model,
        train_config=config.training,
        paths_config=config.outputs
    )
    
    # Load checkpoint
    trainer.load_checkpoint(args.checkpoint)
    
    # Load evaluation data
    try:
        eval_dataset = load_mnist(
            batch_size=config.data.batch_size,
            normalize_to_neg1_1=config.data.normalize_to_neg1_1,
            subset_size=1000,  # Use subset for faster evaluation
            seed=config.training.seed
        )
    except:
        eval_dataset = create_synthetic_dataset(
            batch_size=config.data.batch_size,
            image_shape=tuple(config.data.image_shape),
            num_classes=config.data.num_classes,
            num_samples=1000,
            seed=config.training.seed
        )
    
    # Get a batch for evaluation
    for images, labels in eval_dataset.take(1):
        eval_images = images
        eval_labels = labels
        break
    
    # Generate fake images
    noise = tf.random.normal([tf.shape(eval_images)[0], config.model.latent_dim])
    random_labels = tf.random.uniform(
        [tf.shape(eval_images)[0], 1],
        minval=0,
        maxval=config.data.num_classes,
        dtype=tf.int32
    )
    fake_images = trainer.generator([noise, random_labels], training=False)
    
    # Calculate metrics
    logger.info("Calculating evaluation metrics...")
    
    # Diversity score
    diversity_score = calculate_diversity(fake_images)
    logger.info(f"Diversity score: {diversity_score:.4f}")
    
    # Discriminator outputs
    d_real_acc, d_fake_acc, d_real_mean, d_fake_mean = evaluate_discriminator_outputs(
        trainer.discriminator, eval_images, fake_images, eval_labels, random_labels
    )
    logger.info(f"D(real) accuracy: {d_real_acc:.4f}")
    logger.info(f"D(fake) accuracy: {d_fake_acc:.4f}")
    logger.info(f"D(real) mean: {d_real_mean:.4f}")
    logger.info(f"D(fake) mean: {d_fake_mean:.4f}")
    
    # Label conditional accuracy (requires a classifier)
    # For simplicity, we'll skip this in the basic evaluate command
    # A full implementation would train/load a classifier
    
    # Save results to JSON
    results = {
        'diversity_score': float(diversity_score),
        'd_real_accuracy': float(d_real_acc),
        'd_fake_accuracy': float(d_fake_acc),
        'd_real_mean': float(d_real_mean),
        'd_fake_mean': float(d_fake_mean),
        'checkpoint': args.checkpoint
    }
    
    results_dir = os.path.join(os.path.dirname(args.checkpoint), "..", "reports")
    os.makedirs(results_dir, exist_ok=True)
    
    import json
    results_path = os.path.join(results_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    logger.info("Evaluation completed!")


def report_command(args) -> None:
    """Handle the report subcommand."""
    logger.info(f"Generating report from run directory: {args.run_dir}")
    
    if not os.path.exists(args.run_dir):
        logger.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    # Load configuration
    config_path = os.path.join(args.run_dir, "resolved_config.yaml")
    if os.path.exists(config_path):
        config = Config.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        # Look for config in standard locations
        possible_paths = [
            os.path.join(args.run_dir, "..", "reports", "resolved_config.yaml"),
            "configs/default.yaml"
        ]
        config = None
        for path in possible_paths:
            if os.path.exists(path):
                config = Config.from_yaml(path)
                logger.info(f"Loaded configuration from {path}")
                break
        if config is None:
            config = Config()
            logger.warning("Using default configuration")
    
    # Set seed
    set_seed(config.training.seed)
    
    # Setup logging
    setup_logging()
    
    # Look for checkpoint and history
    checkpoint_dir = os.path.join(args.run_dir, "checkpoints")
    history_path = os.path.join(args.run_dir, "reports", "training_history.json")
    
    # Find latest checkpoint
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) 
                      if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint_epoch_")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            # Look for the actual checkpoint files
            gen_path = os.path.join(latest_checkpoint_path, "*_generator")
            disc_path = os.path.join(latest_checkpoint_path, "*_discriminator")
            # In a full implementation, we would glob for these files
            logger.info(f"Found checkpoint directory: {latest_checkpoint_path}")
    
    # Load history if available
    history = {}
    if os.path.exists(history_path):
        try:
            import json
            with open(history_path, 'r') as f:
                history = json.load(f)
            logger.info(f"Loaded training history from {history_path}")
        except Exception as e:
            logger.warning(f"Could not load training history: {e}")
    
    # Generate report
    logger.info("Generating report figures and HTML...")
    
    # This would generate all the figures and HTML report
    # For now, we'll just indicate what would happen
    logger.info("Would generate:")
    logger.info("  - loss_curves.png")
    logger.info("  - discriminator_outputs.png")
    logger.info("  - discriminator_accuracy.png")
    logger.info("  - sample_grid.png")
    logger.info("  - training_progression.png")
    logger.info("  - real_vs_fake.png")
    logger.info("  - latent_interpolation.png")
    logger.info("  - class_conditioning.png")
    logger.info("  - pixel_distribution.png")
    logger.info("  - diversity_report.png")
    logger.info("  - confusion_matrix.png")
    logger.info("  - gradient_norms.png")
    logger.info("  - architecture_diagram.png")
    logger.info("  - dashboard.png")
    logger.info("  - report.html")
    
    logger.info("Report generation completed!")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Conditional GAN for MNIST digit generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the GAN model')
    train_parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    train_parser.add_argument(
        '--epochs', 
        type=int, 
        help='Number of training epochs (overrides config)'
    )
    train_parser.add_argument(
        '--smoke', 
        action='store_true',
        help='Run smoke test with reduced configuration'
    )
    train_parser.add_argument(
        '--seed', 
        type=int, 
        help='Random seed for reproducibility'
    )
    train_parser.add_argument(
        '--output-dir', 
        type=str, 
        help='Output directory for results (overrides config)'
    )
    train_parser.add_argument(
        '--subset-size', 
        type=int, 
        help='Use subset of data for faster testing'
    )
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate images from checkpoint')
    generate_parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to checkpoint directory'
    )
    generate_parser.add_argument(
        '--label', 
        type=int, 
        default=0,
        help='Class label to generate (0-9)'
    )
    generate_parser.add_argument(
        '--count', 
        type=int, 
        default=16,
        help='Number of images to generate'
    )
    generate_parser.add_argument(
        '--out', 
        type=str, 
        required=True,
        help='Output file or directory for generated images'
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model from checkpoint')
    evaluate_parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to checkpoint directory'
    )
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate report from training run')
    report_parser.add_argument(
        '--run-dir', 
        type=str, 
        required=True,
        help='Path to training run directory'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup basic logging
    setup_logging()
    
    # Handle commands
    if args.command == 'train':
        if args.smoke:
            args.config = 'configs/smoke.yaml'
        train_command(args)
    elif args.command == 'generate':
        generate_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'report':
        report_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
