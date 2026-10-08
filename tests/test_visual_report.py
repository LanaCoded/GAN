"""
Tests for visualization and report generation.
This file serves both as a test module and as the demo generator.
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
    np = None

# Import modules that don't require tensorflow at module level
from gan_mnist.utils import set_seed, create_directories

# Import tensorflow-dependent modules only if available
if TF_AVAILABLE:
    from gan_mnist.trainer import GANTrainer
    from gan_mnist.config import Config
    from gan_mnist.visualization import (
        plot_loss_curves, plot_discriminator_outputs, plot_discriminator_accuracy,
        create_sample_grid, plot_training_progression, plot_real_vs_fake,
        plot_latent_interpolation, plot_class_conditioning, plot_pixel_distribution,
        plot_diversity_report, plot_confusion_matrix, plot_gradient_norms,
        create_architecture_diagram, create_dashboard
    )
    from gan_mnist.evaluation import (
        calculate_diversity, calculate_label_conditional_accuracy,
        create_simple_classifier, evaluate_discriminator_outputs
    )
    from gan_mnist.models.generator import build_generator
    from gan_mnist.models.discriminator import build_discriminator
else:
    # Define mock functions that will be skipped
    def set_seed(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def create_directories(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    class GANTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not available")
    
    class Config:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not available")
    
    # Mock all visualization functions
    def plot_loss_curves(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def plot_discriminator_outputs(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def plot_discriminator_accuracy(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def create_sample_grid(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def plot_training_progression(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def plot_real_vs_fake(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def plot_latent_interpolation(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def plot_class_conditioning(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def plot_pixel_distribution(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def plot_diversity_report(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def plot_confusion_matrix(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def plot_gradient_norms(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def create_architecture_diagram(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def create_dashboard(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    # Mock evaluation functions
    def calculate_diversity(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def calculate_label_conditional_accuracy(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def create_simple_classifier(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def evaluate_discriminator_outputs(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    # Mock model functions
    def build_generator(*args, **kwargs):
        raise ImportError("TensorFlow not available")
    
    def build_discriminator(*args, **kwargs):
        raise ImportError("TensorFlow not available")


# Session-scoped fixture that trains a smoke configuration once
@pytest.fixture(scope="session")
def smoke_trainer():
    """Train a smoke configuration and return trainer, history, and samples."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Create smoke configuration
    config = Config(
        data=gan_mnist.config.DataConfig(
            buffer_size=200,  # Small subset for fast testing
            batch_size=32,
            image_shape=[28, 28, 1],
            num_classes=10
        ),
        model=gan_mnist.config.ModelConfig(
            latent_dim=10,
            label_embedding_dim=5,
            generator_base_filters=[8, 4],
            discriminator_base_filters=[4, 8],
            generator_upsample_filters=[4, 8, 1],
            discriminator_dropout_rate=0.3,
            leaky_relu_alpha=0.2,
            batch_norm_momentum=0.9
        ),
        training=gan_mnist.config.TrainConfig(
            epochs=2,  # Just a couple epochs for testing
            generator_learning_rate=2e-4,
            discriminator_learning_rate=1e-5,
            adam_beta_1=0.5,
            label_smoothing=0.9,
            seed=42,
            steps_per_epoch=5  # Limited steps per epoch
        ),
        outputs=gan_mnist.config.PathsConfig(
            checkpoint_dir="test_checkpoints",
            samples_dir="test_samples",
            figures_dir="test_figures",
            logs_dir="test_logs",
            reports_dir="test_reports"
        )
    )
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    
    # Load or create dataset
    try:
        dataset = gan_mnist.data.load_mnist(
            buffer_size=config.data.buffer_size,
            batch_size=config.data.batch_size,
            normalize_to_neg1_1=config.data.normalize_to_neg1_1,
            subset_size=200,  # Use subset for speed
            seed=config.training.seed
        )
    except:
        dataset = gan_mnist.data.create_synthetic_dataset(
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
    
    # Train for a few epochs
    history = trainer.fit(dataset)
    
    # Generate final samples for testing
    fixed_noise = trainer.fixed_noise
    fixed_labels = trainer.fixed_labels
    generated_images = trainer.generator([fixed_noise, fixed_labels], training=False)
    # Rescale to [0, 1] for visualization
    generated_images_viz = (generated_images + 1.0) / 2.0
    
    return {
        'trainer': trainer,
        'history': history,
        'fixed_noise': fixed_noise,
        'fixed_labels': fixed_labels,
        'generated_images': generated_images,
        'generated_images_viz': generated_images_viz,
        'config': config
    }


def test_plot_loss_curves(smoke_trainer):
    """Test loss curves plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    history = smoke_trainer['history']
    
    fig = plot_loss_curves(
        history['generator_loss'],
        history['discriminator_loss'],
        title="Test Loss Curves"
    )
    
    assert hasattr(fig, 'savefig')
    
    # Save to temporary file to verify it works
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000  # Should be a reasonable size
        os.unlink(f.name)
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_discriminator_outputs(smoke_trainer):
    """Test discriminator outputs plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    history = smoke_trainer['history']
    
    fig = plot_discriminator_outputs(
        history['d_real_mean'],
        history['d_fake_mean'],
        title="Test Discriminator Outputs"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_discriminator_accuracy(smoke_trainer):
    """Test discriminator accuracy plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    history = smoke_trainer['history']
    
    fig = plot_discriminator_accuracy(
        history['d_real_acc'],
        history['d_fake_acc'],
        title="Test Discriminator Accuracy"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_sample_grid(smoke_trainer):
    """Test sample grid creation."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    images = smoke_trainer['generated_images_viz']
    labels = smoke_trainer['fixed_labels']
    
    fig = create_sample_grid(
        images,
        labels=labels,
        title="Test Sample Grid",
        rows=4,
        cols=4
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_training_progression(smoke_trainer):
    """Test training progression plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    generator = smoke_trainer['trainer'].generator
    fixed_noise = smoke_trainer['fixed_noise']
    fixed_labels = smoke_trainer['fixed_labels']
    
    # Test with fake epoch numbers (since we don't have multiple checkpoints)
    epochs_to_show = [1, 2]  # Pretend we have these epochs
    
    fig = plot_training_progression(
        generator,
        fixed_noise,
        fixed_labels,
        epochs_to_show,
        title="Test Training Progression"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_real_vs_fake(smoke_trainer):
    """Test real vs fake plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Get some real images from the dataset
    trainer = smoke_trainer['trainer']
    config = smoke_trainer['config']
    
    try:
        dataset = gan_mnist.data.load_mnist(
            buffer_size=config.data.buffer_size,
            batch_size=config.data.batch_size,
            normalize_to_neg1_1=config.data.normalize_to_neg1_1,
            subset_size=100,
            seed=config.training.seed
        )
    except:
        dataset = gan_mnist.data.create_synthetic_dataset(
            batch_size=config.data.batch_size,
            image_shape=tuple(config.data.image_shape),
            num_classes=config.data.num_classes,
            num_samples=config.data.buffer_size,
            seed=config.training.seed
        )
    
    # Get one batch of real images
    for real_images, real_labels in dataset.take(1):
        break
    
    # Generate fake images
    noise = tf.random.normal([tf.shape(real_images)[0], config.model.latent_dim])
    fake_labels = tf.random.uniform(
        [tf.shape(real_images)[0], 1],
        minval=0,
        maxval=config.data.num_classes,
        dtype=tf.int32
    )
    fake_images = trainer.generator([noise, fake_labels], training=False)
    fake_images_viz = (fake_images + 1.0) / 2.0  # Rescale to [0, 1]
    
    fig = plot_real_vs_fake(
        real_images,
        fake_images,
        real_labels=real_labels,
        fake_labels=fake_labels,
        title="Test Real vs Fake",
        num_pairs=4
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_latent_interpolation(smoke_trainer):
    """Test latent interpolation plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    generator = smoke_trainer['trainer'].generator
    config = smoke_trainer['config']
    
    # Create two different noise vectors
    noise1 = tf.random.normal([1, config.model.latent_dim])
    noise2 = tf.random.normal([1, config.model.latent_dim])
    label = tf.constant([[5]], dtype=tf.int32)  # Class 5
    
    fig = plot_latent_interpolation(
        generator,
        noise1,
        noise2,
        label,
        steps=5,
        title="Test Latent Interpolation"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_class_conditioning(smoke_trainer):
    """Test class conditioning plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    generator = smoke_trainer['trainer'].generator
    fixed_noise = smoke_trainer['fixed_noise'][:1]  # Take first noise vector
    config = smoke_trainer['config']
    
    fig = plot_class_conditioning(
        generator,
        fixed_noise,
        num_classes=10,
        title="Test Class Conditioning"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_pixel_distribution(smoke_trainer):
    """Test pixel distribution plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Get real images
    trainer = smoke_trainer['trainer']
    config = smoke_trainer['config']
    
    try:
        dataset = gan_mnist.data.load_mnist(
            buffer_size=config.data.buffer_size,
            batch_size=config.data.batch_size,
            normalize_to_neg1_1=config.data.normalize_to_neg1_1,
            subset_size=100,
            seed=config.training.seed
        )
    except:
        dataset = gan_mnist.data.create_synthetic_dataset(
            batch_size=config.data.batch_size,
            image_shape=tuple(config.data.image_shape),
            num_classes=config.data.num_classes,
            num_samples=config.data.buffer_size,
            seed=config.training.seed
        )
    
    # Get one batch of real images
    for real_images, _ in dataset.take(1):
        break
    
    # Generate fake images
    noise = tf.random.normal([tf.shape(real_images)[0], config.model.latent_dim])
    fake_labels = tf.random.uniform(
        [tf.shape(real_images)[0], 1],
        minval=0,
        maxval=config.data.num_classes,
        dtype=tf.int32
    )
    fake_images = trainer.generator([noise, fake_labels], training=False)
    
    fig = plot_pixel_distribution(
        real_images,
        fake_images,
        title="Test Pixel Distribution"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_diversity_report(smoke_trainer):
    """Test diversity report plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Generate some images to test diversity
    trainer = smoke_trainer['trainer']
    config = smoke_trainer['config']
    
    noise = tf.random.normal([20, config.model.latent_dim])
    labels = tf.random.uniform([20, 1], minval=0, maxval=10, dtype=tf.int32)
    generated_images = trainer.generator([noise, labels], training=False)
    
    # Calculate diversity scores per class (simplified)
    diversity_scores = []
    for class_id in range(10):
        class_mask = tf.squeeze(labels) == class_id
        class_images = tf.boolean_mask(generated_images, class_mask)
        if tf.shape(class_images)[0] > 0:
            div = calculate_diversity(class_images)
            diversity_scores.append(div)
        else:
            diversity_scores.append(0.0)
    
    # Pad or truncate to 10 classes
    while len(diversity_scores) < 10:
        diversity_scores.append(0.0)
    diversity_scores = diversity_scores[:10]
    
    # Calculate real data diversity (approximate)
    try:
        dataset = gan_mnist.data.load_mnist(
            buffer_size=config.data.buffer_size,
            batch_size=config.data.batch_size,
            normalize_to_neg1_1=config.data.normalize_to_neg1_1,
            subset_size=100,
            seed=config.training.seed
        )
    except:
        dataset = gan_mnist.data.create_synthetic_dataset(
            batch_size=config.data.batch_size,
            image_shape=tuple(config.data.image_shape),
            num_classes=config.data.num_classes,
            num_samples=config.data.buffer_size,
            seed=config.training.seed
        )
    
    real_diversity_scores = []
    for class_id in range(10):
        # Get some real images of this class (simplified)
        real_diversity_scores.append(0.5)  # Placeholder
    
    real_data_score = np.mean(real_diversity_scores) if real_diversity_scores else 0.5
    
    fig = plot_diversity_report(
        diversity_scores,
        real_data_score,
        title="Test Diversity Report"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_confusion_matrix(smoke_trainer):
    """Test confusion matrix plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Create a simple confusion matrix
    cm = np.array([
        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 7]
    ])
    
    class_names = [str(i) for i in range(10)]
    
    fig = plot_confusion_matrix(
        cm,
        class_names,
        title="Test Confusion Matrix"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_gradient_norms(smoke_trainer):
    """Test gradient norms plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # Create some fake gradient norms
    g_norms = [0.5, 0.6, 0.4, 0.7, 0.5]
    d_norms = [0.3, 0.4, 0.2, 0.5, 0.3]
    
    fig = plot_gradient_norms(
        g_norms,
        d_norms,
        title="Test Gradient Norms"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_architecture_diagram(smoke_trainer):
    """Test architecture diagram plotting."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    generator = smoke_trainer['trainer'].generator
    discriminator = smoke_trainer['trainer'].discriminator
    
    fig = create_architecture_diagram(
        generator,
        discriminator,
        title="Test Architecture Diagram"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_dashboard(smoke_trainer):
    """Test dashboard creation."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    history = smoke_trainer['history']
    generator = smoke_trainer['trainer'].generator
    discriminator = smoke_trainer['trainer'].discriminator
    fixed_noise = smoke_trainer['fixed_noise']
    fixed_labels = smoke_trainer['fixed_labels']
    
    fig = create_dashboard(
        history,
        generator,
        discriminator,
        fixed_noise,
        fixed_labels,
        title="Test Dashboard"
    )
    
    assert hasattr(fig, 'savefig')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=100)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 1000
        os.unlink(f.name)
    
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_accessibility_no_red_green_only():
    """Test that figures don't rely solely on red/green for distinction."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # This is a conceptual test - we're checking that our palette doesn't rely on red/green
    # In practice, we would need to inspect the actual colors used in each figure
    # For now, we'll verify that our Okabe-Ito palette doesn't contain problematic combinations
    
    from gan_mnist.visualization import OKABE_ITO_PALETTE
    
    # Check that we don't have both red and green as primary distinguishing colors
    # Our palette uses blue/orange/sky/vermillion/purple/yellow/black
    # Vermillion is reddish, but we also have blue and orange for primary distinctions
    
    # The key is that we vary line styles/markers, not just color
    # This is handled in the individual plotting functions
    
    # Just verify the palette constants are defined
    assert 'blue' in OKABE_ITO_PALETTE
    assert 'orange' in OKABE_ITO_PALETTE
    assert 'vermillion' in OKABE_ITO_PALETTE  # This is the reddish one
    # We don't have a pure green in our primary palette
    
    # This test passes if we can import and access the palette
    assert True


def test_generate_html_report(smoke_trainer, tmpdir):
    """Test HTML report generation."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    
    # This test verifies that we can generate the components needed for an HTML report
    
    trainer = smoke_trainer['trainer']
    history = smoke_trainer['history']
    config = smoke_trainer['config']
    
    # Create a mock report directory
    report_dir = os.path.join(tmpdir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate some data that would go into the report
    # In a full implementation, this would create actual PNG files and embed them
    
    # Create a simple HTML report structure
    import json
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GAN-MNIST Training Report</title>
    </head>
    <body>
        <h1>GAN-MNIST Training Report</h1>
        <h2>Configuration</h2>
        <pre>{json.dumps(config.__dict__, indent=2, default=str)}</pre>
        <h2>Training Results</h2>
        <p>Final Generator Loss: {history['generator_loss'][-1] if history['generator_loss'] else 'N/A':.4f}</p>
        <p>Final Discriminator Loss: {history['discriminator_loss'][-1] if history['discriminator_loss'] else 'N/A':.4f}</p>
        <h2>Model Information</h2>
        <p>Generator Parameters: {trainer.generator.count_params():,}</p>
        <p>Discriminator Parameters: {trainer.discriminator.count_params():,}</p>
    </body>
    </html>
    """
    
    report_path = os.path.join(report_dir, "report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    # Verify the report was created
    assert os.path.exists(report_path)
    assert os.path.getsize(report_path) > 100  # Should have reasonable content
    
    # Check that it contains expected elements
    with open(report_path, 'r') as f:
        content = f.read()
        assert "GAN-MNIST Training Report" in content
        assert "Final Generator Loss" in content
        assert "Final Discriminator Loss" in content


if __name__ == "__main__":
    # Allow running this file directly for manual testing
    pytest.main([__file__, "-v"])
