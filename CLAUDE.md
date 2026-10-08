# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📋 Project Overview

This repository contains a refactored, production-grade implementation of a Conditional Generative Adversarial Network (cGAN) for MNIST digit generation. The project has been restructured from a single Jupyter notebook into a modular Python package with proper testing, configuration, and deployment capabilities.

## 🏗️ Project Structure

```
.
├── gan-mnist/                    # Package root
│   ├── pyproject.toml            # Project configuration and dependencies
│   ├── Makefile                  # Development commands
│   ├── README.md                 # This file
│   ├── CLAUDE.md                 # This file
│   ├── LICENSE                   # MIT license
│   ├── .gitignore                # Git ignore rules
│   ├── .pre-commit-config.yaml   # Pre-commit hooks
│   ├── requirements.txt          # Runtime dependencies
│   ├── requirements-dev.txt      # Development dependencies
│   ├── configs/                  # Configuration files
│   │   ├── default.yaml          # Full training config
│   │   └── smoke.yaml            # Quick test config
│   ├── docs/                     # Documentation
│   │   ├── ARCHITECTURE.md       # System architecture
│   │   ├── THEORY.md             # Mathematical background
│   │   ├── DECISIONS.md          # Design decisions
│   │   └── TESTING.md            # Testing guide
│   ├── notebooks/                # Original notebook (preserved)
│   │   └── GAN over MNIST.ipynb  # Original implementation
│   ├── src/                      # Source code
│   │   └── gan_mnist/            # Main package
│   │       ├── __init__.py       # Package exports
│   │       ├── __main__.py       # Entry point for `python -m gan_mnist`
│   │       ├── cli.py            # Command-line interface
│   │       ├── config.py         # Configuration management
│   │       ├── utils.py          # Utility functions
│   │       ├── data.py           # Data loading and preprocessing
│   │       ├── models/           # Neural network models
│   │       │   ├── __init__.py
│   │       │   ├── generator.py  # Generator model
│   │       │   └── discriminator.py # Discriminator model
│   │       ├── losses.py         # Loss functions
│   │       ├── trainer.py        # Training loop
│   │       ├── callbacks.py      # Training callbacks
│   │       ├── evaluation.py     # Evaluation metrics
│   │       └── visualization.py  # Plotting functions
│   ├── tests/                    # Test suite
│   │   ├── conftest.py           # Test fixtures
│   │   ├── test_config.py        # Configuration tests
│   │   ├── test_utils.py         # Utility tests
│   │   ├── test_data.py          # Data loading tests
│   │   ├── test_losses.py        # Loss function tests
│   │   ├── test_models.py        # Model tests
│   │   ├── test_trainer.py       # Trainer tests
│   │   ├── test_evaluation.py    # Evaluation tests
│   │   ├── test_callbacks.py     # Callback tests
│   │   └── test_visual_report.py # Visualization and report tests
│   └── outputs/                  # Generated outputs (gitignored)
│       ├── checkpoints/          # Model checkpoints
│   │   ├── samples/              # Generated image samples
│   │   ├── figures/              # Plots and visualizations
│   │   ├── logs/                 # Training logs
│   │   └── reports/              # Reports and evaluations
```

## 🔧 Development Workflow

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd gan-mnist

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### Configuration

The project uses YAML configuration files located in the `configs/` directory:

- `configs/default.yaml`: Full training configuration (300 epochs, original hyperparameters)
- `configs/smoke.yaml`: Quick test configuration (2 epochs, small subset)

### Commands

#### Makefile Commands

```bash
# Install dependencies
make install

# Run code formatting
make format

# Run linting
make lint

# Run type checking
make typecheck

# Run tests
make test

# Run fast tests (skip slow tests)
make test-fast

# Run smoke training
make smoke

# Generate report from training run
make report

# Clean generated files
make clean

# Do everything: lint + typecheck + test
make all
```

#### CLI Commands

```bash
# Train the model
python -m gan_mnist train --config configs/default.yml
python -m gan_mnist train --config configs/smoke.yml --smoke  # Quick test

# Generate images from a checkpoint
python -m gan_mnist generate --checkpoint outputs/checkpoints/checkpoint_epoch_0100 --label 7 --count 16 --out sample_7.png

# Evaluate a model
python -m gan_mnist evaluate --checkpoint outputs/checkpoints/checkpoint_epoch_0100

# Generate report from training run
python -m gan_mnist report --run-dir outputs/runs/20230101-120000/
```

### Training Process

The training process follows the original GAN architecture with these key features:

1. **Conditional GAN**: Both generator and discriminator receive class labels as input
2. **Label Smoothing**: Real labels smoothed to 0.9 to prevent discriminator overconfidence
3. **LeakyReLU Activations**: Used throughout with alpha=0.2
4. **Batch Normalization**: Applied in both networks (except output layers)
5. **Dropout**: 0.3 rate in discriminator for regularization
6. **Learning Rate Ratio**: Generator learns faster (2e-4) than discriminator (1e-5)

### Testing

The project includes a comprehensive test suite:

- **Unit tests**: For all modules (config, utils, data, models, losses, trainer, etc.)
- **Integration tests**: Testing the full training pipeline with smoke configuration
- **Visualization tests**: Ensuring all 14 required plots can be generated
- **Coverage**: Target of ≥85% coverage on source code

Run tests with:
```bash
# Full test suite
make test

# Fast tests only (skip slow tests)
make test-fast

# With coverage report
make test  # Generates HTML report in outputs/reports/coverage/
```

### Code Quality

The project maintains high code quality through:

- **Type hints**: Full type annotations on public functions and classes
- **Code formatting**: Black enforcement via pre-commit and make format
- **Linting**: Ruff enforcement via pre-commit and make lint
- **Type checking**: MyPy enforcement via make typecheck
- **Documentation**: Google-style docstrings with Args/Returns/Sections

## 🧠 Architecture Overview

### Core Components

1. **Configuration System** (`config.py`)
   - Hierarchical dataclasses for data, model, training, and paths configuration
   - YAML loading/saving with validation
   - Environment-independent defaults

2. **Data Pipeline** (`data.py`)
   - MNIST loading with optional subset for fast testing
   - Synthetic data fallback for offline testing
   - Proper normalization to [-1, 1] range

3. **Models** (`models/`)
   - **Generator**: Maps noise + label → 28x28x1 image
   - **Discriminator**: Maps image + label → real/fake logit
   - Both use embedding layers for label processing
   - Architectures match the original notebook exactly

4. **Loss Functions** (`losses.py`)
   - Discriminator loss with label smoothing (0.9)
   - Generator loss (non-saturating version)
   - Gradient penalty function (optional)

5. **Training System** (`trainer.py`)
   - `GANTrainer` class encapsulating the training loop
   - `@tf.function` decorated `train_step` for performance
   - Fixed seed noise/label generation for consistent sampling
   - Checkpoint saving/loading
   - History tracking of losses and metrics

6. **Callbacks** (`callbacks.py`)
   - Sample generation during training
   - Loss history recording
   - Periodic checkpoint saving

7. **Evaluation** (`evaluation.py`)
   - Diversity metrics (average pairwise L2 distance)
   - Label conditional accuracy (via auxiliary classifier)
   - Discriminator output analysis
   - Inception score placeholder

8. **Visualization** (`visualization.py`)
   - All 14 required plots as pure functions returning matplotlib Figures
   - Accessible color schemes (Okabe-Ito palette)
   - Consistent styling and formatting
   - Dashboard creation function

9. **Command Line Interface** (`cli.py`)
   - Four main commands: train, generate, evaluate, report
   - Configuration override capabilities
   - Timestamped output directories
   - Integrated report generation

## ⚙️ Hyperparameters

The implementation preserves the exact hyperparameters from the original notebook:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Latent dimension | 100 | Size of noise vector |
| Label embedding dim | 50 | Embedding size for class labels |
| Generator base filters | [256, 128, 64] | Internal filter sizes |
| Discriminator base filters | [64, 128] | Internal filter sizes |
| Generator upsample filters | [128, 64, 1] | Upsampling filter sizes |
| Discriminator dropout | 0.3 | Dropout rate in discriminator |
| LeakyReLU alpha | 0.2 | Negative slope coefficient |
| Batch norm momentum | 0.9 | Batch normalization momentum |
| Generator LR | 2e-4 | Adam learning rate |
| Discriminator LR | 1e-5 | Adam learning rate (5× slower) |
| Adam β₁ | 0.5 | Adam beta_1 parameter |
| Label smoothing | 0.9 | Real label smoothing factor |
| Batch size | 256 | Training batch size |
| Buffer size | 60000 | Shuffle buffer size |
| Epochs | 300 | Training epochs (full) |
| Image shape | 28×28×1 | MNIST image dimensions |
| Num classes | 10 | Digits 0-9 |

## 📊 Key Features

### Reproducibility
- Seeding helper for Python, NumPy, and TensorFlow
- Deterministic results with fixed seeds
- Configurable seed via CLI and configuration files

### Modularity
- Clear separation of concerns
- Dependency injection through configuration
- Easy to extend or modify components

### Testability
- Pure functions where possible
- Mock-friendly interfaces
- Comprehensive test suite
- Synthetic data fallback for offline testing

### Accessibility
- Colorblind-safe palettes in all visualizations
- Multiple visual indicators (line styles, markers, shapes)
- No reliance on color alone for information conveyance

### Performance
- TensorFlow graph execution via `@tf.function`
- Efficient data loading with tf.data.Dataset
- Minimal memory overhead

## 🔄 Running the Project

### Quick Start (Smoke Test)

```bash
# Install in development mode
pip install -e ".[dev]"

# Run smoke test (2 epochs, small subset)
make smoke
# or
python -m gan_mnist train --config configs/smoke.yml

# Generate report
make report
```

### Full Training

```bash
# Full training (300 epochs)
make train
# or
python -m gan_mnist train --config configs/default.yml

# This will take significant time depending on hardware
```

### Using Generated Models

```bash
# Generate images of digit 3
python -m gan_mnist generate \
    --checkpoint outputs/runs/20230101-120000/checkpoints/checkpoint_epoch_0300 \
    --label 3 \
    --count 16 \
    --out outputs/figures/digit_3_samples.png

# Evaluate model performance
python -m gan_mnist evaluate \
    --checkpoint outputs/runs/20230101-120000/checkpoints/checkpoint_epoch_0300
```

## 📚 Documentation

- **README.md**: This file - quick start and overview
- **docs/ARCHITECTURE.md**: Detailed system architecture with Mermaid diagrams
- **docs/THEORY.md**: Mathematical foundations of GANs and cGANs
- **docs/DECISIONS.md**: Record of all design decisions and trade-offs
- **docs/TESTING.md**: Guide to running and interpreting tests
- **docs/PRESENTATION_NOTES.md**: Talking points for project presentation

## 🛠️ Development Guidelines

### Adding New Features
1. Add configuration options to config.py dataclasses
2. Implement core functionality in appropriate src/ modules
3. Add unit tests in tests/ directory
4. Update documentation as needed
5. Run `make all` to verify code quality

### Making Changes
1. Run `make format` to format code
2. Run `make lint` to check for style issues
3. Run `make typecheck` to verify type safety
4. Run `make test` to ensure nothing is broken
5. Update changelog/documentation as needed

### Performance Considerations
- Use `@tf.function` for training steps
- Minimize Python/TensorFlow boundary crossings
- Batch operations where possible
- Profile TensorFlow operations when optimizing

## 🐣 Getting Started

1. **Install**: `pip install -e ".[dev]"`
2. **Test**: `make test-fast` (quick verification)
3. **Experiment**: `make smoke` (quick training run)
4. **Explore**: Try the CLI commands with different options
5. **Read**: Check docs/ for deeper understanding

The project is now ready for experimentation, extension, and presentation!
