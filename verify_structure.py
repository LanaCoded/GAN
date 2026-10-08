#!/usr/bin/env python3
"""
Verification script for GAN-MNIST structure.
Tests that the package can be imported and basic components work.
"""

import sys
import os

def test_basic_imports():
    """Test that basic imports work without tensorflow."""
    print("Testing basic imports...")
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.insert(0, src_path)
    
    # Test config imports
    from gan_mnist.config import Config, DataConfig, ModelConfig, TrainConfig, PathsConfig
    print("✓ Config imports successful")
    
    # Test config instantiation
    config = Config()
    assert config.training.epochs == 300
    assert config.model.latent_dim == 100
    assert config.data.batch_size == 256
    print("✓ Config instantiation and validation successful")
    
    # Test utils imports
    from gan_mnist.utils import set_seed, create_directories, format_tensor_shape, safe_divide
    print("✓ Utils imports successful")
    
    # Test utils functions
    set_seed(42)
    assert format_tensor_shape((28, 28, 1)) == "[28, 28, 1]"
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(10, 0) == 0.0
    print("✓ Utils functions working")
    
    # Test directory creation
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "test", "subdir")
        create_directories(test_dir)
        assert os.path.exists(test_dir)
    print("✓ Directory creation working")
    
    print("\n✅ All basic functionality tests passed!")
    return True

def test_lazy_loading():
    """Test that lazy loading works correctly."""
    print("\nTesting lazy loading...")
    
    import gan_mnist
    
    # Test that we can access non-tensorflow attributes
    assert hasattr(gan_mnist, 'Config')
    assert hasattr(gan_mnist, 'DataConfig')
    print("✓ Non-tensorflow attributes accessible")
    
    # Test that tensorflow-dependent attributes raise appropriate errors
    tensorflow_attrs = ['GANTrainer', 'create_trainer_from_config', 'main']
    for attr in tensorflow_attrs:
        try:
            getattr(gan_mnist, attr)
            print(f"✗ ERROR: Should not be able to access {attr} without tensorflow")
            return False
        except AttributeError as e:
            if 'TensorFlow not available' in str(e):
                print(f"✓ {attr} correctly raises TensorFlow not available error")
            else:
                print(f"✗ ERROR: Unexpected error for {attr}: {e}")
                return False
    
    print("✓ Lazy loading working correctly")
    return True

def test_file_structure():
    """Test that the expected file structure exists."""
    print("\nTesting file structure...")
    
    expected_files = [
        'pyproject.toml',
        'README.md',
        'CLAUDE.md',
        'LICENSE',
        '.gitignore',
        'requirements.txt',
        'requirements-dev.txt',
        'src/gan_mnist/__init__.py',
        'src/gan_mnist/config.py',
        'src/gan_mnist/utils.py',
        'src/gan_mnist/data.py',
        'src/gan_mnist/models/__init__.py',
        'src/gan_mnist/models/generator.py',
        'src/gan_mnist/models/discriminator.py',
        'src/gan_mnist/losses.py',
        'src/gan_mnist/trainer.py',
        'src/gan_mnist/callbacks.py',
        'src/gan_mnist/evaluation.py',
        'src/gan_mnist/visualization.py',
        'src/gan_mnist/cli.py',
        'tests/conftest.py',
        'tests/test_config.py',
        'tests/test_utils.py',
        'configs/default.yaml',
        'configs/smoke.yaml'
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    else:
        print("✓ All expected files present")
        return True

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("GAN-MNIST Structure Verification")
    print("=" * 60)
    
    success = True
    success &= test_basic_imports()
    success &= test_lazy_loading()
    success &= test_file_structure()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL VERIFICATION TESTS PASSED!")
        print("The GAN-MNIST package structure is correct and functional.")
    else:
        print("❌ SOME VERIFICATION TESTS FAILED!")
        print("Please check the errors above.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
