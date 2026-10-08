"""
Tests for utility functions.
"""

import pytest
import numpy as np
import sys
import os
import tempfile

# Try to import tensorflow, but don't fail if it's not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from gan_mnist.utils import (
    setup_logging, 
    set_seed, 
    get_device_info, 
    create_directories,
    format_tensor_shape,
    safe_divide
)


def test_setup_logging():
    """Test logging setup."""
    # This should not raise an exception
    setup_logging()
    setup_logging(level=0)  # DEBUG
    setup_logging(level=50)  # CRITICAL


def test_set_seed():
    """Test setting random seeds."""
    # Set seed and check that we get reproducible results
    set_seed(42)
    a1 = np.random.rand()
    b1 = tf.random.uniform([1]).numpy() if TF_AVAILABLE else 0  # If TF not available, we skip this part
    
    set_seed(42)
    a2 = np.random.rand()
    b2 = tf.random.uniform([1]).numpy() if TF_AVAILABLE else 0
    
    assert a1 == a2
    if TF_AVAILABLE:
        assert b1 == b2
    
    # Different seed should give different results
    set_seed(123)
    a3 = np.random.rand()
    assert a1 != a3


@pytest.mark.tensorflow
def test_get_device_info():
    """Test getting device information."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    device_info = get_device_info()
    assert isinstance(device_info, str)
    assert len(device_info) > 0
    # Should contain either GPU info or CPU message
    assert ("GPU:" in device_info) or ("CPU:" in device_info)


def test_create_directories():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dirs = [
            os.path.join(tmpdir, "dir1"),
            os.path.join(tmpdir, "dir2", "subdir"),
            os.path.join(tmpdir, "dir3")
        ]
        
        # This should not raise an exception
        create_directories(*test_dirs)
        
        # Check that directories were created
        for test_dir in test_dirs:
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)


def test_format_tensor_shape():
    """Test tensor shape formatting."""
    assert format_tensor_shape((28, 28, 1)) == "[28, 28, 1]"
    assert format_tensor_shape((100,)) == "[100]"
    assert format_tensor_shape((10, 20)) == "[10, 20]"
    assert format_tensor_shape(()) == "[]"


def test_safe_divide():
    """Test safe division function."""
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(1, 4) == 0.25
    assert safe_divide(0, 5) == 0.0
    
    # Test default value when dividing by zero
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, 0, default=-1) == -1.0
    assert safe_divide(10, 0, default=float('inf')) == float('inf')
    
    # Test with floats
    assert safe_divide(3.0, 2.0) == 1.5
