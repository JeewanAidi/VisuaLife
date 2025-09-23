import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from visualife.core.activations import ReLU, LeakyReLU, ParametricReLU, ELU, Swish, Sigmoid, Tanh

def test_relu():
    print("Testing ReLU...")
    relu = ReLU()
    X = np.array([[-1, 2, -3], [4, -5, 6]])
    
    # Forward pass
    output = relu.forward(X)
    expected = np.array([[0, 2, 0], [4, 0, 6]])
    assert np.allclose(output, expected), "ReLU forward failed!"
    
    # Backward pass
    dZ = np.ones_like(X)
    dX = relu.backward(dZ)
    expected_dX = np.array([[0, 1, 0], [1, 0, 1]])
    assert np.allclose(dX, expected_dX), "ReLU backward failed!"
    print("ReLU test passed!\n")

def test_leaky_relu():
    print("Testing LeakyReLU...")
    leaky_relu = LeakyReLU(alpha=0.1)
    X = np.array([[-1, 2, -3], [4, -5, 6]])
    
    # Forward pass
    output = leaky_relu.forward(X)
    expected = np.array([[-0.1, 2, -0.3], [4, -0.5, 6]])
    assert np.allclose(output, expected), "LeakyReLU forward failed!"
    
    # Backward pass
    dZ = np.ones_like(X)
    dX = leaky_relu.backward(dZ)
    expected_dX = np.array([[0.1, 1, 0.1], [1, 0.1, 1]])
    assert np.allclose(dX, expected_dX), "LeakyReLU backward failed!"
    print("LeakyReLU test passed!\n")

def test_elu():
    print("Testing ELU...")
    elu = ELU(alpha=1.0)
    X = np.array([[-1, 0, 1], [-2, 2, -0.5]])
    
    # Forward pass
    output = elu.forward(X)
    expected_neg = 1.0 * (np.exp(-1) - 1)  # ELU(-1)
    assert np.allclose(output[0, 0], expected_neg), "ELU forward failed for negative values!"
    assert output[0, 1] == 0, "ELU forward failed for zero!"
    assert output[0, 2] == 1, "ELU forward failed for positive values!"
    
    print("ELU test passed!\n")

def test_swish():
    print("Testing Swish...")
    swish = Swish()
    X = np.array([[0, 1, -1]])
    
    # Forward pass
    output = swish.forward(X)
    # Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert np.allclose(output[0, 0], 0), "Swish(0) should be 0!"
    
    print("Swish test passed!\n")

def test_all_activations():
    print("🧪 Testing all activation shapes...")
    X = np.random.randn(3, 4)  # Batch of 3, 4 features each
    
    activations = {
        'ReLU': ReLU(),
        'LeakyReLU': LeakyReLU(),
        'ParametricReLU': ParametricReLU(),
        'ELU': ELU(),
        'Swish': Swish(),
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh()
    }
    
    for name, activation in activations.items():
        # Test forward pass
        output = activation.forward(X)
        assert output.shape == X.shape, f"{name} forward shape mismatch!"
        
        # Test backward pass
        dZ = np.ones_like(X)
        dX = activation.backward(dZ)
        assert dX.shape == X.shape, f"{name} backward shape mismatch!"
        
        print(f"{name} shape test passed!")
    
    print("All activation shape tests passed!\n")

if __name__ == "__main__":
    print("Testing Advanced Activation Functions")
    print("========================================")
    test_relu()
    test_leaky_relu()
    test_elu()
    test_swish()
    test_all_activations()
    print("All activation tests completed successfully!")