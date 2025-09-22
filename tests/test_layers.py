import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.core.layers import Dense

def test_dense_forward():
    """Test the forward pass of the Dense layer."""
    print("Testing Dense layer forward pass...")
    
    # Create a Dense layer with known weights and biases for predictable output
    dense_layer = Dense(input_size=2, output_size=3)
    dense_layer.weights = np.array([[1.0, 2.0, 3.0],  # Force known values for testing
                                   [0.5, 1.5, 1]])
    dense_layer.biases = np.array([[0.1, 0.5, 0.3]])
    
    # Create test input
    X = np.array([[4.0, 3.0],  # Batch of 2 samples
                  [5.0, 6.0],
                  [1.0, 7.0]])
    
    # Perform forward pass
    output = dense_layer.forward(X)
    
    # Manually calculate expected output: Z = X.W + b
    expected_output = np.array([
        [4.0*1.0 + 3.0*0.5 + 0.1, 4.0*2.0 + 3.0*1.5 + 0.5, 4.0*3.0 + 3.0*1 + 0.3],
        [5.0*1.0 + 6.0*0.5 + 0.1, 5.0*2.0 + 6.0*1.5 + 0.5, 5.0*3.0 + 6.0*1 + 0.3],
        [1.0*1.0 + 7.0*0.5 + 0.1, 1.0*2.0 + 7.0*1.5 + 0.5, 1.0*3.0 + 7.0*1 + 0.3]
    ])
    
    print("Input X:\n", X)
    print("Weights W:\n", dense_layer.weights)
    print("Biases b:\n", dense_layer.biases)
    print("Our Output Z:\n", output)
    print("Expected Output Z:\n", expected_output)
    
    assert np.allclose(output, expected_output), "Dense layer forward pass failed!"
    print("Dense layer forward pass test passed!\n")

def test_dense_backward():
    """Test the backward pass of the Dense layer."""
    print("Testing Dense layer backward pass...")
    
    # Initialize a layer
    dense_layer = Dense(2, 3)
    dense_layer.weights = np.array([[1.0, 2.0, 1.0],
                                   [0.5, 1.0, 0.5]])
    dense_layer.biases = np.array([[0.1, 0.2, 0.3]])
    
    # Set a fixed input (will be stored during forward pass)
    X = np.array([[1.0, 2.0]])
    _ = dense_layer.forward(X)  # Perform forward pass to store input
    
    # Mock gradient from next layer
    dZ = np.array([[0.5, 1.0, 0.5]])
    learning_rate = 0.1
    
    # Perform backward pass
    dX = dense_layer.backward(dZ, learning_rate)
    
    # Manually calculate expected gradients
    # dW = X.T . dZ
    expected_dweights = np.array([[1.0 * 0.5, 1.0 * 1.0, 1.0 * 0.5],
                                 [2.0 * 0.5, 2.0 * 1.0, 2.0 * 0.5]])
    # db = sum(dZ, axis=0)
    expected_dbiases = np.array([[0.5, 1.0, 0.5]])
    # dX = dZ . W.T
    expected_dX = np.array([[0.5*1.0 + 1.0*2.0 + 0.5*1.0, 
                            0.5*0.5 + 1.0*1.0 + 0.5*0.5]])
    
    print("Input gradient dZ:\n", dZ)
    print("Our calculated dW:\n", dense_layer.dweights)
    print("Expected dW:\n", expected_dweights)
    print("Our calculated db:\n", dense_layer.dbiases)
    print("Expected db:\n", expected_dbiases)
    print("Our calculated dX:\n", dX)
    print("Expected dX:\n", expected_dX)
    
    assert np.allclose(dense_layer.dweights, expected_dweights), "Weight gradients incorrect!"
    assert np.allclose(dense_layer.dbiases, expected_dbiases), "Bias gradients incorrect!"
    assert np.allclose(dX, expected_dX), "Input gradients incorrect!"
    print("Dense layer backward pass test passed!\n")

if __name__ == "__main__":
    print("Running VisuaLife Layers Test Suite")
    print("======================================")
    test_dense_forward()
    test_dense_backward()
    print("All Dense layer tests passed! Ready for learning.")