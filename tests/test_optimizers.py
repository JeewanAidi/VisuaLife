import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualife.core.optimizers import SGD, Momentum, Adam
from visualife.core.layers import Dense

def test_sgd_optimizer():
    """Test SGD optimizer weight updates"""
    print("🧪 Testing SGD Optimizer...")
    
    # Create a simple layer with known gradients
    layer = Dense(3, 2)
    layer.weights = np.ones((3, 2)) * 0.5  # Set known weights
    layer.biases = np.ones((1, 2)) * 0.2   # Set known biases
    layer.dweights = np.ones((3, 2)) * 0.1  # Set gradients
    layer.dbiases = np.ones((1, 2)) * 0.05
    
    # Create SGD optimizer
    sgd = SGD(learning_rate=0.1)
    
    # Store original values
    original_weights = layer.weights.copy()
    original_biases = layer.biases.copy()
    
    # Update weights
    sgd.update(layer)
    
    # Check if weights were updated correctly
    expected_weights = original_weights - 0.1 * layer.dweights
    expected_biases = original_biases - 0.1 * layer.dbiases
    
    assert np.allclose(layer.weights, expected_weights), "SGD weight update incorrect"
    assert np.allclose(layer.biases, expected_biases), "SGD bias update incorrect"
    print("✅ SGD optimizer test passed!")

def test_momentum_optimizer():
    """Test Momentum optimizer with velocity"""
    print("🧪 Testing Momentum Optimizer...")
    
    layer = Dense(2, 1)
    layer.weights = np.array([[0.5], [0.3]])
    layer.dweights = np.array([[0.1], [0.2]])
    
    momentum = Momentum(learning_rate=0.1, momentum=0.9)
    
    # First update
    momentum.update(layer)
    
    # Velocity should be: v = -lr * gradient = -0.1 * [0.1, 0.2]
    expected_velocity = -0.1 * np.array([[0.1], [0.2]])
    expected_weights = np.array([[0.5], [0.3]]) + expected_velocity
    
    assert np.allclose(layer.weights, expected_weights), "Momentum first update incorrect"
    
    # Second update with same gradient
    layer.dweights = np.array([[0.1], [0.2]])  # Same gradient
    momentum.update(layer)
    
    # Velocity should be: v = 0.9 * previous_v - lr * gradient
    expected_velocity = 0.9 * expected_velocity - 0.1 * np.array([[0.1], [0.2]])
    expected_weights += expected_velocity
    
    assert np.allclose(layer.weights, expected_weights), "Momentum second update incorrect"
    print("✅ Momentum optimizer test passed!")

def test_adam_optimizer():
    """Test Adam optimizer with moment estimates"""
    print("🧪 Testing Adam Optimizer...")
    
    layer = Dense(2, 1)
    layer.weights = np.array([[0.5], [0.3]])
    layer.dweights = np.array([[0.1], [0.2]])
    
    adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
    
    # Store original weights
    original_weights = layer.weights.copy()
    
    # First update
    adam.update(layer)
    
    # Weights should change (decrease due to negative gradient)
    assert not np.allclose(layer.weights, original_weights), "Adam should update weights"
    
    # Check that moments are being tracked
    layer_id = id(layer)
    assert layer_id in adam.m, "Adam should track first moment"
    assert layer_id in adam.v, "Adam should track second moment"
    assert 'weights' in adam.m[layer_id], "Adam should track weights moment"
    
    print("✅ Adam optimizer test passed!")

def test_optimizer_with_batch_norm():
    """Test optimizers with BatchNorm parameters"""
    print("🧪 Testing Optimizers with BatchNorm...")
    
    from visualife.core.layers import BatchNorm
    
    # Create BatchNorm layer
    bn_layer = BatchNorm(3)
    bn_layer.gamma = np.ones((1, 3)) * 1.5
    bn_layer.beta = np.ones((1, 3)) * 0.5
    bn_layer.dgamma = np.ones((1, 3)) * 0.1
    bn_layer.dbeta = np.ones((1, 3)) * 0.05
    
    # Test with SGD
    sgd = SGD(learning_rate=0.1)
    original_gamma = bn_layer.gamma.copy()
    original_beta = bn_layer.beta.copy()
    
    sgd.update(bn_layer)
    
    # Check BatchNorm parameters were updated
    assert not np.allclose(bn_layer.gamma, original_gamma), "SGD should update gamma"
    assert not np.allclose(bn_layer.beta, original_beta), "SGD should update beta"
    print("✅ Optimizer with BatchNorm test passed!")

def test_optimizer_learning_rate_effect():
    """Test that learning rate actually affects update magnitude"""
    print("🧪 Testing Learning Rate Effect...")
    
    layer1 = Dense(2, 1)
    layer1.weights = np.array([[0.5], [0.3]])
    layer1.dweights = np.array([[0.1], [0.2]])
    
    layer2 = Dense(2, 1) 
    layer2.weights = np.array([[0.5], [0.3]])
    layer2.dweights = np.array([[0.1], [0.2]])
    
    # Different learning rates
    sgd_small_lr = SGD(learning_rate=0.01)
    sgd_large_lr = SGD(learning_rate=0.1)
    
    sgd_small_lr.update(layer1)
    sgd_large_lr.update(layer2)
    
    # Layer with larger LR should have bigger change
    change1 = np.abs(layer1.weights - np.array([[0.5], [0.3]]))
    change2 = np.abs(layer2.weights - np.array([[0.5], [0.3]]))
    
    assert np.mean(change2) > np.mean(change1), "Larger learning rate should cause bigger changes"
    print("✅ Learning rate effect test passed!")

if __name__ == "__main__":
    print("🚀 Optimizer Tests")
    print("==================")
    test_sgd_optimizer()
    test_momentum_optimizer()
    test_adam_optimizer()
    test_optimizer_with_batch_norm()
    test_optimizer_learning_rate_effect()
    print("🎉 All optimizer tests passed!")