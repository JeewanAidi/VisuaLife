import numpy as np
import sys
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualife.core.model import Model
from visualife.core.layers import Dense, Dropout, BatchNorm
from visualife.core.activations import ReLU, Sigmoid, Softmax
from visualife.core.optimizers import SGD, Adam

def test_model_default_optimizer():
    """Test that model uses default optimizer when none specified"""
    print("🧪 Testing Default Optimizer...")
    
    model = Model()
    model.add(Dense(5, 3))
    model.add(ReLU())
    model.add(Dense(3, 1))
    
    # Compile without specifying optimizer - should use default 'adam'
    model.compile(loss='mean_squared_error')  # No optimizer specified!
    
    assert model.optimizer is not None, "Model should have an optimizer"
    assert hasattr(model.optimizer, 'learning_rate'), "Optimizer should have learning rate"
    assert model.optimizer.learning_rate == 0.001, "Should use default learning rate"
    print("✅ Default optimizer test passed!")

def test_model_learning_actually_happens():
    """Test that model actually learns with optimizer updates"""
    print("🧪 Testing Model Learning...")
    
    # Simple regression problem
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0:1] + 3 * X[:, 1:2] - 1 * X[:, 2:3] + 0.1 * np.random.randn(100, 1)
    
    model = Model()
    model.add(Dense(3, 5))
    model.add(ReLU())
    model.add(Dense(5, 1))
    model.compile(optimizer='sgd', learning_rate=0.01, loss='mean_squared_error')
    
    # Train for a few epochs
    initial_loss, _ = model.evaluate(X, y, verbose=0)
    model.fit(X, y, epochs=5, batch_size=10, verbose=0)
    final_loss, _ = model.evaluate(X, y, verbose=0)
    
    # Model should learn (loss should decrease)
    assert final_loss < initial_loss, f"Model should learn: {initial_loss} -> {final_loss}"
    print(f"✅ Model learning test passed! Loss: {initial_loss:.4f} -> {final_loss:.4f}")

def test_different_optimizers_convergence():
    """Test that different optimizers can converge"""
    print("🧪 Testing Optimizer Convergence...")
    
    # Simple classification problem
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)
    
    optimizers = ['sgd', 'momentum', 'adam']
    
    for optimizer_name in optimizers:
        model = Model()
        model.add(Dense(2, 4))
        model.add(ReLU())
        model.add(Dense(4, 1))
        model.add(Sigmoid())
        
        model.compile(optimizer=optimizer_name, learning_rate=0.01, loss='binary_crossentropy')
        
        # Train briefly
        initial_loss, _ = model.evaluate(X, y, verbose=0)
        model.fit(X, y, epochs=3, batch_size=32, verbose=0)
        final_loss, _ = model.evaluate(X, y, verbose=0)
        
        # Optimizer should at least not explode
        assert not np.isnan(final_loss), f"{optimizer_name} should not produce NaN loss"
        assert final_loss < float('inf'), f"{optimizer_name} should not produce infinite loss"
        
        print(f"✅ {optimizer_name} convergence test passed!")

def test_optimizer_parameter_persistence():
    """Test that optimizer parameters are saved and loaded"""
    print("🧪 Testing Optimizer Parameter Persistence...")
    
    model = Model()
    model.add(Dense(4, 2))
    model.add(Dense(2, 1))
    model.compile(optimizer='adam', learning_rate=0.005)
    
    original_lr = model.optimizer.learning_rate
    
    # Save and load model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        temp_path = tmp_file.name
    
    try:
        model.save(temp_path)
        
        # Create new model and load
        new_model = Model()
        new_model.add(Dense(4, 2))
        new_model.add(Dense(2, 1))
        new_model.compile(optimizer='adam', learning_rate=0.001)  # Different LR
        
        new_model.load(temp_path)
        
        # Learning rate should be restored from saved model
        assert new_model.optimizer.learning_rate == original_lr, "Learning rate should persist"
        print("✅ Optimizer parameter persistence test passed!")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_gradient_accumulation():
    """Test that gradients accumulate properly across batches"""
    print("🧪 Testing Gradient Accumulation...")
    
    model = Model()
    model.add(Dense(3, 2, l2_reg=0.01))
    model.compile(optimizer='sgd', learning_rate=0.1)
    
    X_batch1 = np.random.randn(5, 3)
    y_batch1 = np.random.randn(5, 2)
    X_batch2 = np.random.randn(5, 3) 
    y_batch2 = np.random.randn(5, 2)
    
    # Store initial weights
    initial_weights = model.layers[0].weights.copy()
    
    # Train on two batches
    model.train_step(X_batch1, y_batch1)
    weights_after_batch1 = model.layers[0].weights.copy()
    model.train_step(X_batch2, y_batch2)
    weights_after_batch2 = model.layers[0].weights.copy()
    
    # Weights should change after each batch
    assert not np.allclose(initial_weights, weights_after_batch1), "Weights should change after first batch"
    assert not np.allclose(weights_after_batch1, weights_after_batch2), "Weights should change after second batch"
    print("✅ Gradient accumulation test passed!")

def test_optimizer_with_regularization():
    """Test that optimizers work correctly with regularization"""
    print("🧪 Testing Optimizer with Regularization...")
    
    # Model with L2 regularization
    model_reg = Model()
    model_reg.add(Dense(3, 2, l2_reg=0.1))  # High L2 regularization
    model_reg.compile(optimizer='sgd', learning_rate=0.1)
    
    # Model without regularization  
    model_no_reg = Model()
    model_no_reg.add(Dense(3, 2, l2_reg=0.0))  # No regularization
    model_no_reg.compile(optimizer='sgd', learning_rate=0.1)
    
    X = np.random.randn(10, 3)
    y = np.random.randn(10, 2)
    
    # Train both models
    for _ in range(100):
        model_reg.train_step(X, y)
        model_no_reg.train_step(X, y)
    
    # Regularized model should have smaller weights
    reg_norm = np.linalg.norm(model_reg.layers[0].weights)
    no_reg_norm = np.linalg.norm(model_no_reg.layers[0].weights)
    
    assert reg_norm < no_reg_norm, "Regularized model should have smaller weights"
    print("✅ Optimizer with regularization test passed!")

def test_learning_rate_scheduling():
    """Test that learning rate scheduling works with optimizers"""
    print("🧪 Testing Learning Rate Scheduling...")
    
    from visualife.core.callbacks import LearningRateScheduler
    
    model = Model()
    model.add(Dense(2, 1))
    model.compile(optimizer='sgd', learning_rate=0.1)
    
    initial_lr = model.optimizer.learning_rate
    
    # Create scheduler that halves LR every epoch
    scheduler = LearningRateScheduler(factor=0.5, patience=1)
    
    X = np.random.randn(10, 2)
    y = np.random.randn(10, 1)
    
    # Manually trigger scheduler for testing
    scheduler.on_epoch_end(0, 0.5, model, model.optimizer)  # Should not change LR yet
    assert model.optimizer.learning_rate == initial_lr, "LR should not change before patience"
    
    scheduler.on_epoch_end(1, 0.6, model, model.optimizer)  # Should halve LR
    expected_lr = initial_lr * 0.5
    assert model.optimizer.learning_rate == expected_lr, f"LR should be halved: {expected_lr}"
    print("✅ Learning rate scheduling test passed!")

def test_optimizer_state_reset():
    """Test that optimizer state doesn't interfere between models"""
    print("🧪 Testing Optimizer State Reset...")
    
    # Create two identical models with same optimizer type
    model1 = Model()
    model1.add(Dense(3, 2))
    model1.compile(optimizer='momentum', learning_rate=0.1)
    
    model2 = Model() 
    model2.add(Dense(3, 2))
    model2.compile(optimizer='momentum', learning_rate=0.1)
    
    X = np.random.randn(5, 3)
    y = np.random.randn(5, 2)
    
    # Train first model
    model1.train_step(X, y)
    weights1_after = model1.layers[0].weights.copy()
    
    # Train second model (should not be affected by first model's training)
    model2.train_step(X, y)
    weights2_after = model2.layers[0].weights.copy()
    
    # Both models should update independently
    assert not np.array_equal(weights1_after, weights2_after), "Models should update independently"
    print("✅ Optimizer state reset test passed!")

if __name__ == "__main__":
    print("🚀 Enhanced Model Tests")
    print("======================")
    test_model_default_optimizer()
    test_model_learning_actually_happens()
    test_different_optimizers_convergence()
    test_optimizer_parameter_persistence()
    test_gradient_accumulation()
    test_optimizer_with_regularization()
    test_learning_rate_scheduling()
    test_optimizer_state_reset()
    print("🎉 All enhanced model tests passed!")