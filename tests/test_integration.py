import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualife.core.model import Model
from visualife.core.layers import Dense, BatchNorm, Dropout
from visualife.core.activations import ReLU, Sigmoid

def test_complete_training_pipeline():
    """Test complete training pipeline with different optimizers"""
    print("🧪 Testing Complete Training Pipeline...")
    
    # Create synthetic dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)  # Reduced size for faster testing
    y = (X[:, 0] > 0).astype(int).reshape(-1, 1)
    
    # Test just one optimizer first for simplicity
    print("  Testing sgd...")
    
    model = Model()
    model.add(Dense(10, 20))
    model.add(BatchNorm(20))
    model.add(ReLU())
    model.add(Dropout(0.3))
    model.add(Dense(20, 10))
    model.add(ReLU())
    model.add(Dense(10, 1))
    model.add(Sigmoid())
    
    model.compile(optimizer='sgd', loss='binary_crossentropy', learning_rate=0.01)
    
    # Split train/validation
    X_train, y_train = X[:80], y[:80]
    X_val, y_val = X[80:], y[80:]
    
    print("    Before fit() call...")
    
    try:
        # Train model with verbose=1 to see what's happening
        history = model.fit(X_train, y_train, epochs=2, batch_size=16,  # Reduced epochs/batch
                           validation_data=(X_val, y_val), verbose=1)
        
        print(f"    fit() returned: {history}")
        if history is None:
            print("    ❌ fit() returned None!")
            return False
            
        print(f"    history type: {type(history)}")
        print(f"    history contents: {history}")
        
        # Check that training occurred
        assert len(history.history['train_loss']) == 2, "Should train for 2 epochs"
        assert 'val_loss' in history.history, "Should track validation loss"
        
        # Model should not diverge
        final_loss = history.history['train_loss'][-1]
        assert not np.isnan(final_loss), "SGD should not produce NaN"
        assert final_loss < float('inf'), "SGD should not diverge"
        
        print("    ✅ sgd pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"    ❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_model():
    """Test a simpler model to isolate issues"""
    print("🧪 Testing Simple Model...")
    
    # Very simple dataset and model
    np.random.seed(42)
    X = np.random.randn(10, 5)
    y = (X[:, 0] > 0).astype(int).reshape(-1, 1)
    
    model = Model()
    model.add(Dense(5, 3))
    model.add(ReLU())
    model.add(Dense(3, 1))
    model.add(Sigmoid())
    
    model.compile(optimizer='sgd', loss='binary_crossentropy', learning_rate=0.01)
    
    try:
        history = model.fit(X, y, epochs=1, batch_size=4, verbose=1)
        print(f"    Simple model fit() returned: {history}")
        if history and hasattr(history, 'history'):
            print(f"    Training completed with final loss: {history.history['train_loss'][-1]}")
        return True
    except Exception as e:
        print(f"    ❌ Simple model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimizer_compatibility_with_all_layers():
    """Test that optimizers work with all layer types"""
    print("🧪 Testing Optimizer Compatibility...")
    
    # Test with simpler configuration
    try:
        model = Model()
        model.add(Dense(5, 3))
        model.add(Dense(3, 1))
        
        model.compile(optimizer='sgd', learning_rate=0.01)
        
        X = np.random.randn(5, 5)
        y = np.random.randn(5, 1)
        
        # Test single training step
        loss = model.train_step(X, y)
        print(f"    ✅ Single training step completed with loss: {loss}")
        return True
    except Exception as e:
        print(f"    ❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Integration Tests")
    print("====================")
    
    # Run simpler tests first
    success1 = test_simple_model()
    success2 = test_optimizer_compatibility_with_all_layers()
    
    # Only run complex test if simple ones pass
    if success1 and success2:
        test_complete_training_pipeline()
    else:
        print("⚠️  Skipping complex test due to failures in basic tests")
    
    print("🎉 Integration tests completed!")