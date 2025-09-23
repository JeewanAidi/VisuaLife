"""
Test the complete training pipeline
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from visualife.core.model import Model
from visualife.core.layers import Dense
from visualife.core.activations import ReLU, Sigmoid

def test_complete_training():
    """Test a complete training cycle"""
    print("🧪 Testing Complete Training Pipeline...")
    
    # Create a simple model
    model = Model()
    model.add(Dense(2, 4, l1_reg=0.01, l2_reg=0.01))
    model.add(ReLU())
    model.add(Dense(4, 1))
    model.add(Sigmoid())
    
    # Compile with Adam (your default)
    model.compile(loss='binary_crossentropy', optimizer='adam', learning_rate=0.001)
    
    # Create simple training data
    X_train = np.random.randn(10, 2)
    y_train = np.random.randint(0, 2, (10, 1))
    
    # Test forward pass
    output = model.forward(X_train)
    assert output.shape == (10, 1), f"Forward pass failed. Expected (10,1), got {output.shape}"
    print("✅ Forward pass test passed!")
    
    # Test prediction
    predictions = model.predict(X_train)
    assert predictions.shape == (10, 1), "Prediction failed"
    print("✅ Prediction test passed!")
    
    # Test training step
    try:
        loss, accuracy = model.train_step(X_train, y_train)
        assert isinstance(loss, float), "Training step failed"
        print("✅ Training step test passed!")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False
    
    # Test model summary
    model.summary()
    print("✅ Model summary test passed!")
    
    print("🎉 Complete pipeline test passed!")
    return True

if __name__ == "__main__":
    test_complete_training()