"""
Simple MLP test without external dependencies
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from visualife.core.model import Model
from visualife.core.layers import Dense
from visualife.core.activations import ReLU, Softmax

def create_simple_data():
    """Create simple synthetic data for testing"""
    # Simple binary classification problem
    np.random.seed(42)
    
    # Create linearly separable data
    n_samples = 1000
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear decision boundary
    
    # Convert to one-hot
    y_onehot = np.eye(2)[y]
    
    return X, y_onehot

def test_simple_mlp():
    """Test MLP on simple synthetic data"""
    print("🧪 Testing MLP on Synthetic Data")
    print("=" * 40)
    
    # Create data
    X, y = create_simple_data()
    
    # Create simple model
    model = Model()
    model.add(Dense(2, 4))  # 2 inputs -> 4 hidden units
    model.add(ReLU())
    model.add(Dense(4, 2))  # 4 hidden -> 2 outputs
    model.add(Softmax())
    
    # Compile
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam', 
        learning_rate=0.01
    )
    
    print("Model architecture:")
    model.summary()
    
    # Train for a few epochs
    print("\nTraining...")
    history = model.fit(X, y, epochs=10, batch_size=16, verbose=1)
    
    # Test predictions
    test_X = np.array([[1, 1], [-1, -1], [2, -1], [-2, 1]])  # Easy test cases
    predictions = model.predict(test_X)
    
    print("\n🔍 Predictions:")
    for i, pred in enumerate(predictions):
        class_idx = np.argmax(pred)
        confidence = pred[class_idx]
        print(f"Input {test_X[i]} -> Class {class_idx} (confidence: {confidence:.3f})")
    
    # Check if model learned the simple pattern
    # For points with positive sum, should predict class 1
    test_sums = test_X[:, 0] + test_X[:, 1]
    predicted_classes = np.argmax(predictions, axis=1)
    
    correct = 0
    for i in range(len(test_X)):
        expected = 1 if test_sums[i] > 0 else 0
        if predicted_classes[i] == expected:
            correct += 1
    
    accuracy = correct / len(test_X)
    print(f"\n✅ Simple pattern test accuracy: {accuracy:.1%}")
    
    if accuracy > 0.7:  # Should be very easy for MLP
        print("🎉 MLP is learning correctly!")
        return True
    else:
        print("❌ MLP might have issues")
        return False

if __name__ == "__main__":
    success = test_simple_mlp()
    if success:
        print("\n🚀 VisuaLife Engine is working! Ready for MNIST and Conv2D!")
    else:
        print("\n⚠️  There might be issues with the implementation.")