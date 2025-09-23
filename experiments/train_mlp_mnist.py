"""
Train a simple MLP on MNIST using our VisuaLife Engine
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from visualife.core.model import Model
from visualife.core.layers import Dense, Dropout
from visualife.core.activations import ReLU, Softmax
from visualife.core.callbacks import EarlyStopping

def load_mnist_data(samples=5000):
    """
    Load MNIST data using Keras (just for data loading)
    But train with our own VisuaLife Engine!
    """
    try:
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical
        
        print("📦 Loading MNIST data...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Preprocess the data
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Flatten the images (28x28 -> 784)
        X_train = X_train.reshape(-1, 28*28)
        X_test = X_test.reshape(-1, 28*28)
        
        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        # Use only a subset for quick training
        X_train = X_train[:samples]
        y_train = y_train[:samples]
        X_test = X_test[:1000]  # Small test set for validation
        y_test = y_test[:1000]
        
        print(f"✅ Loaded {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        return (X_train, y_train), (X_test, y_test)
        
    except ImportError:
        print("❌ TensorFlow/Keras not available. Using random data instead.")
        # Fallback: Create random data with same shape as MNIST
        X_train = np.random.rand(5000, 784).astype('float32')
        y_train = np.eye(10)[np.random.randint(0, 10, 5000)]
        X_test = np.random.rand(1000, 784).astype('float32') 
        y_test = np.eye(10)[np.random.randint(0, 10, 1000)]
        return (X_train, y_train), (X_test, y_test)

def create_mlp_model():
    """Create a simple MLP model for MNIST"""
    model = Model()
    
    # Input: 784 features (28x28 flattened), Output: 10 classes
    model.add(Dense(784, 128, l1_reg=0.0001, l2_reg=0.0001))  # Small regularization
    model.add(ReLU())
    model.add(Dropout(0.3))  # Dropout for regularization
    model.add(Dense(128, 64))
    model.add(ReLU())
    model.add(Dense(64, 10))
    model.add(Softmax())
    
    return model

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['train_loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['train_accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mnist_training_history.png')
    plt.show()

def test_predictions(model, X_test, y_test, num_samples=5):
    """Test model predictions on a few samples"""
    print("\n🔍 Testing predictions on sample images...")
    
    predictions = model.predict(X_test[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:num_samples], axis=1)
    
    for i in range(num_samples):
        print(f"Sample {i+1}: Predicted={predicted_classes[i]}, Actual={true_classes[i]}, "
              f"Confidence={predictions[i][predicted_classes[i]]:.2f}")

def main():
    """Main training function"""
    print("🚀 MNIST MLP Training with VisuaLife Engine")
    print("=" * 50)
    
    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist_data(samples=5000)
    
    # Create model
    model = create_mlp_model()
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        learning_rate=0.001
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    
    print("\n🎯 Starting training...")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1]}")
    print(f"Output classes: {y_train.shape[1]}")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=5,  # Small number of epochs for quick test
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Test predictions
    test_predictions(model, X_test, y_test)
    
    # Plot training history (if matplotlib is available)
    try:
        plot_training_history(history)
    except ImportError:
        print("📊 Matplotlib not available. Skipping plots.")
    
    # Save the model
    model.save('mnist_mlp_model.pkl')
    print("💾 Model saved as 'mnist_mlp_model.pkl'")
    
    print("\n🎉 MNIST Training Complete!")
    print("Our VisuaLife Engine is working correctly! 🚀")

if __name__ == "__main__":
    main()