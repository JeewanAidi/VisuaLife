import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from visualife.core.losses import MeanSquaredError, BinaryCrossEntropy, CrossEntropyLoss

def test_mean_squared_error():
    print("Testing Mean Squared Error...")
    mse = MeanSquaredError()
    
    # Test case 1: Perfect prediction
    y_pred = np.array([[0.5, 0.7], [0.3, 0.9]])
    y_true = np.array([[0.5, 0.7], [0.3, 0.9]])
    
    loss = mse.forward(y_pred, y_true)
    expected_loss = 0.0  # Should be zero for perfect prediction
    assert np.allclose(loss, expected_loss), f"MSE perfect prediction failed! Got {loss}, expected {expected_loss}"
    print("MSE perfect prediction test passed!")
    
    # Test backward pass
    dZ = mse.backward()
    expected_dZ = np.zeros_like(y_pred)  # Gradient should be zero when prediction is perfect
    assert np.allclose(dZ, expected_dZ), "MSE backward pass failed for perfect prediction!"
    print("MSE backward pass (perfect prediction) test passed!")
    
    # Test case 2: Simple error
    y_pred = np.array([[2.0], [4.0]])
    y_true = np.array([[1.0], [3.0]])
    
    loss = mse.forward(y_pred, y_true)
    # MSE = ((2-1)² + (4-3)²) / 2 = (1 + 1) / 2 = 1.0
    expected_loss = 1.0
    assert np.allclose(loss, expected_loss), f"MSE calculation failed! Got {loss}, expected {expected_loss}"
    print("MSE calculation test passed!")
    
    # Test backward pass
    dZ = mse.backward()
    # dL/dy_pred = -2/n * (y_true - y_pred) = -2/2 * ([1-2], [3-4]) = -1 * ([-1], [-1]) = [1, 1]
    expected_dZ = np.array([[1.0], [1.0]])
    assert np.allclose(dZ, expected_dZ), f"MSE gradient failed! Got {dZ}, expected {expected_dZ}"
    print("MSE gradient calculation test passed!\n")




def test_binary_cross_entropy():
    print("Testing Binary Cross Entropy...")
    bce = BinaryCrossEntropy()
    
    # Test case 1: Near-perfect prediction (avoid exact 0 and 1 for stability)
    y_pred = np.array([[0.001], [0.999]])  # Very close to perfect
    y_true = np.array([[0.0], [1.0]])
    
    loss = bce.forward(y_pred, y_true)
    # Loss should be small for good predictions
    assert loss < 1.0, f"BCE good prediction failed! Loss should be small, got {loss}"
    print("BCE good prediction test passed!")
    
    # Test backward pass
    dZ = bce.backward()
    # Gradient should be small for good predictions
    assert np.max(np.abs(dZ)) < 10.0, "BCE gradient should be reasonable for good predictions!"
    print("BCE backward pass (good prediction) test passed!")
    
    # Test case 2: 50% accurate prediction
    y_pred = np.array([[0.5], [0.5]])
    y_true = np.array([[1.0], [0.0]])  # First should be 1, second should be 0
    
    loss = bce.forward(y_pred, y_true)
    # BCE = -[1*log(0.5) + 0*log(0.5) + 0*log(0.5) + 1*log(0.5)] / 2
    # = -[log(0.5) + log(0.5)] / 2 = -log(0.5) ≈ 0.693
    expected_loss = -np.log(0.5)
    assert np.allclose(loss, expected_loss, rtol=1e-5), f"BCE calculation failed! Got {loss}, expected {expected_loss}"
    print("BCE calculation test passed!")
    
    # Test gradient for this case
    dZ = bce.backward()
    # Should have reasonable gradient values
    assert not np.any(np.isnan(dZ)), "BCE gradient contains NaN!"
    assert not np.any(np.isinf(dZ)), "BCE gradient contains Inf!"
    print("BCE gradient stability test passed!")
    
    print("Binary Cross Entropy tests passed!\n")




def test_categorical_cross_entropy():
    print("Testing Categorical Cross Entropy...")
    cel = CrossEntropyLoss()
    
    # Test case 1: Perfect prediction (one-hot encoded)
    y_pred = np.array([[0.1, 0.8, 0.1],  # Should be class 1
                       [0.7, 0.2, 0.1]]) # Should be class 0
    y_true = np.array([[0, 1, 0],        # Class 1
                       [1, 0, 0]])       # Class 0
    
    # Make predictions more confident for perfect case
    y_pred_perfect = np.array([[0.05, 0.90, 0.05],
                               [0.90, 0.05, 0.05]])
    
    loss = cel.forward(y_pred_perfect, y_true)
    # Loss should be small for good predictions
    assert loss < 0.2, f"CrossEntropy good prediction failed! Loss should be small, got {loss}"
    print("CrossEntropy good prediction test passed!")
    
    # Test case 2: Worst possible prediction
    y_pred_worst = np.array([[0.90, 0.05, 0.05],  # Predicts class 0, true is class 1
                             [0.05, 0.90, 0.05]]) # Predicts class 1, true is class 0
    
    loss_worst = cel.forward(y_pred_worst, y_true)
    # Loss should be large for bad predictions
    assert loss_worst > loss, "CrossEntropy should have higher loss for worse predictions!"
    print("CrossEntropy bad prediction test passed!")
    
    # Test backward pass
    dZ = cel.backward()
    assert dZ.shape == y_pred_perfect.shape, "CrossEntropy gradient shape mismatch!"
    print("CrossEntropy gradient shape test passed!")
    
    # Test the special property: when combined with softmax, gradient = (y_pred - y_true)
    y_pred_softmax = np.array([[0.2, 0.7, 0.1],
                               [0.6, 0.3, 0.1]])
    y_true_onehot = np.array([[0, 1, 0],
                              [1, 0, 0]])
    
    cel.forward(y_pred_softmax, y_true_onehot)
    dZ = cel.backward()
    expected_dZ = (y_pred_softmax - y_true_onehot) / y_true_onehot.shape[0]
    assert np.allclose(dZ, expected_dZ), "CrossEntropy+Softmax gradient property failed!"
    print("CrossEntropy+Softmax gradient property test passed!\n")




def test_loss_function_shapes():
    print("Testing loss function shapes...")
    
    # Test various batch sizes and output dimensions
    test_cases = [
        ((5, 1), (5, 1)),    # Binary classification, batch size 5
        ((10, 3), (10, 3)),  # 3-class classification, batch size 10
        ((1, 5), (1, 5)),    # Single sample, 5 outputs
    ]
    
    for (pred_shape, true_shape) in test_cases:
        y_pred = np.random.rand(*pred_shape)
        y_true = np.random.rand(*true_shape)
        
        # Normalize y_true for classification losses
        y_true_class = np.zeros_like(y_true)
        if true_shape[1] > 1:  # Multi-class
            # Create one-hot encoding
            for i in range(true_shape[0]):
                true_class = np.random.randint(0, true_shape[1])
                y_true_class[i, true_class] = 1
        else:  # Binary classification
            y_true_class = (y_true > 0.5).astype(float)
        
        # Test MSE
        mse = MeanSquaredError()
        loss_mse = mse.forward(y_pred, y_true)
        grad_mse = mse.backward()
        assert grad_mse.shape == y_pred.shape, f"MSE gradient shape mismatch for {pred_shape}"
        
        # Test BCE (for binary cases)
        if pred_shape[1] == 1:
            bce = BinaryCrossEntropy()
            loss_bce = bce.forward(y_pred, y_true_class)
            grad_bce = bce.backward()
            assert grad_bce.shape == y_pred.shape, f"BCE gradient shape mismatch for {pred_shape}"
        
        # Test CrossEntropy (for multi-class cases)
        if pred_shape[1] > 1:
            cel = CrossEntropyLoss()
            # Convert predictions to probability distribution (softmax-like)
            y_pred_probs = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
            loss_cel = cel.forward(y_pred_probs, y_true_class)
            grad_cel = cel.backward()
            assert grad_cel.shape == y_pred.shape, f"CrossEntropy gradient shape mismatch for {pred_shape}"
    
    print("All loss function shape tests passed!\n")




def test_numerical_stability():
    print("Testing numerical stability...")
    
    # Test edge cases that could cause numerical issues
    bce = BinaryCrossEntropy()
    cel = CrossEntropyLoss()
    
    # Test very small predictions (shouldn't cause log(0) errors)
    y_pred_small = np.array([[1e-15, 0.5], [0.5, 1e-15]])
    y_true_small = np.array([[1, 0], [0, 1]])
    
    try:
        loss_bce = bce.forward(y_pred_small, y_true_small[:, 0:1])  # Take first column for binary
        print("BCE numerical stability (small values) test passed!")
    except Exception as e:
        assert False, f"BCE failed with small values: {e}"
    
    # Test very large predictions
    y_pred_large = np.array([[0.9999999999, 0.5], [0.5, 0.9999999999]])
    try:
        loss_bce = bce.forward(y_pred_large, y_true_small[:, 0:1])
        print("BCE numerical stability (large values) test passed!")
    except Exception as e:
        assert False, f"BCE failed with large values: {e}"
    
    print("Numerical stability tests passed!\n")




if __name__ == "__main__":
    print("Testing Loss Functions")
    print("=========================")
    test_mean_squared_error()
    test_binary_cross_entropy()
    test_categorical_cross_entropy()
    test_loss_function_shapes()
    test_numerical_stability()
    print("All loss function tests completed successfully!")