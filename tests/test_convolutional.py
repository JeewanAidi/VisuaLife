import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the correct path
from visualife.core.convolutional import Conv2D, MaxPool2D, Flatten
from visualife.core.activations import ReLU
from visualife.core.layers import Dense
from visualife.core.model import Model

def test_conv2d_initialization():
    print("🧪 Testing Conv2D Initialization...")
    
    # Test 1: Basic initialization
    conv = Conv2D(num_filters=8, filter_size=3, stride=1, padding=0)
    assert conv.num_filters == 8, "Filter count incorrect"
    assert conv.filter_size == 3, "Filter size incorrect"
    assert conv.stride == 1, "Stride incorrect"
    assert conv.padding == 0, "Padding incorrect"
    assert conv.filters is None, "Filters should not be initialized yet"
    print("✅ Conv2D initialization test passed!")

def test_conv2d_forward_pass():
    print("🧪 Testing Conv2D Forward Pass...")
    
    # Test 1: Simple 2x2 image with 1 filter - FIXED
    conv = Conv2D(num_filters=1, filter_size=2, stride=1, padding=0)
    
    # Create a simple 2x2 image (batch_size=1, height=2, width=2, channels=1)
    X = np.array([[[[1], [2]], [[3], [4]]]])  # Shape: (1, 2, 2, 1)
    
    # Manually set a simple filter
    conv.initialize_parameters(1)  # Initialize with 1 input channel
    conv.filters = np.ones((2, 2, 1, 1))  # All ones filter
    conv.bias = np.zeros((1, 1, 1, 1))
    
    output = conv.forward(X)
    
    # Expected calculation: 1+2+3+4 = 10
    # Output shape should be: (1, 1, 1, 1) for 2x2 input with 2x2 filter, stride 1, padding 0
    expected_output = np.array([[[[10]]]])
    
    assert output.shape == (1, 1, 1, 1), f"Output shape incorrect: {output.shape}"
    assert np.allclose(output, expected_output), f"Expected {expected_output}, got {output}"
    print("✅ Conv2D forward pass (simple) test passed!")
    
    # Test 2: With padding to maintain size
    conv2 = Conv2D(num_filters=1, filter_size=3, stride=1, padding=1)
    X_small = np.ones((1, 3, 3, 1))
    output2 = conv2.forward(X_small)
    
    # With padding=1, 3x3 input -> 3x3 output (same size)
    assert output2.shape == (1, 3, 3, 1), f"Padding output shape incorrect: {output2.shape}"
    print("✅ Conv2D forward pass (padding) test passed!")
    
    # Test 3: Multiple filters and batch
    conv3 = Conv2D(num_filters=2, filter_size=2, stride=1, padding=0)
    X_batch = np.ones((3, 4, 4, 1))  # Batch of 3, 4x4 images
    output3 = conv3.forward(X_batch)
    
    # Output should be: (batch_size=3, height=3, width=3, filters=2)
    assert output3.shape == (3, 3, 3, 2), f"Output shape incorrect: {output3.shape}"
    print("✅ Conv2D forward pass (batch) test passed!")

def test_conv2d_backward_pass():
    print("🧪 Testing Conv2D Backward Pass...")
    
    # Simple test case - FIXED
    conv = Conv2D(num_filters=1, filter_size=2, stride=1, padding=0)
    X = np.ones((1, 3, 3, 1))  # 3x3 input
    
    # Forward pass
    output = conv.forward(X)
    
    # Backward pass with simple gradient
    dZ = np.ones_like(output)  # Gradient of 1's
    dX = conv.backward(dZ)
    
    # Check gradients are computed
    assert conv.d_filters is not None, "Filter gradients not computed"
    assert conv.d_bias is not None, "Bias gradients not computed"
    assert dX.shape == X.shape, f"Input gradient shape incorrect: {dX.shape}"
    
    # For an input of all 1s and dZ of all 1s, d_filters should be the sum of receptive fields
    # Each position sees a 2x2 region of 1s, so each gradient contribution is 4
    # There are 2x2=4 positions in the output, so total gradient should be 4*4=16 for each filter element
    expected_d_filters_value = 4.0  # Each output position contributes 4 (2x2 region of 1s)
    assert np.allclose(conv.d_filters, expected_d_filters_value), f"Filter gradients incorrect. Expected ~{expected_d_filters_value}, got {conv.d_filters[0,0,0,0]}"
    
    print("✅ Conv2D backward pass test passed!")

def test_conv2d_parameter_update():
    print("🧪 Testing Conv2D Parameter Update...")
    
    conv = Conv2D(num_filters=2, filter_size=2, stride=1, padding=0)
    X = np.random.randn(2, 4, 4, 3)  # Batch of 2, 4x4 RGB images
    
    # Forward and backward to compute gradients
    output = conv.forward(X)
    dZ = np.ones_like(output)
    conv.backward(dZ)
    
    # Store original parameters
    original_filters = conv.filters.copy()
    original_bias = conv.bias.copy()
    
    # Update parameters
    learning_rate = 0.1
    conv.update(learning_rate)
    
    # Check if parameters changed
    assert not np.allclose(conv.filters, original_filters), "Filters should be updated"
    assert not np.allclose(conv.bias, original_bias), "Bias should be updated"
    
    print("✅ Conv2D parameter update test passed!")

def test_maxpool2d_forward():
    print("🧪 Testing MaxPool2D Forward Pass...")
    
    # Test 1: Simple 2x2 max pooling - FIXED
    pool = MaxPool2D(pool_size=2, stride=2)
    
    # Create a 4x4 image to avoid dimension issues
    X = np.array([[[[1], [4], [2], [3]],
                   [[3], [2], [1], [4]],
                   [[4], [1], [3], [2]],
                   [[2], [3], [4], [1]]]])  # Shape: (1, 4, 4, 1)
    
    output = pool.forward(X)
    
    # Output should be 2x2 after 2x2 pooling with stride 2
    assert output.shape == (1, 2, 2, 1), f"Output shape incorrect: {output.shape}"
    
    # Check max values: first 2x2 block max is 4, second is 4, etc.
    assert output[0, 0, 0, 0] == 4, "First pool block max incorrect"
    assert output[0, 0, 1, 0] == 4, "Second pool block max incorrect"
    
    print("✅ MaxPool2D forward pass test passed!")

def test_maxpool2d_backward():
    print("🧪 Testing MaxPool2D Backward Pass...")
    
    pool = MaxPool2D(pool_size=2, stride=2)
    
    # Create input where we know the max positions - FIXED with larger input
    X = np.array([[[[1], [4], [2], [3]],
                   [[3], [2], [1], [4]],
                   [[4], [1], [3], [2]],
                   [[2], [3], [4], [1]]]])
    
    # Forward pass to set mask
    output = pool.forward(X)
    
    # Backward pass - gradient should go to max positions
    dZ = np.ones_like(output)  # Gradient of 1 for each output position
    dX = pool.backward(dZ)
    
    # Gradient should only be at positions of max values
    # Each 2x2 block should have gradient 1 at max position, 0 elsewhere
    assert dX.shape == X.shape, f"Gradient shape incorrect: {dX.shape}"
    
    # Check that gradients are properly routed (simplified test)
    assert np.sum(dX) == np.sum(dZ), "Total gradient should be preserved"
    print("✅ MaxPool2D backward pass test passed!")

def test_flatten_layer():
    print("🧪 Testing Flatten Layer...")
    
    flatten = Flatten()
    
    # Test 1: Simple flattening
    X = np.random.randn(2, 4, 4, 3)  # Batch of 2, 4x4 images with 3 channels
    output = flatten.forward(X)
    
    # Flattened shape should be: (2, 4*4*3) = (2, 48)
    assert output.shape == (2, 48), f"Flattened shape incorrect: {output.shape}"
    print("✅ Flatten forward pass test passed!")
    
    # Test 2: Backward pass (reshape)
    dZ = np.ones((2, 48))
    dX = flatten.backward(dZ)
    
    assert dX.shape == X.shape, f"Reshaped gradient incorrect: {dX.shape}"
    print("✅ Flatten backward pass test passed!")

def test_cnn_architecture():
    print("🧪 Testing CNN Architecture Integration...")
    
    # Create a simple CNN: Conv -> ReLU -> Pool -> Flatten -> Dense
    model = Model()
    model.add(Conv2D(num_filters=4, filter_size=3, stride=1, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    model.add(Flatten())
    model.add(Dense(64, 10))  # Adjust input size based on previous layers
    model.add(ReLU())
    
    model.compile(loss='mean_squared_error', optimizer='sgd', learning_rate=0.01)
    
    # Test forward pass with appropriate input size
    X = np.random.randn(2, 8, 8, 1)  # Batch of 2, 8x8 grayscale images
    output = model.forward(X, training=False)
    
    # Check output shape
    assert output.shape == (2, 10), f"CNN output shape incorrect: {output.shape}"
    print("✅ CNN architecture integration test passed!")

def test_conv2d_gradient_flow():
    print("🧪 Testing Conv2D Gradient Flow...")
    
    # Test that gradients flow correctly through convolutional layers
    conv1 = Conv2D(num_filters=2, filter_size=3, stride=1, padding=1)
    relu = ReLU()
    conv2 = Conv2D(num_filters=1, filter_size=3, stride=1, padding=1)
    
    # Simple input
    X = np.random.randn(1, 5, 5, 1)
    
    # Forward pass
    conv1_out = conv1.forward(X)
    relu_out = relu.forward(conv1_out)
    conv2_out = conv2.forward(relu_out)
    
    # Backward pass
    dZ = np.ones_like(conv2_out)
    d_conv2 = conv2.backward(dZ)
    d_relu = relu.backward(d_conv2)
    d_conv1 = conv1.backward(d_relu)
    
    # Check gradients exist and have correct shapes
    assert conv1.d_filters is not None, "Conv1 filter gradients missing"
    assert conv2.d_filters is not None, "Conv2 filter gradients missing"
    assert d_conv1.shape == X.shape, "Final gradient shape incorrect"
    
    print("✅ Conv2D gradient flow test passed!")

def test_edge_cases():
    print("🧪 Testing Edge Cases...")
    
    # Test 1: Very small input
    conv = Conv2D(num_filters=1, filter_size=1, stride=1, padding=0)
    X = np.ones((1, 1, 1, 1))  # 1x1 input
    output = conv.forward(X)
    assert output.shape == (1, 1, 1, 1), "1x1 input handling failed"
    print("✅ Small input handling test passed!")
    
    # Test 2: Multiple channels
    conv = Conv2D(num_filters=2, filter_size=2, stride=1, padding=0)
    X = np.ones((1, 3, 3, 3))  # RGB image
    output = conv.forward(X)
    assert output.shape == (1, 2, 2, 2), "Multi-channel handling failed"
    print("✅ Multi-channel handling test passed!")
    
    # Test 3: Different strides
    conv = Conv2D(num_filters=1, filter_size=2, stride=2, padding=0)
    X = np.ones((1, 4, 4, 1))
    output = conv.forward(X)
    assert output.shape == (1, 2, 2, 1), "Stride 2 handling failed"
    print("✅ Stride handling test passed!")

def run_convolutional_tests():
    """Run all convolutional tests"""
    print("🚀 Running Convolutional Layer Tests...")
    print("="*60)
    
    tests = [
        test_conv2d_initialization,
        test_conv2d_forward_pass,
        test_conv2d_backward_pass,
        test_conv2d_parameter_update,
        test_maxpool2d_forward,
        test_maxpool2d_backward,
        test_flatten_layer,
        test_cnn_architecture,
        test_conv2d_gradient_flow,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__} PASSED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("="*60)
    print("📊 CONVOLUTIONAL TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL CONVOLUTIONAL TESTS PASSED!")
        print("🚀 Your Conv2D implementation is ready for CIFAR-10!")
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the implementation.")

if __name__ == "__main__":
    run_convolutional_tests()