import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.core.engine import matrix_multiply, elementwise_multiply, sum_array

if __name__ == "__main__":
    print("VisuaLife Engine - Test: Matrix Multiplication")
    print("=======================================================")

    # Test Case 1: Simple 2x2 * 2x2
    print("\n1. Testing 2x2 * 2x2")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])

    result = matrix_multiply(A, B)
    expected = np.array([[19, 22],
                         [43, 50]])

    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Our Result:\n", result)
    print("Expected Result:\n", expected)
    print("Test Passed:", np.allclose(result, expected))
    print("2x2 Test Passed")

    # Test Case 2: 3x2 * 2x3 -> should be 3x3
    print("\n\n2. Testing 3x2 * 2x3")
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    B = np.array([[1, 2, 3],
                  [4, 5, 6]])

    result = matrix_multiply(A, B)
    expected = np.array([[9, 12, 15],
                         [19, 26, 33],
                         [29, 40, 51]])

    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Our Result:\n", result)
    print("Expected Result:\n", expected)
    print("Test Passed:", np.allclose(result, expected))
    print("3x2 * 2x3 Test Passed")

    print("All matrix_multiply tests passed!\n")

        # Test Case 3: Element-wise Multiplication
    print("\n\n3. Testing Element-wise Multiplication")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = elementwise_multiply(A, B)
    expected = A * B 
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Our Result (A ⊙ B):\n", result)
    print("Expected Result:\n", expected)
    print("Test Passed:", np.allclose(result, expected))
    print("Element-wise multiplication test passed!\n")

    # Test Case 4: Sum Array
    print("\n\n4. Testing Sum Array")
    A = np.array([[1, 2], [3, 4]])
    print("Matrix A:\n", A)
    
    total_sum = sum_array(A, axis=None)
    print(f"Sum of all elements (axis=None): {total_sum}. Expected: {np.sum(A)}")
    print("Test Passed:", np.allclose(total_sum, np.sum(A)))
    print("Sum of all elements test passed!")
    
    col_sum = sum_array(A, axis=0)
    print(f"Sum along columns (axis=0): {col_sum}. Expected: {np.sum(A, axis=0, keepdims=True)}")
    print("Test Passed:", np.allclose(col_sum, np.sum(A, axis=0, keepdims=True)))
    print("Sum along columns test passed!")
    
    row_sum = sum_array(A, axis=1)
    print(f"Sum along rows (axis=1): {row_sum}. Expected: {np.sum(A, axis=1, keepdims=True)}")
    print("Test Passed:", np.allclose(row_sum, np.sum(A, axis=1, keepdims=True)))
    print("Sum along rows test passed!")

    print("All sum_array tests passed!\n")

    print("\nStep 2 FULLY Complete! The Core Engine is ready.")