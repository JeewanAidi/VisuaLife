import numpy as np

def matrix_multiply(A, B):
    # Check if the matrices can be multiplied.
    # The number of columns in A must equal the number of rows in B.
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Shapes {A.shape} and {B.shape} are not aligned for matrix multiplication. Columns of A must equal Rows of B.")

    # Get the dimensions of the matrices.
    m, n = A.shape # A has 'm' rows and 'n' columns
    n, p = B.shape # B has 'n' rows and 'p' columns

    # Initialize the output matrix C with zeros.
    # We use np.zeros here only for array allocation, which is allowed.
    C = np.zeros((m, p))

    # Perform the matrix multiplication using triple nested loops.
    # For every row 'i' in matrix A...
    for i in range(m):
        # For every column 'j' in matrix B...
        for j in range(p):
            # Initialize a variable to hold the sum for this element C[i, j]
            total = 0.0
            # For each element 'k' in the shared dimension 'n'...
            for k in range(n):
                # Multiply element A[i, k] by element B[k, j] and add to the total
                total += A[i, k] * B[k, j]
            # Assign the computed total to the corresponding position in the result matrix
            C[i, j] = total

    return C


# ===== TEST CODE =====
# This block only runs if we execute this file directly: `python engine.py`
if __name__ == "__main__":
    print("🧠 VisuaLife Engine - Day 2 Test: Matrix Multiplication")
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
    print("✅ Test Passed:", np.allclose(result, expected))

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
    print("✅ Test Passed:", np.allclose(result, expected))

    print("\n🎉 Day 2 Complete! The foundation of the VisuaLife Engine is solid.")