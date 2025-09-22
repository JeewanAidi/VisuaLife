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



def elementwise_multiply(A, B):
    # Check if the matrices have the same shape.
    if A.shape != B.shape:
        raise ValueError(f"Shapes {A.shape} and {B.shape} must be identical for element-wise multiplication.")

    # Initialize the output matrix with zeros.
    result = np.zeros(A.shape)

    # Perform element-wise multiplication using nested loops.
    # For a 2D array, we need two loops.
    for i in range(A.shape[0]): # Iterate over rows
        for j in range(A.shape[1]): # Iterate over columns
            result[i, j] = A[i, j] * B[i, j]

    return result



def sum_array(A, axis=None):
    if axis is None:
        # Sum all elements into a single scalar value.
        total = 0.0
        # Use a nested loop to iterate through every element
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                total += A[i, j]
        return total
    elif axis == 0:
        # Sum along columns (result will be a row with shape (1, columns))
        result = np.zeros((1, A.shape[1]))
        for j in range(A.shape[1]): # For each column...
            col_sum = 0.0
            for i in range(A.shape[0]): # ...sum all rows in that column
                col_sum += A[i, j]
            result[0, j] = col_sum
        return result
    elif axis == 1:
        # Sum along rows (result will be a column with shape (rows, 1))
        result = np.zeros((A.shape[0], 1))
        for i in range(A.shape[0]): # For each row...
            row_sum = 0.0
            for j in range(A.shape[1]): # ...sum all columns in that row
                row_sum += A[i, j]
            result[i, 0] = row_sum
        return result
    else:
        raise ValueError("Axis must be None, 0, or 1 for 2D arrays.")
    
