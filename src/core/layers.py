import numpy as np
from src.core.engine import matrix_multiply, sum_array 

class Dense:
   
    def __init__(self, input_size, output_size):
        # He initialization: weights ~ N(0, sqrt(2./fan_in))
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        
        # Store inputs and gradients for the backward pass
        self.X = None
        self.dweights = None
        self.dbiases = None

    def forward(self, X):
        # Save the input for use in the backward pass
        self.X = X
        # Perform the linear transformation: Z = X.W + b
        output = matrix_multiply(X, self.weights) + self.biases
        return output

    def backward(self, dZ, learning_rate):
        # Get the batch size for averaging the gradients
        batch_size = self.X.shape[0]
        
        # Calculate gradients for parameters
        # dW = X.T . dZ
        self.dweights = matrix_multiply(self.X.T, dZ) / batch_size # Average over the batch
        # db = sum(dZ, axis=0) --> sum of gradients for each neuron in the layer
        self.dbiases = sum_array(dZ, axis=0) / batch_size # Average over the batch
        
        # Calculate gradient for the input (to pass to previous layer)
        # dX = dZ . W.T
        dX = matrix_multiply(dZ, self.weights.T)
        
        # Update parameters using Gradient Descent
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
        
        return dX