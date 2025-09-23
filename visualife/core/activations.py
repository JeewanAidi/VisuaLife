import numpy as np

class ReLU:
    """
    Rectified Linear Unit activation function.
    forward: f(x) = max(0, x)
    backward: f'(x) = 1 if x > 0 else 0
    """
    def __init__(self):
        self.input = None
    
    def forward(self, X):
        self.input = X
        return np.maximum(0, X)
    
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        return dZ * (self.input > 0)

class LeakyReLU:
    """
    Leaky ReLU activation function.
    forward: f(x) = x if x > 0 else alpha * x
    backward: f'(x) = 1 if x > 0 else alpha
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.input = None
    
    def forward(self, X):
        self.input = X
        return np.where(X > 0, X, self.alpha * X)
    
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        return dZ * np.where(self.input > 0, 1, self.alpha)

class ParametricReLU:
    """
    Parametric ReLU activation function.
    forward: f(x) = x if x > 0 else alpha * x
    backward: f'(x) = 1 if x > 0 else alpha
    Learnable alpha parameter.
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha  # This can be made learnable
        self.input = None
    
    def forward(self, X):
        self.input = X
        return np.where(X > 0, X, self.alpha * X)
    
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        return dZ * np.where(self.input > 0, 1, self.alpha)
class ELU:
    """
    Exponential Linear Unit activation function.
    forward: f(x) = x if x > 0 else alpha * (exp(x) - 1)
    backward: f'(x) = 1 if x > 0 else f(x) + alpha
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.output = None
    
    def forward(self, X):
        self.output = np.where(X > 0, X, self.alpha * (np.exp(X) - 1))
        return self.output
    
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        grad = np.where(self.output > 0, 1, self.output + self.alpha)
        return dZ * grad

class Swish:
    """
    Swish activation function: x * sigmoid(x)
    forward: f(x) = x * sigmoid(x)
    backward: f'(x) = f(x) + sigmoid(x) * (1 - f(x))
    """
    def __init__(self):
        self.output = None
        self.sigmoid = None
    
    def forward(self, X):
        self.sigmoid = 1 / (1 + np.exp(-np.clip(X, -500, 500)))
        self.output = X * self.sigmoid
        return self.output
    
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        grad = self.output + self.sigmoid * (1 - self.output)
        return dZ * grad

class Sigmoid:
    """
    Sigmoid activation function.
    forward: f(x) = 1 / (1 + exp(-x))
    backward: f'(x) = f(x) * (1 - f(x))
    """
    def __init__(self):
        self.output = None
    
    def forward(self, X):
        X = np.clip(X, -500, 500)
        self.output = 1 / (1 + np.exp(-X))
        return self.output
    
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        return dZ * self.output * (1 - self.output)
    
class Softmax:
    """
    Softmax activation function for multi-class classification.
    forward: f(x_i) = exp(x_i) / sum(exp(x_j))
    """
    def __init__(self):
        self.output = None
    
    def forward(self, X):
        # Numerical stability
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        return dZ

class Tanh:
    """
    Hyperbolic tangent activation function.
    forward: f(x) = tanh(x)
    backward: f'(x) = 1 - tanh²(x)
    """
    def __init__(self):
        self.output = None
    
    def forward(self, X):
        self.output = np.tanh(X)
        return self.output
    
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        return dZ * (1 - self.output ** 2)