import numpy as np
from visualife.core.engine import matrix_multiply, sum_array

class Dense:
    """
    A fully connected neural network layer with L1/L2 regularization.
    """
    def __init__(self, input_size, output_size, l1_reg=0.0, l2_reg=0.0):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.l1_reg = l1_reg  # L1 regularization strength
        self.l2_reg = l2_reg  # L2 regularization strength
        self.X = None
        self.dweights = None
        self.dbiases = None

    def forward(self, X):
        self.X = X
        output = matrix_multiply(X, self.weights) + self.biases
        return output

    # UPDATE: Removed learning_rate parameter since optimizer handles updates
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        batch_size = self.X.shape[0]
        
        self.dweights = matrix_multiply(self.X.T, dZ) / batch_size
        self.dbiases = sum_array(dZ, axis=0) / batch_size
        
        # Add regularization gradients
        if self.l1_reg > 0:
            self.dweights += self.l1_reg * np.sign(self.weights) / batch_size
        if self.l2_reg > 0:
            self.dweights += self.l2_reg * self.weights / batch_size
        
        dX = matrix_multiply(dZ, self.weights.T)
        return dX

    def regularization_loss(self):
        """Calculate L1 and L2 regularization loss for this layer"""
        reg_loss = 0
        if self.l1_reg > 0:
            reg_loss += self.l1_reg * np.sum(np.abs(self.weights))
        if self.l2_reg > 0:
            reg_loss += self.l2_reg * 0.5 * np.sum(self.weights ** 2)
        return reg_loss


class Dropout:
    """
    Dropout layer for regularization.
    Randomly sets a fraction of inputs to zero during training.
    """
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None
        self.training = True
    
    def forward(self, X):
        if self.training and self.rate > 0:
            self.mask = np.random.binomial(1, 1 - self.rate, size=X.shape) / (1 - self.rate)
            return X * self.mask
        else:
            return X
    
    # UPDATE: Removed learning_rate parameter
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        if self.training and self.rate > 0:
            return dZ * self.mask
        else:
            return dZ


class BatchNorm:
    """
    Batch Normalization layer.
    Normalizes activations to improve training stability and speed.
    """
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        self.training = True
        self.X_centered = None
        self.std = None
        self.X_norm = None
        self.dgamma = None
        self.dbeta = None
    
    def forward(self, X):
        if self.training:
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            self.X_centered = X - mean
            self.std = np.sqrt(var + self.epsilon)
            self.X_norm = self.X_centered / self.std
        else:
            self.X_centered = X - self.running_mean
            self.std = np.sqrt(self.running_var + self.epsilon)
            self.X_norm = self.X_centered / self.std
        
        return self.gamma * self.X_norm + self.beta
    
    # UPDATE: Removed learning_rate parameter
    def backward(self, dZ, learning_rate=None):  # ADD learning_rate parameter
        batch_size = dZ.shape[0]
        
        self.dgamma = np.sum(dZ * self.X_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(dZ, axis=0, keepdims=True)
        
        dX_norm = dZ * self.gamma
        dvar = np.sum(dX_norm * self.X_centered * -0.5 * (self.std ** -3), axis=0, keepdims=True)
        dmean = np.sum(dX_norm * -1 / self.std, axis=0, keepdims=True) + dvar * np.mean(-2 * self.X_centered, axis=0, keepdims=True)
        dX = (dX_norm / self.std + dvar * 2 * self.X_centered / batch_size + dmean / batch_size)
        
        return dX