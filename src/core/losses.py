import numpy as np

class MeanSquaredError:
    """
    Mean Squared Error loss for regression tasks.
    forward: L = 1/n * Σ(y_true - y_pred)²
    backward: dL/dy_pred = -2/n * (y_true - y_pred)
    """
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self):
        batch_size = self.y_true.shape[0]
        return -2 * (self.y_true - self.y_pred) / batch_size

class BinaryCrossEntropy:
    """
    Binary Cross Entropy loss for binary classification.
    forward: L = -1/n * Σ[y_true*log(y_pred) + (1-y_true)*log(1-y_pred)]
    backward: dL/dy_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))
    """
    def forward(self, y_pred, y_true):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self):
        batch_size = self.y_true.shape[0]
        y_pred_clipped = np.clip(self.y_pred, 1e-15, 1 - 1e-15)
        return (y_pred_clipped - self.y_true) / (y_pred_clipped * (1 - y_pred_clipped)) / batch_size

class CrossEntropyLoss:
    """
    Categorical Cross Entropy loss for multi-class classification.
    Combined with Softmax for efficient gradient computation.
    """
    def forward(self, y_pred, y_true):
        # y_pred should be softmax outputs, y_true should be one-hot encoded
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_pred = y_pred
        self.y_true = y_true
        # Cross entropy: -Σ y_true * log(y_pred)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def backward(self):
        # When combined with softmax, gradient is simply (y_pred - y_true)
        # This is much more efficient than separate softmax + crossentropy derivatives
        batch_size = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / batch_size