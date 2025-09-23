import numpy as np
from visualife.core.engine import matrix_multiply, elementwise_multiply

class Conv2D:
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = None
        self.bias = None
        self.input = None
        self.output = None
        
    def initialize_parameters(self, input_channels):
        """Initialize filters and bias"""
        # He initialization for ReLU
        scale = np.sqrt(2.0 / (self.filter_size * self.filter_size * input_channels))
        self.filters = np.random.randn(self.num_filters, self.filter_size, self.filter_size, input_channels) * scale
        self.bias = np.zeros((self.num_filters, 1))
    
    def forward(self, X):
        """
        Forward pass for Conv2D layer
        X shape: (batch_size, height, width, channels)
        """
        if self.filters is None:
            self.initialize_parameters(X.shape[-1])
            
        self.input = X
        batch_size, in_height, in_width, in_channels = X.shape
        
        # Calculate output dimensions
        out_height = (in_height - self.filter_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.filter_size + 2 * self.padding) // self.stride + 1
        
        # Initialize output
        self.output = np.zeros((batch_size, out_height, out_width, self.num_filters))
        
        # Apply padding if needed
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (self.padding, self.padding), 
                                 (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            X_padded = X
        
        # Convolution operation
        for i in range(batch_size):  # For each sample in batch
            for h in range(out_height):  # For each output row
                for w in range(out_width):  # For each output column
                    for f in range(self.num_filters):  # For each filter
                        # Extract the receptive field
                        vert_start = h * self.stride
                        vert_end = vert_start + self.filter_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.filter_size
                        
                        receptive_field = X_padded[i, vert_start:vert_end, horiz_start:horiz_end, :]
                        
                        # Perform convolution (element-wise multiplication and sum)
                        conv_result = np.sum(receptive_field * self.filters[f])
                        self.output[i, h, w, f] = conv_result + self.bias[f]
        
        return self.output
    
    def backward(self, dZ, learning_rate=0.01):
        """
        Backward pass for Conv2D layer
        dZ shape: (batch_size, out_height, out_width, num_filters)
        """
        X = self.input
        batch_size, in_height, in_width, in_channels = X.shape
        _, out_height, out_width, num_filters = dZ.shape
        
        # Initialize gradients
        d_filters = np.zeros_like(self.filters)
        d_bias = np.zeros_like(self.bias)
        dX = np.zeros_like(X)
        
        # Apply padding if needed
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (self.padding, self.padding), 
                                 (self.padding, self.padding), (0, 0)), mode='constant')
            dX_padded = np.pad(dX, ((0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            X_padded = X
            dX_padded = dX
        
        # Compute gradients
        for i in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for f in range(num_filters):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.filter_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.filter_size
                        
                        # Gradient for filters
                        receptive_field = X_padded[i, vert_start:vert_end, horiz_start:horiz_end, :]
                        d_filters[f] += receptive_field * dZ[i, h, w, f]
                        
                        # Gradient for input
                        dX_padded[i, vert_start:vert_end, horiz_start:horiz_end, :] += self.filters[f] * dZ[i, h, w, f]
        
        # Remove padding from dX_padded if needed
        if self.padding > 0:
            dX = dX_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX = dX_padded
        
        # Gradient for bias
        d_bias = np.sum(dZ, axis=(0, 1, 2)).reshape(-1, 1)
        
        # Update parameters
        self.filters -= learning_rate * d_filters / batch_size
        self.bias -= learning_rate * d_bias / batch_size
        
        return dX

class MaxPool2D:
    """Max Pooling Layer"""
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.output = None
        self.mask = None
    
    def forward(self, X):
        self.input = X
        batch_size, height, width, channels = X.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        self.output = np.zeros((batch_size, out_height, out_width, channels))
        self.mask = np.zeros_like(X)
        
        for i in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(channels):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.pool_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.pool_size
                        
                        window = X[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        self.output[i, h, w, c] = np.max(window)
                        
                        # Create mask for backward pass
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        self.mask[i, vert_start + max_pos[0], horiz_start + max_pos[1], c] = 1
        
        return self.output
    
    def backward(self, dZ, learning_rate=None):
        dX = np.zeros_like(self.input)
        batch_size, out_height, out_width, channels = dZ.shape
        
        for i in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(channels):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.pool_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.pool_size
                        
                        # Route gradient to the max position
                        dX[i, vert_start:vert_end, horiz_start:horiz_end, c] += (
                            self.mask[i, vert_start:vert_end, horiz_start:horiz_end, c] * dZ[i, h, w, c]
                        )
        
        return dX

class Flatten:
    """Flatten layer for CNN to Dense transition"""
    def __init__(self):
        self.input_shape = None
    
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dZ, learning_rate=None):
        return dZ.reshape(self.input_shape)