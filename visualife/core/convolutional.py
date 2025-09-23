import numpy as np
from visualife.core.engine import matrix_multiply, elementwise_multiply

class Conv2D:
    def __init__(self, num_filters, filter_size, stride=1, padding=0, input_channels=None):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        self.filters = None
        self.bias = None
        self.input = None
        self.output = None
        self.d_filters = None
        self.d_bias = None
        
    def initialize_parameters(self, input_channels):
        """Initialize filters and bias with proper shape"""
        self.input_channels = input_channels
        # He initialization for ReLU
        scale = np.sqrt(2.0 / (self.filter_size * self.filter_size * input_channels))
        self.filters = np.random.randn(self.filter_size, self.filter_size, input_channels, self.num_filters) * scale
        self.bias = np.zeros((1, 1, 1, self.num_filters))
    
    def forward(self, X):
        """
        Forward pass for Conv2D layer
        X shape: (batch_size, height, width, channels)
        """
        if self.filters is None:
            self.initialize_parameters(X.shape[-1])
            
        self.input = X
        batch_size, in_height, in_width, in_channels = X.shape
        
        # Calculate output dimensions - FIXED FORMULA
        out_height = (in_height + 2 * self.padding - self.filter_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.filter_size) // self.stride + 1
        
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
                        conv_result = np.sum(receptive_field * self.filters[:, :, :, f])
                        self.output[i, h, w, f] = conv_result + self.bias[0, 0, 0, f]
        
        return self.output
    
    def backward(self, dZ):
        """
        Backward pass for Conv2D layer
        dZ shape: (batch_size, out_height, out_width, num_filters)
        """
        X = self.input
        batch_size, in_height, in_width, in_channels = X.shape
        _, out_height, out_width, num_filters = dZ.shape
        
        # Initialize gradients
        self.d_filters = np.zeros_like(self.filters)
        self.d_bias = np.zeros_like(self.bias)
        dX = np.zeros_like(X)
        
        # Apply padding if needed for dX
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (self.padding, self.padding), 
                                 (self.padding, self.padding), (0, 0)), mode='constant')
            dX_padded = np.zeros_like(X_padded)
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
                        self.d_filters[:, :, :, f] += receptive_field * dZ[i, h, w, f]
                        
                        # Gradient for input
                        dX_padded[i, vert_start:vert_end, horiz_start:horiz_end, :] += self.filters[:, :, :, f] * dZ[i, h, w, f]
        
        # Remove padding from dX_padded if needed
        if self.padding > 0:
            dX = dX_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX = dX_padded
        
        # Gradient for bias (sum over all dimensions except filters)
        self.d_bias = np.sum(dZ, axis=(0, 1, 2)).reshape(1, 1, 1, -1)
        
        return dX
    
    def update(self, learning_rate):
        """Update parameters using gradients"""
        if self.d_filters is not None:
            self.filters -= learning_rate * self.d_filters
        if self.d_bias is not None:
            self.bias -= learning_rate * self.d_bias

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
        
        # Calculate output dimensions - FIXED FORMULA
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        self.output = np.zeros((batch_size, out_height, out_width, channels))
        self.mask = np.zeros_like(X, dtype=float)
        
        for i in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(channels):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.pool_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.pool_size
                        
                        # Ensure we don't go out of bounds
                        if vert_end <= height and horiz_end <= width:
                            window = X[i, vert_start:vert_end, horiz_start:horiz_end, c]
                            if window.size > 0:  # Check if window is not empty
                                self.output[i, h, w, c] = np.max(window)
                                
                                # Create mask for backward pass
                                max_pos = np.unravel_index(np.argmax(window), window.shape)
                                self.mask[i, vert_start + max_pos[0], horiz_start + max_pos[1], c] = 1
        
        return self.output
    
    def backward(self, dZ):
        dX = np.zeros_like(self.input, dtype=float)  
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
                        if vert_end <= self.input.shape[1] and horiz_end <= self.input.shape[2]:
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