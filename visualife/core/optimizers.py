import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.name = 'SGD'
    
    def update(self, layer):
        """Update layer parameters using SGD"""
        # Update weights if they exist and have gradients
        if hasattr(layer, 'weights') and hasattr(layer, 'dweights') and layer.dweights is not None:
            layer.weights -= self.learning_rate * layer.dweights
        
        # Update biases if they exist and have gradients
        if hasattr(layer, 'biases') and hasattr(layer, 'dbiases') and layer.dbiases is not None:
            layer.biases -= self.learning_rate * layer.dbiases
        
        # Update BatchNorm gamma if they exist and have gradients
        if hasattr(layer, 'gamma') and hasattr(layer, 'dgamma') and layer.dgamma is not None:
            layer.gamma -= self.learning_rate * layer.dgamma
        
        # Update BatchNorm beta if they exist and have gradients
        if hasattr(layer, 'beta') and hasattr(layer, 'dbeta') and layer.dbeta is not None:
            layer.beta -= self.learning_rate * layer.dbeta


class Momentum:
    """SGD with Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}  # Store velocity for each layer
        self.name = 'Momentum'
    
    def update(self, layer):
        layer_id = id(layer)
        
        # Initialize velocity for this layer if not exists
        if layer_id not in self.velocity:
            self.velocity[layer_id] = {}
        
        # Update weights with momentum
        if hasattr(layer, 'weights') and hasattr(layer, 'dweights') and layer.dweights is not None:
            if 'weights' not in self.velocity[layer_id]:
                self.velocity[layer_id]['weights'] = np.zeros_like(layer.weights)
            
            # Momentum update: v = momentum * v - learning_rate * gradient
            self.velocity[layer_id]['weights'] = (self.momentum * self.velocity[layer_id]['weights'] 
                                                - self.learning_rate * layer.dweights)
            layer.weights += self.velocity[layer_id]['weights']
        
        # Update biases with momentum
        if hasattr(layer, 'biases') and hasattr(layer, 'dbiases') and layer.dbiases is not None:
            if 'biases' not in self.velocity[layer_id]:
                self.velocity[layer_id]['biases'] = np.zeros_like(layer.biases)
            
            self.velocity[layer_id]['biases'] = (self.momentum * self.velocity[layer_id]['biases'] 
                                               - self.learning_rate * layer.dbiases)
            layer.biases += self.velocity[layer_id]['biases']
        
        # Update BatchNorm parameters with momentum
        if hasattr(layer, 'gamma') and hasattr(layer, 'dgamma') and layer.dgamma is not None:
            if 'gamma' not in self.velocity[layer_id]:
                self.velocity[layer_id]['gamma'] = np.zeros_like(layer.gamma)
            
            self.velocity[layer_id]['gamma'] = (self.momentum * self.velocity[layer_id]['gamma'] 
                                              - self.learning_rate * layer.dgamma)
            layer.gamma += self.velocity[layer_id]['gamma']
        
        if hasattr(layer, 'beta') and hasattr(layer, 'dbeta') and layer.dbeta is not None:
            if 'beta' not in self.velocity[layer_id]:
                self.velocity[layer_id]['beta'] = np.zeros_like(layer.beta)
            
            self.velocity[layer_id]['beta'] = (self.momentum * self.velocity[layer_id]['beta'] 
                                             - self.learning_rate * layer.dbeta)
            layer.beta += self.velocity[layer_id]['beta']


class Adam:
    """Adam optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step
        self.name = 'Adam'
    
    def update(self, layer):
        layer_id = id(layer)
        self.t += 1
        
        # Initialize moments for this layer if not exists
        if layer_id not in self.m:
            self.m[layer_id] = {}
            self.v[layer_id] = {}
        
        # Update weights with Adam
        if hasattr(layer, 'weights') and hasattr(layer, 'dweights') and layer.dweights is not None:
            if 'weights' not in self.m[layer_id]:
                self.m[layer_id]['weights'] = np.zeros_like(layer.weights)
                self.v[layer_id]['weights'] = np.zeros_like(layer.weights)
            
            # Update biased first moment estimate
            self.m[layer_id]['weights'] = (self.beta1 * self.m[layer_id]['weights'] 
                                         + (1 - self.beta1) * layer.dweights)
            
            # Update biased second raw moment estimate
            self.v[layer_id]['weights'] = (self.beta2 * self.v[layer_id]['weights'] 
                                         + (1 - self.beta2) * (layer.dweights ** 2))
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[layer_id]['weights'] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[layer_id]['weights'] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Update biases with Adam
        if hasattr(layer, 'biases') and hasattr(layer, 'dbiases') and layer.dbiases is not None:
            if 'biases' not in self.m[layer_id]:
                self.m[layer_id]['biases'] = np.zeros_like(layer.biases)
                self.v[layer_id]['biases'] = np.zeros_like(layer.biases)
            
            self.m[layer_id]['biases'] = (self.beta1 * self.m[layer_id]['biases'] 
                                        + (1 - self.beta1) * layer.dbiases)
            self.v[layer_id]['biases'] = (self.beta2 * self.v[layer_id]['biases'] 
                                        + (1 - self.beta2) * (layer.dbiases ** 2))
            
            m_hat = self.m[layer_id]['biases'] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer_id]['biases'] / (1 - self.beta2 ** self.t)
            layer.biases -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Update BatchNorm parameters with Adam
        if hasattr(layer, 'gamma') and hasattr(layer, 'dgamma') and layer.dgamma is not None:
            if 'gamma' not in self.m[layer_id]:
                self.m[layer_id]['gamma'] = np.zeros_like(layer.gamma)
                self.v[layer_id]['gamma'] = np.zeros_like(layer.gamma)
            
            self.m[layer_id]['gamma'] = (self.beta1 * self.m[layer_id]['gamma'] 
                                       + (1 - self.beta1) * layer.dgamma)
            self.v[layer_id]['gamma'] = (self.beta2 * self.v[layer_id]['gamma'] 
                                       + (1 - self.beta2) * (layer.dgamma ** 2))
            
            m_hat = self.m[layer_id]['gamma'] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer_id]['gamma'] / (1 - self.beta2 ** self.t)
            layer.gamma -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        if hasattr(layer, 'beta') and hasattr(layer, 'dbeta') and layer.dbeta is not None:
            if 'beta' not in self.m[layer_id]:
                self.m[layer_id]['beta'] = np.zeros_like(layer.beta)
                self.v[layer_id]['beta'] = np.zeros_like(layer.beta)
            
            self.m[layer_id]['beta'] = (self.beta1 * self.m[layer_id]['beta'] 
                                      + (1 - self.beta1) * layer.dbeta)
            self.v[layer_id]['beta'] = (self.beta2 * self.v[layer_id]['beta'] 
                                      + (1 - self.beta2) * (layer.dbeta ** 2))
            
            m_hat = self.m[layer_id]['beta'] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer_id]['beta'] / (1 - self.beta2 ** self.t)
            layer.beta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)