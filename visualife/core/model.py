import numpy as np
import pickle
from visualife.core.losses import CrossEntropyLoss
from visualife.core.optimizers import SGD, Momentum, Adam
from visualife.core.callbacks import EarlyStopping, LearningRateScheduler
from visualife.core.layers import Dropout

class Model:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.history = {'train_loss': [], 'train_accuracy': [],
                        'val_loss': [], 'val_accuracy': [],
                        'learning_rate': []}
        self.current_epoch = 0

    def add(self, layer):
        self.layers.append(layer)
        print(f"Added {layer.__class__.__name__} layer")

    def compile(self, loss='categorical_crossentropy',
                optimizer='adam', learning_rate=0.001, **optimizer_params):
        """
        Compile the model with loss function and optimizer.
        
        Args:
            loss: Loss function ('categorical_crossentropy', 'mse', etc.)
            optimizer: Optimizer ('sgd', 'momentum', 'adam') - defaults to 'adam'
            learning_rate: Learning rate for optimizer
            **optimizer_params: Additional optimizer parameters
        """
        
        # Set default optimizer if None provided
        if optimizer is None:
            optimizer = 'adam'
            print("Using default optimizer: Adam")
        
        # Loss function
        if loss == 'categorical_crossentropy':
            self.loss_function = CrossEntropyLoss()
        else:
            from visualife.core.losses import MeanSquaredError, BinaryCrossEntropy
            if loss == 'mean_squared_error':
                self.loss_function = MeanSquaredError()
            elif loss == 'binary_crossentropy':
                self.loss_function = BinaryCrossEntropy()
            else:
                raise ValueError(f"Unsupported loss: {loss}")

        # Optimizer with proper learning rate
        if optimizer == 'sgd':
            self.optimizer = SGD(learning_rate, **optimizer_params)
        elif optimizer == 'momentum':
            self.optimizer = Momentum(learning_rate, **optimizer_params)
        elif optimizer == 'adam':
            self.optimizer = Adam(learning_rate, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        print(f"Model compiled with {loss}, optimizer={optimizer}, lr={learning_rate}")

    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training
            output = layer.forward(output)
        return output

    def backward(self, dZ):
        """Perform backward pass and update parameters using optimizer."""
        gradient = dZ
        
        # Backward pass through all layers
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                # ALWAYS pass learning_rate to maintain compatibility
                gradient = layer.backward(gradient, self.optimizer.learning_rate)
        
        # Update parameters using optimizer
        for layer in self.layers:
            if (hasattr(layer, 'weights') and 
                hasattr(layer, 'dweights') and 
                layer.dweights is not None):
                self.optimizer.update(layer)
        
        return gradient

    def compute_loss(self, y_pred, y_true):
        loss = self.loss_function.forward(y_pred, y_true)
        reg_loss = sum(layer.regularization_loss() for layer in self.layers if hasattr(layer, 'regularization_loss'))
        loss += reg_loss
        dZ = self.loss_function.backward()
        return loss, dZ

    def train_step(self, X_batch, y_batch):
        y_pred = self.forward(X_batch, training=True)
        loss, dZ = self.compute_loss(y_pred, y_batch)
        self.backward(dZ)

        if y_batch.shape[1] > 1:
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            accuracy = np.mean(predictions == true_labels)
        else:
            accuracy = np.mean((y_pred > 0.5) == (y_batch > 0.5))
        return loss, accuracy

    def fit(self, X_train, y_train, epochs=10, batch_size=32,
            validation_data=None, callbacks=None, verbose=1):
        """
        Train the model and return training history
        """
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        # Reset history for new training session
        self.history = {
            'train_loss': [], 
            'train_accuracy': [],
            'val_loss': [], 
            'val_accuracy': [],
            'learning_rate': []
        }
        
        callbacks = callbacks or []
        early_stopper = None
        lr_scheduler = None
        for callback in callbacks:
            if isinstance(callback, EarlyStopping):
                early_stopper = callback
            elif isinstance(callback, LearningRateScheduler):
                lr_scheduler = callback

        for epoch in range(epochs):
            self.current_epoch = epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            epoch_accuracy = 0

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                batch_loss, batch_accuracy = self.train_step(X_batch, y_batch)
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy

            epoch_loss /= n_batches
            epoch_accuracy /= n_batches
            self.history['train_loss'].append(epoch_loss)
            self.history['train_accuracy'].append(epoch_accuracy)
            self.history['learning_rate'].append(self.optimizer.learning_rate)

            val_loss, val_accuracy = None, None
            if validation_data:
                X_val, y_val = validation_data
                val_loss, val_accuracy = self.evaluate(X_val, y_val, verbose=0)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

            if verbose >= 1:
                val_info = f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}" if validation_data else ""
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, LR: {self.optimizer.learning_rate:.6f}{val_info}")

            stop_training = False
            for callback in callbacks:
                if isinstance(callback, EarlyStopping) and val_loss is not None:
                    if callback.on_epoch_end(epoch, val_loss, self):
                        stop_training = True
                elif isinstance(callback, LearningRateScheduler) and val_loss is not None:
                    callback.on_epoch_end(epoch, val_loss, self, self.optimizer)
            if stop_training:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return History(self.history)

    def evaluate(self, X, y, verbose=1):
        y_pred = self.forward(X, training=False)
        loss = self.loss_function.forward(y_pred, y)
        reg_loss = sum(layer.regularization_loss() for layer in self.layers if hasattr(layer, 'regularization_loss'))
        loss += reg_loss

        if y.shape[1] > 1:
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels)
        else:
            accuracy = np.mean((y_pred > 0.5) == (y > 0.5))

        if verbose:
            print(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def predict(self, X):
        return self.forward(X, training=False)

    def predict_proba(self, X):
        """Return probability predictions"""
        return self.forward(X, training=False)

    def predict_classes(self, X):
        """Return class predictions"""
        y_pred = self.forward(X, training=False)
        if y_pred.shape[1] > 1:
            return np.argmax(y_pred, axis=1)
        else:
            return (y_pred > 0.5).astype(int)

    def save(self, filepath):
        model_data = {
            'layer_weights': [],
            'model_config': {
                'learning_rate': self.optimizer.learning_rate,
                'history': self.history
            }
        }
        for layer in self.layers:
            layer_data = {}
            if hasattr(layer, 'weights'):
                layer_data['weights'] = layer.weights.copy() if layer.weights is not None else None
            if hasattr(layer, 'biases'):
                layer_data['biases'] = layer.biases.copy() if layer.biases is not None else None
            if hasattr(layer, 'gamma'):
                layer_data['gamma'] = layer.gamma.copy() if layer.gamma is not None else None
                layer_data['beta'] = layer.beta.copy() if layer.beta is not None else None
                layer_data['running_mean'] = layer.running_mean.copy() if layer.running_mean is not None else None
                layer_data['running_var'] = layer.running_var.copy() if layer.running_var is not None else None
            model_data['layer_weights'].append(layer_data)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to {filepath}")

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        for layer, layer_data in zip(self.layers, model_data['layer_weights']):
            if hasattr(layer, 'weights') and layer_data.get('weights') is not None:
                layer.weights = layer_data['weights']
            if hasattr(layer, 'biases') and layer_data.get('biases') is not None:
                layer.biases = layer_data['biases']
            if hasattr(layer, 'gamma') and layer_data.get('gamma') is not None:
                layer.gamma = layer_data['gamma']
                layer.beta = layer_data['beta']
                layer.running_mean = layer_data['running_mean']
                layer.running_var = layer_data['running_var']
        if 'model_config' in model_data:
            self.optimizer.learning_rate = model_data['model_config'].get('learning_rate', 0.01)
            self.history = model_data['model_config'].get('history', {})

    def summary(self):
        print("\n" + "="*60)
        print("Model Summary")
        print("="*60)
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            params = 0
            if hasattr(layer, 'weights') and layer.weights is not None:
                params += layer.weights.size
            if hasattr(layer, 'biases') and layer.biases is not None:
                params += layer.biases.size
            if hasattr(layer, 'gamma') and layer.gamma is not None:
                params += layer.gamma.size + layer.beta.size
            total_params += params
            output_shape = "?"
            if hasattr(layer, 'weights') and layer.weights is not None:
                output_shape = f"(?, {layer.weights.shape[1]})"
            elif isinstance(layer, Dropout):
                output_shape = "Same as input"
            print(f"{i+1:2d}. {layer_name:15} | Output: {output_shape:15} | Params: {params:>8,}")
        print("="*60)
        print(f"Total Parameters: {total_params:,}")
        print("="*60)

    def get_weights(self):
        """Get all weights and biases from the model"""
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                weights.append(layer.weights.copy())
            if hasattr(layer, 'biases'):
                weights.append(layer.biases.copy())
        return weights

    def set_weights(self, weights):
        """Set weights and biases for the model"""
        weight_idx = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = weights[weight_idx].copy()
                weight_idx += 1
            if hasattr(layer, 'biases'):
                layer.biases = weights[weight_idx].copy()
                weight_idx += 1

    def get_config(self):
        """Get model configuration"""
        return {
            'layers': [layer.__class__.__name__ for layer in self.layers],
            'loss': self.loss_function.__class__.__name__,
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.optimizer.learning_rate
        }