import copy

class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.
    Keeps best weights and can restore them.
    """
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, loss, model):
        # require a real model to save weights
        if model is None:
            # nothing to save/restore — just use patience logic
            if loss < self.best_loss - self.min_delta:
                self.best_loss = loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    return True
            return False

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self._get_model_weights(model)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None:
                    self._set_model_weights(model, self.best_weights)
                return True  # Stop training
        return False  # Continue training

    def _get_model_weights(self, model):
        weights = []
        for layer in model.layers:
            if hasattr(layer, 'weights'):
                layer_weights = {
                    'weights': layer.weights.copy() if layer.weights is not None else None,
                    'biases': layer.biases.copy() if hasattr(layer, 'biases') and layer.biases is not None else None
                }
                weights.append(layer_weights)
        return weights

    def _set_model_weights(self, model, weights):
        for layer, layer_weights in zip(model.layers, weights):
            if hasattr(layer, 'weights') and layer_weights['weights'] is not None:
                layer.weights = layer_weights['weights']
            if hasattr(layer, 'biases') and layer_weights['biases'] is not None:
                layer.biases = layer_weights['biases']

class LearningRateScheduler:
    """
    Learning rate scheduler that reduces LR on plateau.
    """
    def __init__(self, factor=0.5, patience=3, min_lr=1e-6):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, loss, model, optimizer):
        if loss < self.best_loss - 1e-4:  # Small delta for LR scheduling
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(optimizer.learning_rate * self.factor, self.min_lr)
                if new_lr != optimizer.learning_rate:
                    print(f"Reducing learning rate from {optimizer.learning_rate:.6f} to {new_lr:.6f}")
                    optimizer.learning_rate = new_lr
                    self.wait = 0
