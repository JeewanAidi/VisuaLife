# mlp_mnist_vectorized_full.py
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# ---------------------
# Layers (vectorized)
# ---------------------
class Dense:
    def __init__(self, input_dim, output_dim):
        # He / Xavier style initialization
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.biases = np.zeros((1, output_dim))
        # placeholders for gradients
        self.dW = None
        self.db = None
        self.X = None

    def forward(self, X):
        # X shape: (batch, input_dim)
        self.X = X
        return X @ self.weights + self.biases  # shape: (batch, output_dim)

    def backward(self, grad_output):
        # grad_output shape: (batch, output_dim)
        m = grad_output.shape[0]
        # compute gradients (averaged over batch)
        self.dW = (self.X.T @ grad_output) / m       # (input_dim, output_dim)
        self.db = np.sum(grad_output, axis=0, keepdims=True) / m  # (1, output_dim)
        # grad wrt input to pass to previous layer
        dX = grad_output @ self.weights.T            # (batch, input_dim)
        return dX


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X > 0).astype(X.dtype)
        return X * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


# ---------------------
# Loss (softmax + cross-entropy)
# ---------------------
class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        """
        logits: (batch, classes)
        y_true: (batch, classes) one-hot
        returns scalar loss (mean over batch)
        """
        # numeric stability
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.y_true = y_true
        # cross-entropy loss
        m = logits.shape[0]
        eps = 1e-12
        loss = -np.sum(y_true * np.log(self.probs + eps)) / m
        return loss

    def backward(self):
        # gradient w.r.t logits (already averaged)
        m = self.y_true.shape[0]
        return (self.probs - self.y_true) / m


# ---------------------
# Optimizer: Adam (vectorized)
# ---------------------
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        # use layer id keys
        self.m_w = {}  # first moment for weights (key: id(layer))
        self.v_w = {}  # second moment
        self.m_b = {}
        self.v_b = {}

    def update(self, layer):
        """
        Update parameters of a Dense layer in-place.
        Expects layer.dW and layer.db (already computed and averaged).
        """
        lid = id(layer)
        if not hasattr(layer, 'weights'):
            return

        # initialize moments if needed
        if lid not in self.m_w:
            self.m_w[lid] = np.zeros_like(layer.weights)
            self.v_w[lid] = np.zeros_like(layer.weights)
            self.m_b[lid] = np.zeros_like(layer.biases)
            self.v_b[lid] = np.zeros_like(layer.biases)

        self.t += 1

        # weights
        g_w = layer.dW
        self.m_w[lid] = self.beta1 * self.m_w[lid] + (1 - self.beta1) * g_w
        self.v_w[lid] = self.beta2 * self.v_w[lid] + (1 - self.beta2) * (g_w ** 2)
        m_hat_w = self.m_w[lid] / (1 - self.beta1 ** self.t)
        v_hat_w = self.v_w[lid] / (1 - self.beta2 ** self.t)
        layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)

        # biases
        g_b = layer.db
        self.m_b[lid] = self.beta1 * self.m_b[lid] + (1 - self.beta1) * g_b
        self.v_b[lid] = self.beta2 * self.v_b[lid] + (1 - self.beta2) * (g_b ** 2)
        m_hat_b = self.m_b[lid] / (1 - self.beta1 ** self.t)
        v_hat_b = self.v_b[lid] / (1 - self.beta2 ** self.t)
        layer.biases -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


# ---------------------
# Model (vectorized training)
# ---------------------
class Model:
    def __init__(self):
        self.layers = []
        self.criterion = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)
        print(f"Added {layer.__class__.__name__} layer")

    def compile(self, loss='softmax_crossentropy', optimizer=None):
        if loss == 'softmax_crossentropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Only 'softmax_crossentropy' supported in this script")
        self.optimizer = optimizer if optimizer is not None else Adam()
        print(f"Model compiled with loss={loss}, optimizer={self.optimizer.__class__.__name__}, lr={self.optimizer.lr}")

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def train_on_batch(self, X_batch, y_batch):
        # forward
        logits = self.forward(X_batch)
        # loss
        loss = self.criterion.forward(logits, y_batch)
        # backward (grad w.r.t logits)
        grad_logits = self.criterion.backward()
        # propagate backwards and compute layer gradients
        self.backward(grad_logits)
        # update parameters (only Dense layers have weights)
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                self.optimizer.update(layer)
        # compute accuracy
        preds = np.argmax(logits, axis=1)
        truths = np.argmax(y_batch, axis=1)
        acc = np.mean(preds == truths)
        return loss, acc

    def fit(self, X_train, y_train, epochs=3, batch_size=64, X_val=None, y_val=None, verbose=1):
        n = X_train.shape[0]
        steps_per_epoch = int(np.ceil(n / batch_size))
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            # shuffle
            perm = np.random.permutation(n)
            X_shuff = X_train[perm]
            y_shuff = y_train[perm]
            epoch_loss = 0.0
            epoch_acc = 0.0
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = min(start + batch_size, n)
                X_batch = X_shuff[start:end]
                y_batch = y_shuff[start:end]
                loss, acc = self.train_on_batch(X_batch, y_batch)
                epoch_loss += loss
                epoch_acc += acc

                # progress print every 10%
                if step % max(1, steps_per_epoch // 10) == 0:
                    pct = int((step + 1) / steps_per_epoch * 100)
                    print(f"Epoch {epoch} progress: {pct}%  (step {step+1}/{steps_per_epoch})", end='\r')

            epoch_loss /= steps_per_epoch
            epoch_acc /= steps_per_epoch
            elapsed = time.time() - start_time

            val_info = ""
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val, verbose=0)
                val_info = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"

            if verbose:
                print(f"\nEpoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, LR: {self.optimizer.lr:.6f}{val_info} (time: {elapsed:.1f}s)")

    def predict(self, X):
        logits = self.forward(X)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs

    def evaluate(self, X, y, verbose=1):
        logits = self.forward(X)
        loss = self.criterion.forward(logits, y)
        preds = np.argmax(logits, axis=1)
        truths = np.argmax(y, axis=1)
        acc = np.mean(preds == truths)
        if verbose:
            print(f"Evaluation - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return loss, acc


# ---------------------
# Utility: one-hot encode
# ---------------------
def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]


# ---------------------
# Main: load MNIST, build, train
# ---------------------
def main():
    print("Loading MNIST (this may take ~10-30s the first time)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0
    y = y.astype(int)

    # reduce dataset for speed (5000 train, 1000 test)
    X_small, X_rest, y_small, y_rest = train_test_split(X, y, train_size=6000, random_state=42)
    X_train, X_test, y_train, y_test = X_small[:5000], X_small[5000:6000], y_small[:5000], y_small[5000:6000]
    # if above slicing produces less than desired just fallback
    if X_test.shape[0] < 1000:
        X_train, X_test, y_train, y_test = train_test_split(X[:6000], y[:6000], test_size=1000, random_state=42)
    y_train_oh = to_one_hot(y_train, 10)
    y_test_oh = to_one_hot(y_test, 10)

    # build model: 784 -> 64 -> 32 -> 10
    model = Model()
    model.add(Dense(784, 64))
    model.add(ReLU())
    model.add(Dense(64, 32))
    model.add(ReLU())
    model.add(Dense(32, 10))

    # compile with optimizer
    optimizer = Adam(lr=0.001) if hasattr(Adam, 'lr') else Adam(lr=0.001)  # compatibility
    model.compile(loss='softmax_crossentropy', optimizer=optimizer)

    # train
    epochs = 3
    batch_size = 64
    model.fit(X_train, y_train_oh, epochs=epochs, batch_size=batch_size, X_val=X_test, y_val=y_test_oh)

    # final evaluate
    print("\nFinal evaluation on test set:")
    model.evaluate(X_test, y_test_oh)

if __name__ == "__main__":
    main()
