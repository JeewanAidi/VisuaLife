import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from visualife.core.model import Model
from visualife.core.layers import Dense
from visualife.core.activations import ReLU, Softmax
from visualife.core.losses import CrossEntropyLoss
from visualife.core.optimizers import Adam

# ------------------------------
# Step 1: Load MNIST dataset
# ------------------------------
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # normalize
y = y.astype(int)

# One-hot encode labels
num_classes = 10
Y = np.eye(num_classes)[y]

# Reduce dataset to 10k train / 2k test for fast testing
X_train, X_test, Y_train, Y_test = train_test_split(X[:12000], Y[:12000], test_size=2000, random_state=42)

# ------------------------------
# Step 2: Build MLP Model
# ------------------------------
print("Building model...")
model = Model()
model.add(Dense(784, 128))
model.add(ReLU())
model.add(Dense(128, 64))
model.add(ReLU())
model.add(Dense(64, 10))
model.add(Softmax())

# Compile
loss = CrossEntropyLoss()
optimizer = Adam(learning_rate=0.001)
model.compile()

# ------------------------------
# Step 3: Train the Model
# ------------------------------
print("Training model...")
model.fit(X_train, Y_train, epochs=3, batch_size=64, validation_data=(X_test, Y_test))

# ------------------------------
# Step 4: Evaluate
# ------------------------------
print("Evaluating model...")
preds = model.predict(X_test)
acc = np.mean(np.argmax(preds, axis=1) == np.argmax(Y_test, axis=1))
print("Final Test Accuracy:", acc)
