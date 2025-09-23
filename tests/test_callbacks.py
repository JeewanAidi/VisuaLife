"""
Tests for Callbacks
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from visualife.core.callbacks import EarlyStopping, LearningRateScheduler
from visualife.core.layers import Dense

# Mock model to use with EarlyStopping
class MockModel:
    def __init__(self):
        self.layers = [Dense(2, 2)]

def test_early_stopping():
    print("🧪 Testing Early Stopping...")

    model = MockModel()
    early_stop = EarlyStopping(patience=2, min_delta=0.1)

    # Simulate improving and worsening loss
    assert not early_stop.on_epoch_end(0, 1.0, model)   # First epoch
    assert not early_stop.on_epoch_end(1, 0.8, model)   # Improved
    assert not early_stop.on_epoch_end(2, 0.9, model)   # Worse, but patience=2
    assert early_stop.on_epoch_end(3, 0.95, model)      # Worse again, should stop

    print("✅ Early stopping test passed!")

def test_lr_scheduler():
    print("🧪 Testing Learning Rate Scheduler...")

    lr_scheduler = LearningRateScheduler(factor=0.5, patience=1)

    class MockOptimizer:
        def __init__(self):
            self.learning_rate = 0.01

    model = MockModel()
    optimizer = MockOptimizer()

    # Simulate no improvement
    lr_scheduler.on_epoch_end(0, 1.0, model, optimizer)
    lr_scheduler.on_epoch_end(1, 1.0, model, optimizer)  # No improvement → should reduce LR

    assert optimizer.learning_rate == 0.005, "LR scheduler did not reduce learning rate!"
    print("✅ Learning rate scheduler test passed!")

if __name__ == "__main__":
    print("🚀 Testing Callbacks")
    print("===================")
    test_early_stopping()
    test_lr_scheduler()
    print("🎉 All callback tests passed!")
