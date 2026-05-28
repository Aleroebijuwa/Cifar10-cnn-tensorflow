"""Train the CIFAR-10 CNN with on-the-fly augmentation and callbacks.

Augmentation is applied through Keras preprocessing layers inside a tf.data
pipeline (ImageDataGenerator was removed in Keras 3). Training uses:
  * EarlyStopping     - stop once val_loss stops improving, restore best weights
  * ReduceLROnPlateau - decay the learning rate when val_loss plateaus
  * ModelCheckpoint   - keep the best model by validation accuracy

Set the EPOCHS env var to override the default (e.g. EPOCHS=5 for a smoke test).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib
matplotlib.use("Agg")  # headless-safe (works when run in the background)
import matplotlib.pyplot as plt

from model import build_compiled_model

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = int(os.environ.get("EPOCHS", 60))
VAL_SIZE = 5000
SEED = 42
LEARNING_RATE = 1e-3

MODEL_PATH = "models/cifar10_model.keras"

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_val, y_val = x_train[-VAL_SIZE:], y_train[-VAL_SIZE:]
x_tr, y_tr = x_train[:-VAL_SIZE], y_train[:-VAL_SIZE]

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),       # ~ +/- 15 degrees
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
    .shuffle(10_000, seed=SEED)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ----------------------------------------------------------------------------
# Model + callbacks
# ----------------------------------------------------------------------------
model = build_compiled_model(learning_rate=LEARNING_RATE)
model.summary()

os.makedirs("models", exist_ok=True)
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
)

# ----------------------------------------------------------------------------
# Evaluate + persist
# ----------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Epochs run: {len(history.history['loss'])} (early stopping may have triggered)")

# Best weights are already restored; save the final model too.
model.save(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")

os.makedirs("results", exist_ok=True)
np.save("results/history.npy", history.history)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Over Time")

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Over Time")

plt.tight_layout()
plt.savefig("results/training_history.png", dpi=300)
plt.close()
print("Saved results/training_history.png")
