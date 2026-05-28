"""Grid search over learning rate, batch size, and optimizer for the CIFAR-10 CNN.

This is a *relative* search: each configuration trains a lightweight version of
the architecture for a few epochs so the sweep stays tractable on CPU. The goal
is to compare hyperparameters against each other, not to reach final accuracy
(that is what train.py does, for many more epochs and with augmentation).

Results are written to:
  * results/hyperparameter_results.csv  - full sortable table
  * results/hyperparameter_results.md   - ranked markdown summary for the README

Override the cost via env vars, e.g. EPOCHS=2 to speed the sweep up.
"""

import os
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = int(os.environ.get("EPOCHS", 3))

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

learning_rates = [0.001, 0.0001, 0.00001]
batch_sizes = [32, 64, 128]
optimizers_config = ["adam", "sgd", "rmsprop"]

results = []


def build_search_model():
    """Lightweight single-conv-per-block model used only for the sweep."""
    return keras.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )


def make_optimizer(name, lr):
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)
    return keras.optimizers.RMSprop(learning_rate=lr)


for lr in learning_rates:
    for batch_size in batch_sizes:
        for opt_name in optimizers_config:
            print(f"\nTesting: LR={lr}, Batch={batch_size}, Optimizer={opt_name}")

            model = build_search_model()
            model.compile(
                optimizer=make_optimizer(opt_name, lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=EPOCHS,
                validation_split=0.2,
                verbose=1,
            )

            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            results.append(
                {
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "optimizer": opt_name,
                    "accuracy": test_acc,
                    "loss": test_loss,
                }
            )
            print(f"Test Accuracy: {test_acc:.4f}  Test Loss: {test_loss:.4f}")


sorted_results = sorted(results, key=lambda r: r["accuracy"], reverse=True)

print("\n=== Hyperparameter Tuning Results (ranked) ===")
for r in sorted_results:
    print(
        f"LR: {r['learning_rate']}, Batch: {r['batch_size']}, "
        f"Optimizer: {r['optimizer']} -> Accuracy: {r['accuracy']:.4f}, "
        f"Loss: {r['loss']:.4f}"
    )

best = sorted_results[0]
print("\n=== BEST CONFIGURATION ===")
print(f"Learning Rate: {best['learning_rate']}")
print(f"Batch Size:    {best['batch_size']}")
print(f"Optimizer:     {best['optimizer']}")
print(f"Accuracy:      {best['accuracy']:.4f}")
print(f"Loss:          {best['loss']:.4f}")

# ----------------------------------------------------------------------------
# Persist results
# ----------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

csv_path = "results/hyperparameter_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["learning_rate", "batch_size", "optimizer", "accuracy", "loss"]
    )
    writer.writeheader()
    for r in sorted_results:
        writer.writerow(r)
print(f"\nSaved {csv_path}")

md_path = "results/hyperparameter_results.md"
with open(md_path, "w") as f:
    f.write(f"# Hyperparameter Tuning Results\n\n")
    f.write(f"Search: {len(sorted_results)} configurations, {EPOCHS} epochs each "
            f"(lightweight proxy model).\n\n")
    f.write("| Rank | Learning Rate | Batch Size | Optimizer | Accuracy | Loss |\n")
    f.write("|------|---------------|------------|-----------|----------|------|\n")
    for i, r in enumerate(sorted_results, 1):
        f.write(
            f"| {i} | {r['learning_rate']} | {r['batch_size']} | "
            f"{r['optimizer']} | {r['accuracy']:.4f} | {r['loss']:.4f} |\n"
        )
    f.write(
        f"\n**Best configuration:** optimizer=`{best['optimizer']}`, "
        f"learning_rate=`{best['learning_rate']}`, batch_size=`{best['batch_size']}` "
        f"(test accuracy {best['accuracy']:.4f}).\n"
    )
print(f"Saved {md_path}")
