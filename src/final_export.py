import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


model = tf.keras.models.load_model("models/cifar10_model.keras")


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32") / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


os.makedirs("saved_model", exist_ok=True)
os.makedirs("results", exist_ok=True)


# Export to TensorFlow SavedModel format (produces saved_model.pb)
model.export("saved_model/cifar10_cnn")
print("Model exported to SavedModel format at saved_model/cifar10_cnn/")


# Training history plot (uses history.npy saved by train.py)
if os.path.exists("results/history.npy"):
    history = np.load("results/history.npy", allow_pickle=True).item()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_history.png", dpi=300)
    plt.close()
    print("Saved results/training_history.png")
else:
    print("results/history.npy not found - skipping training history plot. "
          "Re-run train.py to generate it.")


# Evaluate + predictions
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = y_test.flatten()

precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1 Score:      {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()
print("Saved results/confusion_matrix.png")


# Performance dashboard (single figure with key metrics)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

per_class_acc = cm.diagonal() / cm.sum(axis=1)
axes[0].barh(class_names, per_class_acc, color='steelblue')
axes[0].set_xlim(0, 1)
axes[0].set_xlabel("Accuracy")
axes[0].set_title("Per-Class Accuracy")
for i, v in enumerate(per_class_acc):
    axes[0].text(v + 0.01, i, f"{v:.2f}", va='center')

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
metrics_vals = [test_acc, precision, recall, f1]
axes[1].bar(metrics_names, metrics_vals, color=['#4c72b0', '#55a868', '#c44e52', '#8172b2'])
axes[1].set_ylim(0, 1)
axes[1].set_title("Overall Metrics")
for i, v in enumerate(metrics_vals):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha='center')

plt.tight_layout()
plt.savefig("results/performance_dashboard.png", dpi=300)
plt.close()
print("Saved results/performance_dashboard.png")


# Metrics summary text file
with open("results/metrics.txt", "w") as f:
    f.write("CIFAR-10 CNN - Final Metrics\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss:     {test_loss:.4f}\n")
    f.write(f"Precision:     {precision:.4f}\n")
    f.write(f"Recall:        {recall:.4f}\n")
    f.write(f"F1 Score:      {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names))

print("Saved results/metrics.txt")
print("\nAll results saved successfully!")
