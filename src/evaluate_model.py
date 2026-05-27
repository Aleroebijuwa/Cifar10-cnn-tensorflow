import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_test = x_test.astype("float32") / 255.0


model = tf.keras.models.load_model("../models/cifar10_model.h5")


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()


class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']


print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))


cm = confusion_matrix(y_true, y_pred_classes)


plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")


plt.savefig("confusion_matrix.png")

plt.show()  
