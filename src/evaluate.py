import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_test = x_test / 255.0

model = tf.keras.models.load_model("../models/cifar10_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_acc)