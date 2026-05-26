import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


print("Train:", x_train.shape, y_train.shape)
print("Test:", x_test.shape, y_test.shape)


x_train = x_train / 255.0
x_test = x_test / 255.0

print("Data loaded and normalized successfully!")