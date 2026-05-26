import tensorflow as tf
from tensorflow import keras
from model import model  


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)


test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)


model.save("models/cifar10_model.keras")