import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator()

train_datagen.fit(x_train)


model = tf.keras.models.load_model("models/cifar10_model.keras")


history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(x_val, y_val),
    verbose=1
)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("\nFINAL RESULTS AFTER AUGMENTATION")
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)


augmented_images = train_datagen.flow(x_train[:1], batch_size=1)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i, ax in enumerate(axes):
    ax.imshow(next(augmented_images)[0])
    ax.axis("off")

plt.show()