import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {x_test.shape}")


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


print("Min pixel value:", x_train.min())
print("Max pixel value:", x_train.max())


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']



plt.figure(figsize=(10,5))

for i in range(10):
    idx = np.where(y_train.flatten() == i)[0][0]
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[idx])
    plt.title(class_names[i])
    plt.axis("off")

plt.tight_layout()
plt.show() 
