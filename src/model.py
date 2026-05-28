"""CNN architecture for CIFAR-10 classification.

A VGG-style network with three double-convolution blocks. Each block doubles
the filter count (32 -> 64 -> 128), applies BatchNormalization after every
convolution to stabilise training, downsamples with MaxPooling2D, and uses
progressively stronger spatial dropout to regularise. GlobalAveragePooling
replaces a large Flatten+Dense stack, which sharply cuts parameter count and
overfitting compared with the original single-conv-per-block design.
"""

from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)


def build_model(num_classes: int = NUM_CLASSES, input_shape=INPUT_SHAPE) -> keras.Model:
    """Build and return the (uncompiled) CIFAR-10 CNN."""
    model = keras.Sequential(name="cifar10_cnn")
    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    # Classifier head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def build_compiled_model(learning_rate: float = 1e-3) -> keras.Model:
    """Build the model already compiled with Adam + sparse CE loss."""
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    build_compiled_model().summary()
