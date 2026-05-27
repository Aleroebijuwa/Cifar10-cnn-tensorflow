import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


learning_rates = [0.001, 0.0001, 0.00001]
batch_sizes = [32, 64, 128]
optimizers_config = ['adam', 'sgd', 'rmsprop']

results = []


def build_model():

    model = keras.Sequential([

        layers.Conv2D(32, (3,3), activation='relu',
                      padding='same',
                      input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu',
                      padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu',
                      padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    return model


for lr in learning_rates:

    for batch_size in batch_sizes:

        for opt_name in optimizers_config:

            print(f"\nTesting: LR={lr}, Batch={batch_size}, Optimizer={opt_name}")

            model = build_model()

            if opt_name == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=lr)

            elif opt_name == 'sgd':
                optimizer = keras.optimizers.SGD(learning_rate=lr)

            else:
                optimizer = keras.optimizers.RMSprop(learning_rate=lr)

            # Compile model
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Train model (keep epochs small for speed)
            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=3,
                validation_split=0.2,
                verbose=1
            )

            # Evaluate model
            test_loss, test_acc = model.evaluate(
                x_test,
                y_test,
                verbose=0
            )

            # Save results
            results.append({
                'learning_rate': lr,
                'batch_size': batch_size,
                'optimizer': opt_name,
                'accuracy': test_acc,
                'loss': test_loss
            })

            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}")


print("\n=== Hyperparameter Tuning Results ===")

sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

for r in sorted_results:

    print(
        f"LR: {r['learning_rate']}, "
        f"Batch: {r['batch_size']}, "
        f"Optimizer: {r['optimizer']} "
        f"-> Accuracy: {r['accuracy']:.4f}, "
        f"Loss: {r['loss']:.4f}"
    )


best = sorted_results[0]

print("\n=== BEST CONFIGURATION ===")
print(f"Learning Rate: {best['learning_rate']}")
print(f"Batch Size: {best['batch_size']}")
print(f"Optimizer: {best['optimizer']}")
print(f"Accuracy: {best['accuracy']:.4f}")
print(f"Loss: {best['loss']:.4f}")