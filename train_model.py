import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

print("Carregando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

x_train = x_train[..., tf.newaxis]
x_test  = x_test[...,  tf.newaxis]

print(f"  Treino : {x_train.shape} | Teste: {x_test.shape}")


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
], name="mnist_cnn_edge")

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nIniciando treinamento...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1,
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n{'='*50}")
print(f"  Acurácia nos testes : {test_acc * 100:.2f} %")
print(f"  Loss nos testes     : {test_loss:.4f}")
print(f"{'='*50}")

val_acc_final = history.history["val_accuracy"][-1]
print(f"  Acurácia de validação (última época): {val_acc_final * 100:.2f} %")
print(f"{'='*50}\n")

model.save("model.h5")
print("Modelo salvo: model.h5")
