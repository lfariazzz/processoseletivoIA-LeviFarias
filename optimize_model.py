import tensorflow as tf
import os
import numpy as np

print("Carregando modelo treinado (model.h5)...")
model = tf.keras.models.load_model("model.h5")
print("  Modelo carregado com sucesso.\n")

print("Aplicando Dynamic Range Quantization")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

size_kb = os.path.getsize("model.tflite") / 1024
print(f"  Salvo: model.tflite  |  Tamanho: {size_kb:.1f} KB\n")

print("Validando inferência")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = (x_test.astype("float32") / 255.0)[..., tf.newaxis]

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

correct = 0
for i in range(100):
    sample = np.expand_dims(x_test[i], axis=0)
    interpreter.set_tensor(input_details[0]["index"], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    if np.argmax(output) == y_test[i]:
        correct += 1

acc = correct / 100 * 100
print(f"{'='*50}")
print(f"  Modelo : model.tflite")
print(f"  Tamanho: {size_kb:.1f} KB")
print(f"  Acurácia: {acc:.1f} %")
print(f"{'='*50}")
print("\nOtimização concluída!")