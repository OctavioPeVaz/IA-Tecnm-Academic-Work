import os
import re

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf

import keras
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten

from keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D
)
from keras.layers import LeakyReLU
import time, os

datasetPath = '/mnt/d/Projects/tecnm_projects/IA-Tecnm-Academic-Work/datasets/animals_dataset'
destinationModelPath = '/mnt/d/Projects/tecnm_projects/IA-Tecnm-Academic-Work/trained_models/cnn/animals'

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Parámetros
batch_size = 64
img_height = 100
img_width = 100

# Crear un dataset de entrenamiento (80% de las imágenes)
train_dataset = tf.keras.utils.image_dataset_from_directory(
  datasetPath,
  validation_split=0.2,  # Reserva 20% para validación/testing
  subset="training",
  seed=1337, # Semilla para que la división sea reproducible
  image_size=(img_height, img_width),
  batch_size=batch_size,
  #label_mode='categorical'
)

# Crear un dataset de validación (el 20% restante)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
  datasetPath,
  validation_split=0.2,
  subset="validation",
  seed=1337,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Guardar los nombres de las clases (ej: ['cat', 'dog', 'ladybug'])
sriesgos = train_dataset.class_names
nClasses = len(sriesgos)
print(f"Clases encontradas: {sriesgos}")
print(f"Número de clases: {nClasses}")

# Normalización
# En lugar de convertir a float32 y dividir, lo hacemos con una capa en el modelo
# o usando la función .map() del dataset.

# Opción A (más fácil): Añadir la capa al modelo
# (Verás esto en el Paso 3)

# Opción B (más eficiente): Usar .map()
# Normaliza los datos de 0-255 a 0-1
def normalize_img(image, label):
    return (tf.cast(image, tf.float32) / 255.0), label

train_dataset = train_dataset.map(normalize_img)
validation_dataset = validation_dataset.map(normalize_img)

# Optimizar la carga de datos USANDO CACHÉ EN DISCO
AUTOTUNE = tf.data.AUTOTUNE

# Definimos nombres para los archivos temporales
cache_file_train = './trained_models/train_cache'
cache_file_val = './trained_models/val_cache'

# IMPORTANTE: Borrar caches viejos si existen.
# Si cambiaste el número de imágenes y TF intenta leer el cache viejo, fallará.
if os.path.exists(cache_file_train + ".index"):
    os.remove(cache_file_train + ".index")
    os.remove(cache_file_train + ".data-00000-of-00001")
if os.path.exists(cache_file_val + ".index"):
    os.remove(cache_file_val + ".index")
    os.remove(cache_file_val + ".data-00000-of-00001")

# Ahora usamos .cache(nombre_archivo) en lugar de .cache() vacío
train_dataset = train_dataset.cache(cache_file_train).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache(cache_file_val).prefetch(buffer_size=AUTOTUNE)

print("Configuración de caché en disco aplicada correctamente.")

print("Datasets de entrenamiento y validación creados exitosamente.")


# Define learning rate, Epoch

#declaramos variables con los parámetros de configuración de la red
INIT_LR = 1e-3 # Valor inicial de learning rate. El valor 1e-3 corresponde con 0.001
epochs = 43 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento
batch_size = 64 # cantidad de imágenes que se toman a la vez en memoria

# Define the CNN model using KERAS Api
# This may include convolutional layers, activation, pooling, normalization (Dropout) and Full Connected

riesgo_model = Sequential()

# Layer 1
riesgo_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(100,100,3)))
riesgo_model.add(LeakyReLU(alpha=0.1))
riesgo_model.add(MaxPooling2D((2, 2),padding='same'))

# Layer 2
riesgo_model.add(Conv2D(64, kernel_size=(3, 3),activation='linear',padding='same'))
riesgo_model.add(LeakyReLU(alpha=0.1))
riesgo_model.add(MaxPooling2D((2, 2),padding='same'))

# Layer 3
riesgo_model.add(Conv2D(128, kernel_size=(3, 3),activation='linear',padding='same'))
riesgo_model.add(LeakyReLU(alpha=0.1))
riesgo_model.add(MaxPooling2D((2, 2),padding='same'))

riesgo_model.add(Dropout(0.5))

riesgo_model.add(Flatten())

riesgo_model.add(Dense(32, activation='linear'))
riesgo_model.add(LeakyReLU(alpha=0.1))
riesgo_model.add(Dropout(0.5))

riesgo_model.add(Dense(nClasses, activation='softmax'))

riesgo_model.summary()


# Configure lose fucntion, optimizer and the metrics usedn on the model training
# This prepares the model to be trained based on the data
riesgo_model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)
# Set optimizer learning rate explicitly to match INIT_LR (avoids Pylance type issue)
riesgo_model.optimizer.learning_rate = INIT_LR


# Performs the real training of CNN using the prepared data
# The model auto adjust the to the data, minimizing the lose
print("Iniciando entrenamiento...")
start_time = time.time()  # <--- 1. TOMAS EL TIEMPO DE INICIO

riesgo_train = riesgo_model.fit(
    train_dataset, 
    batch_size=batch_size,
    epochs=epochs,
    verbose='1',
    validation_data=validation_dataset
)
end_time = time.time()

riesgo_model.save(os.path.join(destinationModelPath, '43-epochs_1e3_64-batch_3-layer.h5'))

# 3. CALCULAS Y MUESTRAS LA DIFERENCIA
total_time = end_time - start_time
minutes = int(total_time // 60)
seconds = int(total_time % 60)

print("\n" + "="*40)
print(f"Entrenamiento finalizado.")
print(f"Tiempo total: {minutes} min {seconds} s")
print("="*40 + "\n")


print("Evaluando el modelo con los datos de validación...")
test_eval = riesgo_model.evaluate(validation_dataset, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# 2. OBTENER ETIQUETAS Y PREDICCIONES (MODO EFICIENTE)

# Para el classification_report, solo necesitamos las etiquetas (y_true),
# que son pequeñas y sí caben en memoria.
print("Obteniendo etiquetas verdaderas (modo eficiente)...")
y_true = np.concatenate([y for x, y in validation_dataset], axis=0)


# 3. GENERAR PREDICCIONES (Tu código original está bien y es eficiente)
print("Generando predicciones en el dataset de validación...")
predicted_classes_prob = riesgo_model.predict(validation_dataset)
predicted_classes = np.argmax(predicted_classes_prob, axis=1)

# 4. REPORTE DE CLASIFICACIÓN
# Ahora podemos crear el reporte porque tenemos y_true y predicted_classes
print("\n" + "="*50)
print("REPORTE DE CLASIFICACIÓN FINAL")
print(classification_report(y_true, predicted_classes, target_names=sriesgos))
print("="*50 + "\n")


# 5. GRÁFICAS DE PREDICCIONES (ACIERTOS Y ERRORES)
# Para evitar el error OOM, solo graficaremos un lote (batch) de muestra.

print("Obteniendo un lote de muestra para las gráficas...")

# Tomamos un solo lote del dataset de validación
(sample_images, sample_labels) = next(iter(validation_dataset))

# Generamos predicciones SOLO para este lote
sample_predictions_prob = riesgo_model.predict(sample_images)
sample_predictions = np.argmax(sample_predictions_prob, axis=1)

# Convertimos las etiquetas de este lote a numpy
sample_labels = sample_labels.numpy()

# Aciertos (dentro de este lote)
correct = np.where(sample_predictions == sample_labels)[0]
print("Encontrados %d aciertos (en este lote de muestra)" % len(correct))
num_correct = min(9, len(correct))
fig_corr, axes_corr = plt.subplots(3, 3, figsize=(9, 9))
for i in range(9):
    r, c = divmod(i, 3)
    ax = axes_corr[r, c]
    if i < num_correct:
        idx = correct[i]
        # sample_images[idx] ya es un tensor (100,100,3) de 0-1 (float)
        # matplotlib.imshow puede manejar esto directamente.
        ax.imshow(sample_images[idx]) 
        ax.set_title("Pred: {}, Real: {}".format(sriesgos[sample_predictions[idx]], sriesgos[sample_labels[idx]]))
        ax.axis('off')
    else:
        ax.axis('off')
fig_corr.suptitle('Predicciones Correctas (Lote de Muestra)', y=0.98)
fig_corr.tight_layout(rect=(0, 0, 1, 0.97))

# Errores (dentro de este lote)
incorrect = np.where(sample_predictions != sample_labels)[0]
print("Encontrados %d errores (en este lote de muestra)" % len(incorrect))
num_incorrect = min(9, len(incorrect))
fig_inc, axes_inc = plt.subplots(3, 3, figsize=(9, 9))
for i in range(9):
    r, c = divmod(i, 3)
    ax = axes_inc[r, c]
    if i < num_incorrect:
        idx = incorrect[i]
        ax.imshow(sample_images[idx])
        ax.set_title("Pred: {}, Real: {}".format(sriesgos[sample_predictions[idx]], sriesgos[sample_labels[idx]]))
        ax.axis('off')
    else:
        ax.axis('off')
fig_inc.suptitle('Predicciones Incorrectas (Lote de Muestra)', y=0.98)
fig_inc.tight_layout(rect=(0, 0, 1, 0.97))

# Mostrar todas las figuras al final
plt.show()

# CREATE THE CLASIFICATION REPORT To know the accurray of the model
print(classification_report(y_true, predicted_classes, target_names=sriesgos))