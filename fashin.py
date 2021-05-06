import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

#Accedemos al set de imagenes de moda
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'''
Cada imagen tiene una etiqueta que va del 0 al 9
Label |   Class
-------------------
  0	  | T-shirt/top
  1	  | Trouser
  2   | Pullover
  3   | Dress
  4   | Coat
  5   | Sandal
  6   | Shirt
  7   | Sneaker
  8   | Bag
  9   | Ankle boot
'''

#Nombres de las clases que vamos a buscar
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Consultamos el set de datos de entrenamiento que hemos obtenido
#Nos mostrará que hay 60.000 imagenes en el set de entrenamiento de 28 x 28 pixeles
print(train_images.shape)

#Consultamos el set de etiquetas de entrenamiento que hemos obtenido
#Nos mostrará que hay 60.000 etiquetas
print(len(train_labels))

#Consultamos el set de datos de test que hemos obtenido
#Nos mostrará que hay 60.000 imagenes en el set de entrenamiento de 28 x 28 pixeles
print(test_images.shape)

#Consultamos el set de etiquetas de test que hemos obtenido
#Nos mostrará que hay 60.000 etiquetas
print(len(test_labels))

'''
Pre-procesamos el set de datos
'''

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

#Comprobamos que todo esté en orden y desplegamos las 25 primeras imagenes

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

#Creamos las capas de la red neuronal
model  = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #Esta capa se encarga de transformar el array bidimensonal de pixeles a un array unidimensional de 28 * 28 = 784 pixeles
    keras.layers.Dense(128, activation='relu'),#Capa de 128 nodos 
    keras.layers.Dense(10, activation='softmax') #Capa de 10 nodos, es la capa de salida por lo que deberá tener el mismo numero de nodos que de clases tengamos
])

#Compilamos el modelo
model.compile(
    optimizer='adam', #Metodo de optimización
    loss='sparse_categorical_crossentropy', #Configuramos la función de perdida
    metrics=['accuracy'] #Se usan para monitorear los pasos de entrenamiento y de pruebas
)

#Entrenamos el modelo
model.fit(train_images, train_labels, epochs = 10)

#Evaluamos la exactitud del modelo
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#Hacemos predicciones sobre una imagen
predictions = model.predict(test_images)
print(predictions[0]) #Miramos que ha predecido sobre la  imagen 1
print(np.argmax(predictions[0])) #Revisamos cual tiene el valor mas alto

#Comparamos lo que ha predecido con la respuesta correcta
print("Valor predecido: ", np.argmax(predictions[0]), " valor real: ", test_labels[0])

#Graficamos las primeas x imagenes para poder ver mejor los datos

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#Seleccionamos las primeras 12 imagenes
num_rows = 4
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

'''
Predecimos la etiqueta para una imagen
Para ello tenemos que agregar la imagen a una lista para que el modelo de keras pueda entenderlo
'''
# Grab an image from the test dataset.
img = test_images[1]

print(img.shape) 
#Añadimos la imagen a un bloque
img = (np.expand_dims(img,0))
print(img.shape)

#Predecimos la etiqueta
predictions_single = model.predict(img)

print(predictions_single)

#Graficamos la prediccion
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
