import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

#Para que nos muestre informacion sobre lo que está usando
#tf.debugging.set_log_device_placement(True)

'''
Vamos a crear una red neuronal para predecir el valor de salida que tendria la ecuacion y=3x-4
Por supuesto la red neuronal no sabrá la ecuación
'''

#Creamos un conjunto de datos x e y respectivamente
xs = np.array([-2.0, 0.0, 4.0, 1.0, 6.0, 5.0, 3.0, -3.0], dtype=float)
ys = np.array([-10.0, -4.0, 8.0, -1.0, 14.0, 11.0, 5.0, -13.0], dtype=float)

'''
Definimos el modelo
En este caso vamos a usar una red neuronal de una entrada (x) y una salida (y)
Definimos el optimizador SGD
Definimos la función de error, en este caso error cuadratico medio
'''

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

#Entrenamos el modelo con n iteraciones
model.fit(xs,ys,epochs=1000)
model.save('./model.h5')

print("Listo")

num = float(input("introduce un numero: "))
result = model.predict([num])
print(result)