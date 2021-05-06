import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

print("Num GPUs Available: ", len(tf.config.list_physical_devices('CPU')))


#Para que nos muestre informacion sobre lo que está usando
#tf.debugging.set_log_device_placement(True)

'''
Vamos a crear una red neuronal para predecir el valor de salida que tendria la ecuacion y=cos(2x + 5)
Por supuesto la red neuronal no sabrá la ecuación
'''

#Creamos un conjunto de datos x e y respectivamente
xs = np.array([1.0, -1.0, 0.0, 8.0, 5.0, 11.0, -6.0, -3.0], dtype=float)
ys = np.array([math.cos(7), math.cos(3), math.cos(5), math.cos(21), math.cos(5), math.cos(27), math.cos(7), math.cos(1)], dtype=float)

'''
Definimos el modelo
En este caso vamos a usar una red neuronal de una entrada (x) y una salida (y)
Definimos el optimizador SGD
Definimos la función de error, en este caso error cuadratico medio
'''

model = keras.([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

#Entrenamos el modelo con n iteraciones
model.fit(xs,ys,epochs=1000)
model.save('./model.h5')

print("Listo")

num = float(input("introduce un numero: "))
result = model.predict([num])
print(result)