import tensorflow as tf
import numpy as np

# (temperatura, humedad, intensidad rayos UV, hora del dia)
#1 lluvioso
#2 soleado
#3  nublado
entradas = np.array ([[18,85,1,9,],[30,20,8,13],[25,60,5,14],[22,50,4,16],[28,30,6,12],[15,90,0,7],[20,40,5,10],[32,25,9,15],[19,70,2,17],[24,55,3,18],[27,35,7,14],[16,80,1,20],[23,45,4,11],[31,20,8,16],[21,65,2,8]])
salidas = np.array([1,2,1,3,2,1,3,2,3,3,2,1,3,2,1])

capaoculta1=tf.keras.layers.Dense(units=8,input_shape=[4])
capaoculta2=tf.keras.layers.Dense(units=4)
capaoculta3=tf.keras.layers.Dense(units=2)
capa_salida=tf.keras.layers.Dense(units=1,activation='tanh')
modelo=tf.keras.Sequential([capaoculta1,capaoculta2,capaoculta3,capa_salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mean_squared_error'
)

historial=modelo.fit(entradas,salidas,epochs=1000, verbose=False)

import matplotlib.pyplot as plt
plt.xlabel('entrenamientos')
plt.ylabel('pérdidas')
plt.plot(historial.history["loss"])

validacion = np.array([[26,80,6,13]])
resultado=modelo.predict(validacion)
print(resultado)

