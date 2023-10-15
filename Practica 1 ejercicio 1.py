# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:30:48 2023

@author: luis mercado
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

# Lectura de los patrones de entrenamiento desde un archivo en formato texto separado por comas
datos_entrenamiento = pd.read_csv('XOR_trn.csv', header=None)

# Extracción de las entradas y las salidas de los datos de entrenamiento
entradas_entrenamiento = datos_entrenamiento.iloc[:, :-1].values
salidas_entrenamiento = datos_entrenamiento.iloc[:, -1].values

# Definición de parámetros
max_epocas = 100  # Número máximo de épocas (ajusta según tus necesidades)
tasa_aprendizaje = 0.1  # Tasa de aprendizaje (ajusta según tus necesidades)

# Creación y entrenamiento del perceptrón
perceptron = Perceptron(max_iter=max_epocas, eta0=tasa_aprendizaje)
perceptron.fit(entradas_entrenamiento, salidas_entrenamiento)

# Lectura de los patrones de prueba desde un archivo en formato texto separado por comas
datos_prueba = pd.read_csv('XOR_tst.csv', header=None)

# Extracción de las entradas y las salidas de los datos de prueba
entradas_prueba = datos_prueba.iloc[:, :-1].values
salidas_prueba = datos_prueba.iloc[:, -1].values

# Prueba del perceptrón entrenado en los datos de prueba
salidas_predecidas = perceptron.predict(entradas_prueba)

# Mostrar resultados
print('Salidas reales vs. Salidas predecidas:')
print(np.column_stack((salidas_prueba, salidas_predecidas)))

# Mostrar gráficamente los patrones y la recta que los separa
plt.scatter(entradas_entrenamiento[:, 0], entradas_entrenamiento[:, 1], c=salidas_entrenamiento, cmap='bwr')
plt.scatter(entradas_prueba[:, 0], entradas_prueba[:, 1], c=salidas_predecidas, cmap='bwr', marker='x')
x = np.linspace(-2, 2, 100)
y = (-perceptron.coef_[0][0] * x - perceptron.intercept_) / perceptron.coef_[0][1]
plt.plot(x, y, color='black', linewidth=2)
plt.legend(['Recta separadora', 'Patrones de entrenamiento', 'Patrones de prueba'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Patrones y Recta Separadora')
plt.show()
