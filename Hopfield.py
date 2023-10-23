# Librería para crear vectores y matrices
import numpy as np
# Librería para visualizar gráficos
import matplotlib.pyplot as plt

# Imagen del aro
aro = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Aplanar la matriz 2D a un vector 1D
aro_vector = aro.flatten()

# Crear la matriz de pesos
num_neuronas = len(aro_vector)
pesos = np.zeros((num_neuronas, num_neuronas))

for i in range(num_neuronas):
    for j in range(num_neuronas):
        if i != j:
            pesos[i, j] = aro_vector[i] * aro_vector[j]

# Función de activación 
def hopfield(patron_entrada_vector, pesos, max_iter=500):
    for _ in range(max_iter):
        for i in range(num_neuronas):
            patron_entrada_vector[i] = np.sign(np.dot(pesos[i, :], patron_entrada_vector))
    return patron_entrada_vector

# Ingresar una imagen ruidosa
patron_entrada = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

patron_entrada_vector = patron_entrada.flatten()

# Aplicar la red de Hopfield
patron_salida = hopfield(patron_entrada_vector, pesos)

# Reconstruir la imagen de salida a 10x10
patron_salida = patron_salida.reshape(aro.shape)

# Mostrar la imagen original, la imagen de entrada y la imagen reconstruida
plt.subplot(1, 3, 1)
plt.imshow(aro, cmap='gray')
plt.title('Imagen Original')

plt.subplot(1, 3, 2)
plt.imshow(patron_entrada_vector.reshape(aro.shape), cmap='gray')
plt.title('Imagen con Ruido')

plt.subplot(1, 3, 3)
plt.imshow(patron_salida, cmap='gray')
plt.title('Imagen Reconstruida')

plt.show()