import cv2
import numpy as np
import matplotlib.pyplot as plt

# Se carga la imagen
imagen = cv2.imread('D:\\Documentos\\Semestre_IX\\Inteligencia Artificial\\PROGRAMAS\\P1\\CA.jpg')

# Dar tamaño a la imagen mostrada
nuevo_ancho = 200  # ancho
nuevo_alto = 200   # alto
imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))

# Conversión de color adecuada para una imagen en color
color = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2RGB)

# Aplicar detección de bordes
bordes = cv2.Canny(color, 100, 200)

# Mostrar las imágenes
cv2.imshow('Imagen original redimensionada', imagen_redimensionada)
cv2.imshow('Imagen en RGB', color)
cv2.imshow('Bordes detectados', bordes)

cv2.waitKey(0)
cv2.destroyAllWindows()
