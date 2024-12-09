import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('D:\Documentos\Semestre_IX\Inteligencia Artificial\PROGRAMAS\P4\ciculos.jpg')
# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detectar círculos usando HoughCircles
circulos = cv2.HoughCircles(
    gris, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
    param1=50, param2=30, minRadius=10, maxRadius=50
)

if circulos is not None:
    # Redondear y convertir a enteros
    circulos = np.round(circulos[0, :]).astype("int")
    escala_referencia = 20.0
    referencia_circulo = circulos[0]  # Usar el primer círculo como referencia

    for (x, y, r) in circulos:
        # Dibujar el círculo detectado
        cv2.circle(imagen, (x, y), r, (0, 255, 0), 2)
        # Dibujar el centro del círculo
        cv2.circle(imagen, (x, y), 2, (0, 0, 255), 3)
        # Calcular distancia en píxeles
        pixel_distancia = np.sqrt((x - referencia_circulo[0]) ** 2 + (y - referencia_circulo[1]) ** 2)
        # Escalar la distancia a unidades reales
        escala = escala_referencia / referencia_circulo[2]
        real_distancia = pixel_distancia * escala

        # Dibujar una línea entre el círculo actual y el de referencia
        cv2.line(imagen, (referencia_circulo[0], referencia_circulo[1]), (x, y), (255, 0, 0), 1)
        # Escribir la distancia en mm cerca del círculo
        cv2.putText(imagen, f"{real_distancia:.2f} mm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Dibujar el círculo de referencia en azul
    cv2.circle(imagen, (referencia_circulo[0], referencia_circulo[1]), referencia_circulo[2], (255, 0, 0), 2)
    # Escribir la etiqueta "Referencia" cerca del círculo de referencia
    cv2.putText(imagen, "Referencia", (referencia_circulo[0] - 50, referencia_circulo[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Mostrar la imagen con los círculos detectados
cv2.imshow('Circulos Detectados', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()