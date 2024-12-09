import cv2
import os

# Inicia la camarita
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# para guardae mi fotito
output_dir = "capturas"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el cuadro de la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 10, 150)
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            figura = 'Triangulo'
            cv2.putText(frame, figura, (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                figura = 'Cuadrado'
            else:
                figura = 'Rectangulo'
            cv2.putText(frame, figura, (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) == 5:
            figura = 'Pentagono'
            cv2.putText(frame, figura, (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) == 6:
            figura = 'Hexagono'
            cv2.putText(frame, figura, (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) > 10:
            figura = 'Circulo'
            cv2.putText(frame, figura, (x, y - 5), 1, 1, (0, 255, 0), 1)

        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

    cv2.imshow('Deteccion de Figuras', frame)

    # Capturo mi captura de la figurita que mostré 
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Presionar ESC para salir
        break
    elif key == ord('z'):  # Presionar Z para tomar captura
        captura_path = os.path.join(output_dir, 'Captura P7.png')
        cv2.imwrite(captura_path, frame)
        print(f"Captura guardada en: {captura_path}")

cap.release()
cv2.destroyAllWindows()