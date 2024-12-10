from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64

# Configuración
IMAGE_SIZE = (128, 128)

# Cargar el modelo
model = tf.keras.models.load_model("music_instrument_classifier.h5")

# Mapear índices a etiquetas de clases
class_labels = ['cuerda', 'viento', 'percusion']

# Crear la aplicación Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Recibir la imagen en base64 desde la solicitud
    data = request.json
    img_data = base64.b64decode(data['image'])
    
    # Convertir la imagen a formato OpenCV
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocesar la imagen
    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Realizar la predicción
    predictions = model.predict(img_batch)
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    confidence = float(np.max(predictions))
    
    # Responder con el resultado
    return jsonify({
        'label': predicted_label,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
