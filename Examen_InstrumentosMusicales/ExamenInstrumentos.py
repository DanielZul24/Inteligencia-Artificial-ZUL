import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# Configuración
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = r"D:\Documentos\Semestre_IX\Inteligencia Artificial\Examen\dataset"  # Cambiar a la ruta de tu conjunto de datos

# Preparar los generadores de datos
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalización
    validation_split=0.2  # Dividir datos en entrenamiento y validación
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 clases: cuerda, viento, percusión
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Guardar el modelo entrenado
model.save("music_instrument_classifier.h5")
print("Modelo guardado como music_instrument_classifier.h5")

# Función para realizar predicciones
def classify_image_from_webcam(model):
    # Iniciar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)
    
    while True:
        # Leer un frame desde la cámara
        ret, frame = cap.read()
        
        if not ret:
            print("Error al capturar la imagen")
            break
        
        # Redimensionar la imagen capturada
        image = cv2.resize(frame, IMAGE_SIZE)
        image = image / 255.0  # Normalización
        image = np.expand_dims(image, axis=0)  # Agregar dimensión batch
        
        # Realizar la predicción
        prediction = model.predict(image)
        class_labels = list(train_generator.class_indices.keys())
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Mostrar la predicción en la pantalla
        cv2.putText(frame, f"Prediccion de tipo: {predicted_class} ({confidence*100:.2f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar la imagen en una ventana
        cv2.imshow('Prediccion en vivo', frame)

        # Si se presiona la tecla 'q', salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Cargar el modelo guardado
model = tf.keras.models.load_model("music_instrument_classifier.h5")

# Iniciar la predicción desde la webcam
classify_image_from_webcam(model)

# Convertir el modelo a formato TensorFlow.js
tfjs_target_dir = "tfjs_model"
tf.converters.save_keras_model(model, tfjs_target_dir)

print(f"Modelo exportado a {tfjs_target_dir}")
