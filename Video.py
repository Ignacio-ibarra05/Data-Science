import cv2
import mediapipe as mp
import numpy as np
import time
from keras.models import load_model

# Cargar el modelo de detección de manos de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Cargar el modelo de clasificación de señas
model = load_model('simplified_classification.keras')

# Configurar la cámara
cap = cv2.VideoCapture(0)

# Crear el mapeo de clases a letras
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
assert len(classes) == 29, "El número de clases no coincide con el número esperado de 29 clases."

def get_hand_bounding_box(image, hand_landmarks):
    h, w, _ = image.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y

    return (x_min, y_min, x_max, y_max)

# Variables para la detección y cooldown
last_detection_time = time.time()
last_move_time = time.time()
detection_interval = 3  # segundos

# Inicializar variables
letter = ''
confidence = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    
    # Convertir la imagen a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar manos en la imagen
    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        last_detection_time = current_time
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtener el cuadro delimitador de la mano
            bbox = get_hand_bounding_box(image, hand_landmarks)
            
            # Verificar si el cuadro delimitador está dentro de los límites de la imagen
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > image.shape[1] or bbox[3] > image.shape[0]:
                continue

            # Recortar la imagen de la mano
            hand_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Verificar si el tamaño de la imagen recortada es mayor que cero
            if hand_image.size == 0:
                continue

            hand_image = cv2.resize(hand_image, (64, 64))  # Match the input size of the model
            hand_image = hand_image / 255.0
            hand_image = np.expand_dims(hand_image, axis=0)
            
            # Clasificar la seña cada 3 segundos
            if (current_time - last_move_time) >= detection_interval:
                last_move_time = current_time
                predictions = model.predict(hand_image)
                class_id = np.argmax(predictions)
                confidence = predictions[0][class_id]
                
                # Verificar que class_id esté dentro del rango de classes
                if class_id < len(classes):
                    letter = classes[class_id]
                else:
                    print(f'Error: class_id {class_id} está fuera del rango de la lista classes.')
    
    # Mostrar el resultado
    if letter and confidence:
        cv2.putText(frame, f'Letter: {letter}, Confidence: {confidence:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # Verificar si ha pasado más de 10 segundos sin detectar movimientos
    if (current_time - last_detection_time) > 10:
        print("No se detecta movimiento por más de 10 segundos. Deteniendo la detección.")
        break
    
    # Mostrar la imagen
    cv2.imshow('Hand Sign Detection', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

