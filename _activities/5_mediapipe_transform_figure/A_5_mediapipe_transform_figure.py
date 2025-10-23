import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

DEFAULT_SIZE = 100
last_size = DEFAULT_SIZE


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame= cv2.flip(frame, 1)
    # Convertir imagen a RGB (MediaPipe usa RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

     # Obtener dimensiones del frame
    h, w, _ = frame.shape

    # Calcular centro
    cx, cy = w // 2, h // 2


    # Variables para guardar coordenadas
    left_index = None
    right_index = None
    
   
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' o 'Right'
            #print(label)
            h, w, _ = frame.shape
            
            # Coordenadas del índice (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            
            # Guardar según la mano
            if label == 'Left':
                left_index = (x, y)
            elif label == 'Right':
                right_index = (x, y)

            # Dibujar los landmarks (opcional)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Si ambas manos detectadas, dibujar línea entre los dos puntos y actualizar tamaño
        if left_index and right_index:
            cv2.line(frame, left_index, right_index, (0, 255, 0), 3)

            # Calcular distancia entre los dedos
            dist = math.hypot(right_index[0] - left_index[0], right_index[1] - left_index[1])

            # Calcular nuevo tamaño
            new_size = int(100 + dist)

            last_size = new_size


    # Calcular las esquinas del cuadrado (centrado)
    half = last_size // 2
    top_left = (cx - half, cy - half)
    bottom_right = (cx + half, cy + half)

    # Dibujar el cuadrado
    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 3)



    cv2.imshow("Line", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()