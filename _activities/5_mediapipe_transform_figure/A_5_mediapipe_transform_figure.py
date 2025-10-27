import cv2
import math
import numpy as np
import mediapipe as mp

mp_hands_solutions = mp.solutions.hands
mp_drawing_utils = mp.solutions.drawing_utils

hands_detector = mp_hands_solutions.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.5
)

# Iniciar caputra de video de camara 0
video_capture = cv2.VideoCapture(0)

DEFAULT_SQUARE_SIZE = 50
current_square_size = DEFAULT_SQUARE_SIZE
current_angle_rad  = 0.0


while True:
    is_frame_read, frame = video_capture.read()
    
    if not is_frame_read:
        break

    frame= cv2.flip(frame, 1)

    # Convertir imagen a RGB (MediaPipe usa RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    hand_detection_results = hands_detector.process(frame_rgb)

    # Obtener dimensiones frame
    height, width, _ = frame.shape

    # Calcular centro
    center_x = width // 2
    center_y = height // 2

    # Variables para guardar coordenadas
    left_index_coords = None
    right_index_coords = None
    
   
    if hand_detection_results.multi_hand_landmarks and hand_detection_results.multi_handedness:

        for hand_landmarks, handedness in zip(hand_detection_results.multi_hand_landmarks, hand_detection_results.multi_handedness):
            # Obtiene etiqueta 'Left' o 'Right'
            label = handedness.classification[0].label 

            # Obtiene coordenadas de la punta del índice (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            
            # Covertir coordenadas normalizada (0.0 a 1.0) a coordenadas de pixeles
            
            pixel_x = int(index_finger_tip.x * width)
            pixel_y = int(index_finger_tip.y * height)
             
            # Guardar coordenadas de acuerdo la mano
            if label == 'Left':
                left_index_coords = (pixel_x, pixel_y)
            elif label == 'Right':
                right_index_coords = (pixel_x, pixel_y)

            # Dibujar los landmarks de las manos
            mp_drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands_solutions.HAND_CONNECTIONS)

        
        if left_index_coords and right_index_coords:
            # Dibujar línea/distancia entre los 2 indices
            cv2.line(frame, left_index_coords, right_index_coords, (0, 255, 0), 3)

            # Calculo distancia entre los indices
            finger_distance = math.hypot(
                right_index_coords[0] - left_index_coords[0], 
                right_index_coords[1] - left_index_coords[1]
            )

            # Calcular nuevo tamaño del cuadrado
            new_square_size = int(DEFAULT_SQUARE_SIZE + finger_distance)
            current_square_size = new_square_size

            # Calcular nuevo angulo del cuadrado
            delta_x = right_index_coords[0] - left_index_coords[0]
            delta_y = right_index_coords[1] - left_index_coords[1]

            current_angle_rad = math.atan2(delta_y, delta_x)

    
    half = current_square_size // 2

    cos_angle = math.cos(current_angle_rad)
    sin_angle = math.sin(current_angle_rad)

    current_corners_rotation = [
        (-half, -half),
        ( half, -half),
        ( half,  half),
        (-half,  half)
    ]

    rotated_corners = []

    # Calcular coordenadas de las 2 esquinas del cuadrado para escalar siempre al centro
    top_left        = (center_x - half, center_y - half)
    bottom_right    = (center_x + half, center_y + half)

    # Dibujar el cuadrado
    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 3)

    cv2.imshow("Escalado y Rotacion Cuadrado", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()