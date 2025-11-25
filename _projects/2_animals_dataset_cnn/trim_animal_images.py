import supervision as sv
import cv2
import os
import glob 
from inference import get_model


INPUT_IMAGE_FOLDER = "D:/Images_datasets/Turtles/resized" 
CROP_FOLDER = "D:/Images_datasets/Turtles/cropped_images_model_2" 

# --- Recognition Model from Roboflow ---
API_KEY = "ECCPsTb8kmgSvgfunnqD" 
MODEL_ID = "turtle-uho6h/1"

INFERENCE_CONFIG = {
    "confidence": 0.70, 
    "iou": 0.5
}
TARGET_CROP_SIZE = (100, 100)

# --- INIT ---

# Create folder for clippings
os.makedirs(CROP_FOLDER, exist_ok=True)
print(f"Carpeta de recortes: {CROP_FOLDER}")

# Init Model
print("Cargando modelo...")
model = get_model(
    model_id=MODEL_ID,
    api_key=API_KEY
)


crop_counter = 0


print(f"Buscando imágenes en: {INPUT_IMAGE_FOLDER}")
image_extensions = ["*.jpg", "*.jpeg", "*.png"]
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_IMAGE_FOLDER, ext)))
print(f"Se encontraron {len(image_files)} imágenes para procesar.")



for image_path in image_files:
    
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Advertencia: No se pudo leer la imagen {image_path}. Omitiendo.")
        continue
        
    print(f"Procesando: {image_path}")
    
    results = model.infer(
        frame,
        inference_configuration=INFERENCE_CONFIG
    )[0]

    detections = sv.Detections.from_inference(results)
    detections = detections[detections.class_id == 0]

   
    
    # Iterar sobre cada detección en la imagen actual
    for bbox in detections.xyxy:
        
        # Obtener coordenadas enteras
        x_min, y_min, x_max, y_max = [int(i) for i in bbox]

        # Recortar la imagen original
        cropped_image = frame[y_min:y_max, x_min:x_max]
        
        # Verificar si el recorte es válido
        if cropped_image.size > 0:
            resized_image = cv2.resize(cropped_image, TARGET_CROP_SIZE, interpolation=cv2.INTER_AREA)
            crop_counter += 1
            filename = os.path.join(CROP_FOLDER, f"{crop_counter}.jpg")
            cv2.imwrite(filename, resized_image)

print(f"✅ Procesamiento terminado. Total de recortes guardados (contando inicio): {crop_counter}.")
print(f"✅ Todos los recortes están en {CROP_FOLDER}.")