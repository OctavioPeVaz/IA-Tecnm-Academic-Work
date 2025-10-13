import cv2 as cv 
import numpy as np 
import os
import time

start_time = time.time()

dataSet = './datasets/emotions_happy_sad_angry_dataset'

faces  = os.listdir(dataSet)
print(faces)

labels = []
facesData = []
label = 0 

for face in faces:
    facePath = dataSet+'/'+face

    for faceName in os.listdir(facePath):
        labels.append(label)
        facesData.append(cv.imread(facePath+'/'+faceName,0))

    label = label + 1
#print(np.count_nonzero(np.array(labels)==0)) 
    
faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))

print("Modelo Entrenado, Escribiendo XML")
faceRecognizer.write('./trained_models/FisherFaceEmotions.xml')
print("Guardado")

end_time = time.time()
total_time = end_time - start_time

hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

if hours > 0:
    time_text = f"{hours} hora{'s' if hours > 1 else ''}, {minutes} min y {seconds} seg"
elif minutes > 0:
    time_text = f"{minutes} min y {seconds} seg"
else:
    time_text = f"{seconds} seg"

print(f"Tiempo total de ejecución: {time_text}")

#['angry', 'happy', 'sad']

#Modelo Entrenado, Escribiendo XML
#Guardado
#Tiempo total de ejecución: 11 min y 46 seg