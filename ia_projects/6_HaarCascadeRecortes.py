import numpy as np
import cv2 as cv
import math 

rostro = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
cap = cv.VideoCapture(0)
#cap = cv.VideoCapture('assets/video_nate.mp4')
i = 0  
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in rostros:
       #frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)a
       frame2 = gray[ y:y+h, x:x+w]
       #frame3 = frame[x+30:x+w-30, y+30:y+h-30]
       
       frame2 = cv.resize(frame2, (48, 48), interpolation=cv.INTER_AREA)
       
        
       if(i%1==0):
           cv.imwrite('datasets/faces_dataset_48_size/josefa/'+str(i)+'.jpg', frame2)

           cv.imshow('rostror', frame2)
    cv.imshow('rostros', frame)
    i = i+1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()