import cv2 as cv
import numpy as np

img = cv.imread("assets/f1.jpg", 0)

print(img.shape, img.size)
w = img.shape[0]
h = img.shape[1]
#c = img.shape[2]
#w1, h1 = img.shape
img2 = img
cv.imshow('imagen', img2)
for x in range(w):
    for y in range(h):
        if(img[x,y]>50):
            img[x,y]=255
        else:
            img[x,y]=0

cv.imshow('imagen1', img)
cv.waitKey(0)
cv.destroyAllWindows()