import cv2 as cv
import numpy as np
import sys

sys.setrecursionlimit(100000)

img = cv.imread("assets/figura.png", 1)
img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

umbralBajo=(40, 80, 80  )
umbralAlto=(80, 255, 255)


mascara = cv.inRange(img2, umbralBajo, umbralAlto)
resultado = cv.bitwise_and(img2, img, mask=mascara)


cv.imshow('resultado', resultado)
cv.imshow('mascara', mascara)


# Contar Figuras
def contar_figuras_color(mascara):
    h, w = mascara.shape
    contador = 0

    def marcar_figura_visitada(y, x):
        # Revisar que no este fuera del rango el pixel visitado
        if (y < 0) or (y >= h) or (x < 0) or (x >= w):
            return
        # Revisar que sea 0 o sea es vacio
        if mascara[y, x] == 0:
            return
        
        # Si es pixel blanco (255) lo borramos
        mascara[y, x] = 0

        # Volvemos a llamar la funcion recursivamente para revisar los vecinos
        marcar_figura_visitada(y+1, x)
        marcar_figura_visitada(y-1, x)
        marcar_figura_visitada(y, x+1)
        marcar_figura_visitada(y, x-1)
    
    # Recorrer la imagen completa
    for y in range(h):
        for x in range(w):
            # Encontramos el pixel de una figura
            if mascara[y, x] == 255:
                contador += 1
                marcar_figura_visitada(y, x)



    return contador

print("Figuras encontradas: ", contar_figuras_color(mascara))


cv.waitKey(0)
cv.destroyAllWindows()