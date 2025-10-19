# Descripcion

Esta practica consto de crear un dataset el cual contuviera 5 clases, 4 compañeros y 1 de un video
en mi caso use 3 de mi familia y 2 personas de 2 videos de internet.

El tamaño de las muestras de cada clase lo deje en 48x48 y en escala de grises, cada clase con aproximadamente 1000 imagenes.

Se entrenaron en 3 modelos distintos (**EigenFace**, **FisherFace** y **LBPH**)

### EigenFace
Este fue el primer modelo en entrenar, tardo aproximadamente unos 20 minutos.
Los resultados fueron que tantito que cambiara la iluminacion el modelo ya no era capaz de reconocerte o si hacias ciertos gestos te detectaba como varias clases, vaya no era el mas preciso y la iluminacion debia ser casi igual al de las imagenes de entrenamiento.

### FisherFace
Este fue el segundo en ser entrenado, tardo aproximadamente 15 minutos.
Este fue el que mejor resultado tuvo, manteniendo mas constante la deteccion de rostros, ya detectaba con mas facilidad aunque la iluminacion en la habitacion fuera diferente.

### LBPH
Por ultimo se entreno con LBPH, fue el mas rapido de todos ya que basicametne tomo 14 segundos en entrenar el modelo.
Sin embargo fue el peor en reconocimiento ya que literalmente no me reconocia ningun rostro todos los rostros de las clases las marcaba como desconocidos, apesar de que fuera el mas rapido en mi caso no fue el que tuvo los mejores resultados.
