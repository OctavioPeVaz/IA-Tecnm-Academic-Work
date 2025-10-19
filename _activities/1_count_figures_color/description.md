# Descripcion

La logica del programa que segui es la siguiente:

1. Recorro la mascara ya que las islas son las figuras y estan pintadas en color blanco (255) y lo demas es vacio (0).
2. Cuando se encuentra un pixel blanco, comienza la funcion (marcar_figura_visitada) lo que hace es que empieza a buscar todos los pixeles de esa isla 
de forma recursiva por asi decirlo se come la isla cambiando los pixeles a 0, no acaba hasta que termine de cambiar todos los blancos a negro de la isla.
3. Una vez termina sigue en la imagen buscando otra isla y contando cuantas van eliminadas en total.
4. Imprime el resultado en consola del numero de figuras encontradas, en este caso de color verde.