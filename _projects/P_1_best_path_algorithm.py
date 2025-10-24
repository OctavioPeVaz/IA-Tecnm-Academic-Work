import pygame

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

VERDE = (49, 180, 73)   # Inicio
ROJO = (180, 49, 49)    # Fin

GRIS_CLARO = (158, 158, 158)
GRIS = (35, 35, 35) 

GRIS_CLARITO = (196, 194, 194) # Cuadro Pendiente
AZUL_CLARO = (94, 119, 245) # Cuadro Visitado

AZUL = (28, 41, 106)    # Color de la ruta mas corta

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == GRIS

    def es_inicio(self):
        return self.color == VERDE

    def es_fin(self):
        return self.color == ROJO

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = VERDE

    def hacer_pared(self):
        self.color = GRIS           

    def hacer_fin(self):
        self.color = ROJO

    def hacer_no_visitado(self):
        self.color = GRIS_CLARITO

    def hacer_visitado(self):
        self.color = AZUL_CLARO

    def hacer_ruta(self):
        self.color = AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS_CLARO, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS_CLARO, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

# ---------------------------- Logica A* ----------------------------

def get_vecinos(nodo_actual, grid):
    vecinos = []
    total_filas_columnas = len(grid)
    fila_actual, columna_actual = nodo_actual.get_pos()
    direcciones = [
        # (y, x, costo)
        # Vertiales u Horizontales
        (-1,  0, 10), # Arriba
        ( 1,  0, 10), # Abajo
        ( 0, -1, 10), # Izquierda
        ( 0,  1, 10), # Derecha

        # Diagonales   
        (-1, -1, 14), # Arriba Izquierda
        (-1,  1, 14), # Arriba Derecha  
        ( 1, -1, 14), # Abajo Izquierda
        ( 1,  1, 14)  # Abajo Derecha
    ]

    for cambio_fila, cambio_columna, costo in direcciones:

        #Calcuar posicion del siguiente vecino
        nueva_fila = fila_actual + cambio_fila
        nueva_columna = columna_actual + cambio_columna

        # Verificar que la posicion no se salga del tablero como filas y columnas
        if (nueva_fila >= 0) and (nueva_fila < total_filas_columnas):
            fila_valida = True
        else:
            fila_valida = False

        if (nueva_columna >= 0) and (nueva_columna < total_filas_columnas):
            columna_valida = True
        else:
            columna_valida = False

        # Si no es valida continuar
        if not (fila_valida and columna_valida):
            continue

        # Si es valida obtenemos el vecino
        nodo_vecino = grid[nueva_fila][nueva_columna]
        # Verificar si es pared continuar con siguiente nodo
        if nodo_vecino.es_pared():
            continue


        # Verificacion para evitar que se traspasen entre 2 esquinas
        es_movimiento_diagonal = (cambio_fila != 0 and cambio_columna != 0)

        if es_movimiento_diagonal:
            adyacente_vertical      = grid[fila_actual + cambio_fila][columna_actual]
            adyacente_horizontal    = grid[fila_actual][columna_actual + cambio_columna]
            if adyacente_vertical.es_pared() and adyacente_horizontal.es_pared():
                continue

        vecinos.append((nodo_vecino, costo))
    return vecinos

def heuristica(a, b):
    costo_recto = 10
    costo_diagonal = 14

    (x1, y1) = a
    (x2, y2) = b

    # Distancia en cada eje
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    pasos_diagonales    = min(dx, dy)
    pasos_rectos        = max(dx, dy) - pasos_diagonales

    # Costo total aproximado
    return (pasos_diagonales * costo_diagonal) + (pasos_rectos * costo_recto)

def reconstruir_camino(nodo_fuente, posicion_final, grid, posicion_inicial):
    # Empezamos desde el final
    # Mientras la posicion actual tenga nodo fuente sigue el ciclo
    posicion_actual = posicion_final

    while posicion_actual in nodo_fuente:
        posicion_padre = nodo_fuente[posicion_actual]

        # Cambiamos ahora a la posicion del padre 
        posicion_actual = posicion_padre

        # Se encontro el inicio
        if posicion_actual == posicion_inicial:
            break

        # Obtener el nodo de la posicion padre para pintarlo
        nodo_camino = grid[posicion_actual[0]][posicion_actual[1]]

        nodo_camino.hacer_ruta()

def algoritmo_a_estrella(grid, inicio, fin):
    posicion_inicial    = inicio.get_pos()
    posicion_final      = fin.get_pos()

    lista_abierta = [posicion_inicial]

    # Ejemplo: {(1, 2): (1, 1)} significa que llegamos a (1, 2) desde (1, 1).
    nodo_fuente = {}

    g_score = {}
    for fila in grid:
        for nodo in fila:
            g_score[nodo.get_pos()] = float('inf')
    g_score[posicion_inicial] = 0

    f_score = {}
    for fila in grid:
        for nodo in fila:
            f_score[nodo.get_pos()] = float('inf')
    f_score[posicion_inicial] = heuristica(posicion_inicial, posicion_final)
    

    while len(lista_abierta) > 0:
        # Buscar el nodo con menor f_score 
        mejor_posicion = None
        menor_f = float('inf')

        for posicion_posible in lista_abierta:
            f_posible = f_score[posicion_posible]

            if f_posible < menor_f:
                menor_f = f_posible
                mejor_posicion = posicion_posible

        # Guardamos la m mejor posicion al actual
        posicion_actual = mejor_posicion

        # Comprobar si es final
        if posicion_actual == posicion_final:
            reconstruir_camino(nodo_fuente, posicion_final, grid, posicion_inicial)
            inicio.hacer_inicio()
            fin.hacer_fin()
            # Se encotro un camino
            return True
        
        # Quitamos de la lista la posicion alctual ya que nos encontramos en la posicion
        lista_abierta.remove(posicion_actual)

        nodo_actual = grid[posicion_actual[0]][posicion_actual[1]]

        # Lo marcamos como visitado
        if not nodo_actual.es_inicio() and not nodo_actual.es_fin():
            nodo_actual.hacer_visitado()

        for nodo_vecino, costo_movimiento in get_vecinos(nodo_actual, grid):
            posicion_vecino = nodo_vecino.get_pos()
            g_actual = g_score[posicion_actual]
            temp_g = g_actual + costo_movimiento

            # Decidir si este es un mejor camino
            g_vecino_anterior = g_score[posicion_vecino]

            if temp_g < g_vecino_anterior:
                # Registra de donde viene, su nodo padre
                nodo_fuente[posicion_vecino] = posicion_actual
                # Actualizamos su g_score
                g_score[posicion_vecino] = temp_g
                # Calcula su f_score actualizado
                h_vecino = heuristica(posicion_vecino, posicion_final)
                f_score[posicion_vecino] = temp_g + h_vecino

                # Agregamos el vecino a la lista abierta para ser evaluado en un futuro
                if posicion_vecino not in lista_abierta:
                    lista_abierta.append(posicion_vecino)
                    # Se pinta de color no visitado
                    if not nodo_vecino.es_inicio() and not nodo_vecino.es_fin():
                        nodo_vecino.hacer_no_visitado()

    print("No se encontro camino")
    return False

def main(ventana, ancho):
    FILAS = 10
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()

                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()

                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            if not nodo.es_pared() and not nodo.es_inicio() and not nodo.es_fin():
                                nodo.restablecer()
                                
                    algoritmo_a_estrella(grid, inicio, fin)

                if event.key == pygame.K_BACKSPACE:
                    inicio = None
                    fin = None
                    for fila in grid:
                        for nodo in fila:
                            nodo.restablecer()

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)