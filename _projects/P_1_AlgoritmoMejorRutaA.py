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

NARANJA = (255, 165, 0) # Cuadro Pendiente
PURPURA = (128, 0, 128) # Cuadro Visitado

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

    def hacer_abierta(self):
        self.color = NARANJA

    def hacer_cerrada(self):
        self.color = PURPURA

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

def get_vecinos(nodo, grid):
    vecinos = []
    filas = len(grid)
    fila, col = nodo.get_pos()
    direcciones = [(-1, 0, 10), (1, 0, 10), (0, -1, 10), (0, 1, 10), (-1, -1, 14), (-1, 1, 14), (1, -1, 14), (1, 1, 14)]

    for df, dc, coste in direcciones:
        nf, nc = fila + df, col + dc

        if 0 <= nf < filas and 0 <= nc < filas:
            vecino = grid[nf][nc]

            if vecino.es_pared():
                continue

            # evitar cortar esquinas: si es diagonal, comprobar ortogonales
            if df != 0 and dc != 0:
                ady1 = grid[fila + df][col]   # fila+df, col (vertical vecino)
                ady2 = grid[fila][col + dc]   # fila, col+dc (horizontal vecino)

                if ady1.es_pared() or ady2.es_pared():
                    # si alguna de las casillas ortogonales es pared, no permitir la diagonal
                    continue

            vecinos.append((vecino, coste))
    return vecinos

def heuristica(a, b):
    (x1, y1) = a
    (x2, y2) = b
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    m = min(dx, dy)

    # coste óptimo = 14*m + 10*(max-min)
    return 14 * m + 10 * (max(dx, dy) - m)
    # equivalente: 10*(dx+dy) - 6*m

def reconstruir_camino(came_from, actual_pos, grid, inicio_pos):
    while actual_pos in came_from:
        actual_pos = came_from[actual_pos]
        if actual_pos == inicio_pos:
            break
        nodo = grid[actual_pos[0]][actual_pos[1]]
        nodo.hacer_ruta()

def algoritmo_a_estrella(grid, inicio, fin):
    inicio_pos = inicio.get_pos()
    fin_pos = fin.get_pos()

    abiertos = [inicio_pos]
    came_from = {}
    g_score = {inicio_pos: 0}
    f_score = {inicio_pos: heuristica(inicio_pos, fin_pos)}

    while len(abiertos) > 0:
        # Buscar el nodo con menor f_score 
        actual_pos = min(abiertos, key=lambda pos: f_score.get(pos, float('inf')))

        if actual_pos == fin_pos:
            reconstruir_camino(came_from, fin_pos, grid, inicio_pos)
            return True

        abiertos.remove(actual_pos)
        nodo_actual = grid[actual_pos[0]][actual_pos[1]]

        if not nodo_actual.es_inicio() and not nodo_actual.es_fin():
            nodo_actual.hacer_cerrada()

        for vecino, coste in get_vecinos(nodo_actual, grid):
            vecino_pos = vecino.get_pos()
            temp_g = g_score.get(actual_pos, float('inf')) + coste

            if temp_g < g_score.get(vecino_pos, float('inf')):
                came_from[vecino_pos] = actual_pos
                g_score[vecino_pos] = temp_g
                f_score[vecino_pos] = temp_g + heuristica(vecino_pos, fin_pos)

                if vecino_pos not in abiertos:
                    abiertos.append(vecino_pos)
                    if not vecino.es_inicio() and not vecino.es_fin():
                        vecino.hacer_abierta()
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

                if event.key == pygame.K_DELETE:
                    inicio = None
                    fin = None
                    for fila in grid:
                        for nodo in fila:
                            nodo.restablecer()

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)