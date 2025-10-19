import pygame
import heapq

# Inicializar pygame
pygame.init()

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización A* - Nodos")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)        # usado para camino final
ROJO = (255, 0, 0)         # cerrado
NARANJA = (255, 165, 0)    # inicio
PURPURA = (128, 0, 128)    # fin
AZUL = (0, 0, 255)         # abierto / frontera
AMARILLO = (255, 255, 0)   # camino reconstruido

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        # Nota: la versión original intercambiaba x/y; mantener compatibilidad con el resto
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        self.color = AZUL

    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_camino(self):
        self.color = AMARILLO

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        filas = self.total_filas
        # movimientos ortogonales
        if self.fila < filas - 1 and not grid[self.fila + 1][self.col].es_pared():  # abajo
            self.vecinos.append((grid[self.fila + 1][self.col], 10))
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():  # arriba
            self.vecinos.append((grid[self.fila - 1][self.col], 10))
        if self.col < filas - 1 and not grid[self.fila][self.col + 1].es_pared():  # derecha
            self.vecinos.append((grid[self.fila][self.col + 1], 10))
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():  # izquierda
            self.vecinos.append((grid[self.fila][self.col - 1], 10))

        # movimientos diagonales
        if self.fila > 0 and self.col > 0 and not grid[self.fila - 1][self.col - 1].es_pared():  # ↖️
            self.vecinos.append((grid[self.fila - 1][self.col - 1], 14))
        if self.fila > 0 and self.col < filas - 1 and not grid[self.fila - 1][self.col + 1].es_pared():  # ↗️
            self.vecinos.append((grid[self.fila - 1][self.col + 1], 14))
        if self.fila < filas - 1 and self.col > 0 and not grid[self.fila + 1][self.col - 1].es_pared():  # ↙️
            self.vecinos.append((grid[self.fila + 1][self.col - 1], 14))
        if self.fila < filas - 1 and self.col < filas - 1 and not grid[self.fila + 1][self.col + 1].es_pared():  # ↘️
            self.vecinos.append((grid[self.fila + 1][self.col + 1], 14))

    # Para usar en heapq (no importa el orden real, solo que exista)
    def __lt__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, Nodo) and self.fila == other.fila and self.col == other.col

    def __hash__(self):
        return hash((self.fila, self.col))


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
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    # la versión original usaba y,x = pos (pos viene como (x,y)), mantener para compatibilidad
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def heuristica(a, b):
    # Manhattan
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        if not current.es_inicio():
            current.hacer_camino()
        draw()

def algoritmo_a_estrella(draw, grid, inicio, fin):
    for fila in grid:
        for nodo in fila:
            nodo.actualizar_vecinos(grid)

    count = 0
    open_set = []
    heapq.heappush(open_set, (0, count, inicio))
    came_from = {}

    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = heuristica(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio}

    while open_set:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = heapq.heappop(open_set)[2]
        open_set_hash.remove(current)

        if current == fin:
            reconstruir_camino(came_from, fin, draw)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return True

        # marcar como cerrado
        if not current.es_inicio():
            current.hacer_cerrado()

        for vecino, costo in current.vecinos:
            temp_g_score = g_score[current] + costo  # coste entre vecinos = 1
            if temp_g_score < g_score[vecino]:
                came_from[vecino] = current
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + heuristica(vecino.get_pos(), fin.get_pos())
                if vecino not in open_set_hash:
                    count += 1
                    heapq.heappush(open_set, (f_score[vecino], count, vecino))
                    open_set_hash.add(vecino)
                    if not vecino.es_fin():
                        vecino.hacer_abierto()

        draw()

        # pequeña pausa para apreciar la visualización (opcional)
        pygame.time.delay(20)

    # si llegamos aquí, no hay camino
    return False

def limpiar_grid(grid):
    for fila in grid:
        for nodo in fila:
            nodo.restablecer()

def main(ventana, ancho):
    FILAS = 10  # puedes ajustar la resolución de la cuadrícula
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
                if fila < 0 or fila >= FILAS or col < 0 or col >= FILAS:
                    continue
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
                if fila < 0 or fila >= FILAS or col < 0 or col >= FILAS:
                    continue
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    # ejecutar A*
                    algoritmo_a_estrella(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)

                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)

        # dibujar(ventana, grid, FILAS, ancho)

    pygame.quit()

if __name__ == "__main__":
    main(VENTANA, ANCHO_VENTANA)
