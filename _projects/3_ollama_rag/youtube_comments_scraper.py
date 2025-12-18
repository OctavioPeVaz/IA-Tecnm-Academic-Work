import csv
import re
import sys
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT

# ================= CONFIGURACIÓN =================
URLS = [
    "https://www.youtube.com/watch?v=P6f7dhazgAM",
]

ARCHIVO_SALIDA = "./datasets/corpus/comentarios_LageneracionZestacondenada.csv"

# Cantidad máxima de comentarios por video
MAX_COMENTARIOS = 2000 

# SORT_BY_POPULAR o SORT_BY_RECENT
ORDEN = SORT_BY_POPULAR 

MIN_CARACTERES = 80
# ==============================================================

def limpiar_texto(texto):
    """
    Elimina emojis y caracteres especiales no deseados, pero mantiene
    signos de puntuación básicos y letras con acentos (Español).
    """
    if not texto:
        return ""
    
    # Expresión regular que elimina caracteres del rango "Astral Plane (emojis)"
    # Mantiene caracteres desde U+0000 hasta U+FFFF (incluye letras, ñ, áéíóú, símbolos básicos)
    texto_sin_emojis = re.sub(r'[^\u0000-\uFFFF]', '', texto)
    
    # Opcional: Limpiar espacios excesivos o saltos de línea
    texto_limpio = " ".join(texto_sin_emojis.split())
    
    return texto_limpio

def main():
    downloader = YoutubeCommentDownloader()
    
    # Columnas para el CSV
    columnas = ['video_id', 'autor', 'texto', 'likes']
    
    print(f"--- INICIANDO SCRAPING ---")
    print(f"Filtro de calidad: Mínimo {MIN_CARACTERES} caracteres.")
    print(f"Guardando en: {ARCHIVO_SALIDA}\n")

    try:
        with open(ARCHIVO_SALIDA, mode='w', encoding='utf-8', newline='') as archivo_csv:
            writer = csv.DictWriter(archivo_csv, fieldnames=columnas)
            writer.writeheader()

            for url in URLS:
                print(f"Procesando video: {url}")
                try:
                    if "v=" in url:
                        video_id = url.split("v=")[1].split("&")[0]
                    else:
                        video_id = "video_id"

                    # Descargar comentarios
                    generator = downloader.get_comments_from_url(url, sort_by=ORDEN)
                    
                    guardados = 0
                    revisados = 0
                    
                    # Usamos un bucle manual para poder filtrar sin romper el límite
                    for comment in generator:
                        revisados += 1
                        if revisados > MAX_COMENTARIOS:
                            break
                            
                        texto_original = comment.get('text', '')
                        texto_limpio = limpiar_texto(texto_original)
                        likes = comment.get('like_count', 0)
                        
                        # --- FILTRO MÁGICO ---
                        # 1. Que tenga texto
                        # 2. Que supere el largo mínimo
                        if texto_limpio and len(texto_limpio) >= MIN_CARACTERES:
                            writer.writerow({
                                'video_id': video_id,
                                'autor': comment.get('author', 'Anónimo'),
                                'texto': texto_limpio,
                                'likes': likes
                            })
                            guardados += 1
                            
                    print(f"   -> Revisados: {revisados} | Guardados (útiles): {guardados}")

                except Exception as e:
                    print(f"   [ERROR] En video {url}: {e}")

    except Exception as e:
        print(f"[ERROR CRÍTICO] No se pudo crear el archivo: {e}")

    print("\n¡Listo! Revisa tu archivo CSV.")

if __name__ == "__main__":
    main()