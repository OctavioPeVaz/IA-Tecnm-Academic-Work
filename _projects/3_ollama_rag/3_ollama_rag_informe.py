import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from create_vectors import retriever
import pandas as pd

# --- CONFIGURACIÓN DEL MODELO ---
model = OllamaLLM(model="phi4") 

template = """
Eres un investigador experto en filosofía, sociología digital y análisis de datos. Estás trabajando en un proyecto titulado "La Generación Z y la Crisis de Sentido en la Era Digital".

Tu objetivo es responder preguntas sintetizando dos tipos de información que se te proporcionarán:
1. MARCO TEÓRICO: Conceptos filosóficos (Existencialismo, Posmodernidad, Foucault, Byung-Chul Han, etc.).
2. EVIDENCIA EMPÍRICA: Datos de redes sociales, comentarios de YouTube y encuestas sintéticas.

Instrucciones:
- Utiliza la información proporcionada en el CONTEXTO para responder.
- Siempre intenta conectar la teoría (ej. "Vacío existencial") con la evidencia real (ej. "Comentarios de usuarios").
- Si la información no está en el contexto, di que no tienes datos suficientes.
- Responde siempre en Español formal y académico pero accesible.
- Usa formato Markdown (negritas, listas) para estructurar tu respuesta.

CONTEXTO RECUPERADO:
{context}

PREGUNTA DEL USUARIO: 
{question}

RESPUESTA (Análisis estructurado):
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def write_markdown(docs):
    """Convierte la lista de documentos en un string con formato de lista Markdown."""
    seen_sources = set()
    md_output = "\n### Fuentes y Evidencia Utilizada\n"
    
    found_any = False
    
    for doc in docs:
        meta = doc.metadata
        
        source_raw = meta.get("source", "desconocido")
        source_clean = str(source_raw).strip().upper()
        
        details = ""
        icon = "📄"
        
        if source_clean == "YOUTUBE_COMENTARIOS":
            author = meta.get('author', 'Anon')
            likes = meta.get('like_count', 0)
            details = f"**Comentario de YouTube** | Autor: *{author}*"
            
        elif source_clean == "TEORIA_FILOSOFICA":
            autor_con = meta.get('autor_concepto', 'Autor desconocido')
            eje = meta.get('eje_analisis', 'General')
            details = f"**Teoría Académica** | Concepto de: *{autor_con}* (Eje: {eje})"
            
        elif source_clean == "REDES_SOCIALES_SINTETICO":
            tema = meta.get('tema', 'General')
            sentimiento = meta.get('sentimiento', 'Neutro')
            details = f"**Data Redes Sociales** | Tema: {tema} (Sentimiento: {sentimiento})"
        else:
            details = f"Fuente: {source_clean}"

        # Construir línea
        identifier = f"{source_clean}_{details}"
        
        # Usamos el texto formateado para verificar duplicados visuales
        formatted_line = f"- {icon} {details}"
        
        if formatted_line not in seen_sources:
            md_output += f"{formatted_line}\n"
            seen_sources.add(formatted_line)
            found_any = True
            
    if not found_any:
        return "\n> *No se encontraron fuentes específicas en la base de datos para esta consulta.*\n"
        
    return md_output + "\n---\n"


def batch_process():
    input_file = "./_projects/3_ollama_rag/questions.txt"
    output_file = "./_projects/3_ollama_rag/informe_final.md"
    
    # Verificar si existen preguntas
    if not os.path.exists(input_file):
        print(f"Error: No se encontró el archivo '{input_file}'.")
        print("Por favor crea un archivo llamado 'questions.txt' y pon una pregunta por línea.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    if not questions:
        print("El archivo questions.txt está vacío.")
        return

    print(f"\n Archivo cargado: {len(questions)} preguntas encontradas.")
    print(f" Generando informe en: {output_file}...\n")

    # Inicializar archivo de reporte
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("#~~~~~~~~~~~Informe de Investigación: Gen Z y Crisis de Sentido~~~~~~~~~~~#\n")
        f.write(f"**Fecha de generación:** {pd.Timestamp.now()}\n")
        f.write("**Modelo:** Ollama Phi-4 (RAG Local)\n\n")
        f.write("Este informe ha sido generado automáticamente analizando fuentes teóricas y empíricas.\n")
        f.write("---\n\n")

    # Iterar preguntas
    total = len(questions)
    
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{total}] Procesando: {question[:50]}...")
        
        # Recuperar Documentos
        docs = retriever.invoke(question)
        
        # Formatear Contexto para el Prompt
        context_text = ""
        for doc in docs:
            source = doc.metadata.get("source", "desconocido")
            content = doc.page_content
            context_text += f"[{source.upper()}]: {content}\n\n"
        
        # Invocar Modelo
        result = chain.invoke({"context": context_text, "question": question})
        
        # Generar Bloque de Fuentes
        sources_md = write_markdown(docs)
        
        # Escribir al archivo append
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"##  Pregunta {i}: {question}\n\n")
            f.write(result)
            f.write("\n")
            f.write(sources_md)
            f.write("\n<br>\n\n") # Espacio extra entre preguntas
        
        print(f" Respuesta {i} guardada.\n")

    print(f"\n ¡Proceso terminado! Revisa el archivo '{output_file}'.")

if __name__ == "__main__":
    # Asegúrate de tener pandas para el timestamp, o borra la linea del timestamp si da error
    try:
        import pandas as pd
    except ImportError:
        pass 
        
    batch_process()