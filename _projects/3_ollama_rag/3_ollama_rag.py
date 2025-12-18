from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from create_vectors import retriever


model = OllamaLLM(model="phi4:latest") 

template = """
Eres un investigador experto en filosofía, sociología digital y análisis de datos. Estás trabajando en un proyecto titulado "La Generación Z y la Crisis de Sentido en la Era Digital".

Tu objetivo es responder preguntas sintetizando dos tipos de información que se te proporcionarán:
1. MARCO TEÓRICO: Conceptos filosóficos (Existencialismo, Posmodernidad, Foucault, Byung-Chul Han, etc.).
2. EVIDENCIA EMPÍRICA: Datos de redes sociales, comentarios de YouTube y encuestas sintéticas.

Instrucciones:
- Utiliza la información proporcionada en el CONTEXTO para responder.
- Siempre intenta conectar la teoría (ej. "Vacío existencial") con la evidencia real (ej. "Comentarios de usuarios").
- Si la información no está en el contexto, di que no tienes datos suficientes, no inventes.
- Responde siempre en Español formal y académico pero accesible.

CONTEXTO RECUPERADO:
{context}

PREGUNTA DEL USUARIO: 
{question}

RESPUESTA (Análisis estructurado):
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


print("\n~~~~~~~~~~~ Proyecto 3 (RAG): GEN Z & IA ~~~~~~~~~~")
print("\n=== La Generación Z y la Crisis de Sentido en la Era Digital ===")
print("\n=== Tecnología, Inteligencia Artificial y la Desaparición de la Autonomía Humana ===")
def chat_loop():
    
    while True:
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        question = input("Pregunta/Consulta: ")
        
        if question.lower() in ["salir", "exit", "q"]:
            break
        
        print("Buscando evidencia y teoría...")
        
        # Recuperar informacion importante
        docs = retriever.invoke(question)
        
        # Formatear el contexto para que el LLM sepa qué es qué
        context_text = ""
        for doc in docs:
            source = doc.metadata.get("source", "desconocido")
            content = doc.page_content
            # Etiquetamos el origen para que el LLM sepa distinguir teoría de opinión
            context_text += f"[{source.upper()}]: {content}\n\n"
        
        
        print("Pensando...\n")
        result = chain.invoke({"context": context_text, "question": question})
        print(result)

        print("\n--- FUENTES UTILIZADAS (EVIDENCIA REAL) ---")
        seen_sources = set()
        for doc in docs:
            meta = doc.metadata
            source_type = meta.get("source", "Desconocido")

            source_label = f"[{source_type.upper()}]"

            details = ""
            if source_type == "youtube_comentarios":
                details = f"Autor: {meta.get('author', 'Anon')} | Likes: {meta.get('like_count', 0)}"
            elif source_type == "teoria_filosofica":
                details = f"Autor: {meta.get('autor_concepto', 'N/A')} | Eje: {meta.get('eje_analisis', 'N/A')}"
            elif source_type == "articulos_externos":
                details = f"Título: {meta.get('title', 'Sin título')}"

            identifier = f"{source_label} {details}"
            if identifier not in seen_sources:
                print(f"📄 {source_label} {details}")
                # Opcional: Si quieres ver el fragmento de texto exacto, descomenta la linea de abajo
                # print(f"   Fragmento: {doc.page_content[:100]}...") 
                seen_sources.add(identifier)

if __name__ == "__main__":
    chat_loop()