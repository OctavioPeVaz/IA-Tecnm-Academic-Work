from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import json



embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./_projects/3_ollama_rag/chroma_db"

# Función para cargar CSVs
def load_csv_data(filepath, text_column, metadata_columns, source_name):
    docs = []
    if os.path.exists(filepath):
        print(f"Cargando {source_name} desde CSV...")
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            # Construir metadatos dinámicamente
            meta = {"source": source_name}
            for col in metadata_columns:
                if col in row:
                    meta[col] = str(row[col])
            
            # Crear documento
            if text_column in row and pd.notna(row[text_column]):
                doc = Document(
                    page_content=str(row[text_column]),
                    metadata=meta
                )
                docs.append(doc)
    else:
        print(f"Advertencia: No se encontró {filepath}")
    return docs

# --- LÓGICA PRINCIPAL ---

# Verificar si la base de datos ya existe para no duplicar
if not os.path.exists(db_location):
    print("Creando nueva base de datos vectorial...")
    all_documents = []

    
    all_documents.extend(load_csv_data(
        "./datasets/corpus/academic_articles.csv", 
        text_column="contenido_fragmento", 
        metadata_columns=["autor_concepto", "eje_analisis"], 
        source_name="teoria_filosofica"
    ))
    all_documents.extend(load_csv_data(
        "./datasets/corpus/dataset_sintetico_5000_ampliado.csv", 
        text_column="texto", 
        metadata_columns=["tema", "sentimiento"], 
        source_name="redes_sociales_sintetico"
    ))
    all_documents.extend(load_csv_data(
        "./datasets/corpus/comentarios_¿PorquelavidaestandificilparalaGeneracionZysoloempeora.csv", 
        text_column="texto", 
        metadata_columns=["author", "video_id"], 
        source_name="youtube_comentarios"
    ))
    all_documents.extend(load_csv_data(
        "./datasets/corpus/comentarios_El peligrosopapeldelaIA.csv", 
        text_column="texto", 
        metadata_columns=["author", "video_id"], 
        source_name="youtube_comentarios"
    ))
    all_documents.extend(load_csv_data(
        "./datasets/corpus/comentarios_LaCrudaREALIDADdelaGENZ.csv", 
        text_column="texto", 
        metadata_columns=["author", "video_id"], 
        source_name="youtube_comentarios"
    ))
    all_documents.extend(load_csv_data(
        "./datasets/corpus/comentarios_LageneracionZestacondenada.csv", 
        text_column="texto", 
        metadata_columns=["author", "video_id"], 
        source_name="youtube_comentarios"
    ))

    print(f"Total de documentos procesados: {len(all_documents)}")

    
    vector_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=db_location,
        collection_name="genz_research"
    )
    print("Base de datos creada exitosamente.")

else:
    print("La base de datos ya existe. Conectando...")
    vector_store = Chroma(
        persist_directory=db_location,
        embedding_function=embeddings,
        collection_name="genz_research"
    )

# Exponer el retriever para que lo use el otro script
retriever = vector_store.as_retriever(
    search_type="mmr", # "mmr" busca diversidad en las respuestas, no solo similitud
    search_kwargs={"k": 10} # Recuperar 10 fragmentos para tener contexto suficiente
)