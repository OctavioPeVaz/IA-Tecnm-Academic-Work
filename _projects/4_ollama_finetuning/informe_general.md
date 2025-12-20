## 1. Definición del Proyecto y Objetivos Pedagógicos
El presente proyecto, "Fine-Tuning de un Tutor Inteligente de Algoritmos", nace para solucionar la brecha entre la teoría estática de los libros y la necesidad de explicaciones dinámicas y adaptativas. A diferencia de un modelo genérico, este proyecto busca especializar un LLM (Large Language Model) mediante técnicas de Fine-Tuning, transformándolo en un mentor virtual capaz de ofrecer analogías claras, ejemplos de código y retroalimentación pedagógica.

Objetivo General
Entrenar, validar y desplegar un modelo de lenguaje ajustado (Fine-Tuned) que actúe como un tutor especializado en algoritmia. El sistema debe ser capaz de:

- Explicar conceptos complejos (ej. Dijkstra, BFS/DFS) con lenguaje accesible.

- Proveer ejemplos de código en Python optimizados y comentados.

Responder dudas conceptuales con analogías pedagógicas, evitando el lenguaje excesivamente técnico cuando no sea necesario.

3. Justificación Técnica: ¿Por qué Fine-Tuning y no RAG?
Aunque los sistemas RAG son excelentes para recuperar datos, este proyecto requiere modificar el comportamiento y el estilo de respuesta del modelo.

Adaptación de Estilo: Se requiere que el modelo adopte la "persona" de un profesor paciente y claro, no solo que de definiciones.

Internalización del Conocimiento: Los algoritmos fundamentales deben formar parte de la "intuición" del modelo para que pueda razonar sobre ellos, en lugar de simplemente buscar definiciones en una base de datos.

Eficiencia: Un modelo fine-tuned (especialmente modelos pequeños como Phi-3.5) puede correr localmente con menor latencia que un sistema RAG complejo que requiere búsquedas vectoriales constantes.




## 2. Dataset

Construcción del Dataset Educativo (algoritmos_fixed.jsonl)
El éxito de un modelo educativo no reside en el algoritmo de entrenamiento, sino en la calidad pedagógica de los datos que consume. Para este proyecto, se curó manualmente un dataset especializado (algoritmos_fixed.jsonl) diseñado para replicar la interacción entre un profesor experto.

### 2.1. Selección del Formato: JSONL
A diferencia del Proyecto 3 que usaba CSV, para el Fine-Tuning de generación de código se seleccionó el formato JSONL.

Manejo de Código: Los algoritmos requieren bloques de código en Python con indentación y saltos de línea complejos. El formato CSV suele romper estos bloques, mientras que JSONL encapsula perfectamente los caracteres especiales (\n, \t) dentro de la estructura JSON.

Eficiencia de Streaming: Al entrenar modelos de lenguaje, el formato permite leer el archivo línea por línea sin cargar todo el dataset en la memoria RAM, lo cual es crucial para la eficiencia del script train_lora.py.

### 2.2. Estructura 
Cada entrada del dataset sigue un esquema estricto de Instrucción Supervisada, modelando el comportamiento deseado:


```
{
  "instruction": "[Tema] - Pregunta específica del estudiante",
  "output": "Explicación pedagógica + Ejemplo de código + Analogía"
}
```
Esta estructura enseña al modelo que, ante una duda técnica ("instruction"), no debe responder con un simple "sí/no", sino desarrollar una explicación completa ("output").


## 3. Metodología de Entrenamiento y Ajuste Fino (Fine-Tuning)
Implementación Técnica del Entrenamiento (train_lora.py)

Para transformar el modelo base en un Tutor Inteligente, se implementó un pipeline de entrenamiento supervisado utilizando la biblioteca Hugging Face Transformers y técnicas de optimización de memoria (PEFT).

### 3.1. Selección del Modelo Base
El script de entrenamiento define como base al modelo:

```
base_model = "microsoft/Phi-3.5-mini-instruct"
```


Phi-3.5 es un modelo "pequeño" (3.8B de parámetros) pero con un razonamiento lógico superior a modelos más grandes (como Llama-2 7B) en tareas de código y matemáticas. Esto lo hace ideal para explicar algoritmos sin requerir hardware industrial.

Instruct-Tuned: Al ser una variante "Instruct", ya sabe seguir órdenes básicas, por lo que el fine-tuning se enfoca en especializarlo en pedagogía, no en enseñarle a hablar desde cero.

### 3.2. LoRA
```
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ...
)
```
Funcionamiento: Se congelan los pesos del modelo original y se inyectan matrices pequeñas (adaptadores) en las capas de atención (q_proj, v_proj, etc.). Solo se entrenan estas matrices pequeñas.


### 3.3. Configuración de Hiperparámetros

Learning Rate (lr = 2e-4): Una tasa de aprendizaje relativamente alta para LoRA (típicamente es 1e-4 o 2e-4). Esto permite que el modelo se adapte rápidamente al nuevo estilo "profesor" sin olvidar su conocimiento base de Python.

Epochs (3): Se iteró 3 veces sobre el dataset completo. Un número mayor podría causar overfitting (memorizar las respuestas de los algoritmos en lugar de entender la lógica).
```
Gradient Accumulation (8) + Batch Size (1):
```
Como la GPU tiene memoria limitada, procesamos 1 ejemplo a la vez (batch_size=1).

Sin embargo, para que el aprendizaje sea estable, acumulamos los errores de 8 ejemplos antes de actualizar el cerebro del modelo. Esto simula un batch size efectivo de 8, logrando estabilidad sin explotar la memoria.

### 3.4. Tokenización
```
full_ids = instruction_ids + output_ids + [tokenizer.eos_token_id]
labels = [-100] * len(instruction_ids) + output_ids + [tokenizer.eos_token_id]
```

## 4. Fusión, Conversión y Despliegue con Ollama
Compilación del Modelo Final (GGUF)
El entrenamiento LoRA genera solo un archivo pequeño de "adaptadores" (unos pocos megabytes) que contiene los cambios neuronales. Para poder usar esto en Ollama, es necesario fusionar estos adaptadores con el modelo base (Phi-3.5) y convertirlo a un formato binario optimizado.

### 4.1. El proceso de Fusión y Conversión (convert_lora_to_gguf.py)
Este script automatiza uno de los procesos más tediosos en la ingeniería de LLMs: la integración con llama.cpp.

Gestión de Dependencias (Llama.cpp): El script es autosuficiente. Verifica si la herramienta llama.cpp existe; si no, la clona automáticamente desde GitHub e instala sus requerimientos (requirements.txt). Esto asegura que el entorno de despliegue sea reproducible en cualquier máquina.

Conversión a GGUF: El script ejecuta convert_hf_to_gguf.py para transformar los tensores de PyTorch a GGUF (GPT-Generated Unified Format).

¿Por qué GGUF? Es el formato estándar actual para inferencia local en CPU/GPU. Permite que el modelo se cargue mapeando directamente la memoria (mmap), lo que lo hace instantáneo al iniciar.

Precisión (F16 vs F32): El script permite elegir la precisión. Por defecto usa f16 que reduce el tamaño del modelo a la mitad sin perder "inteligencia" perceptible respecto a f32.

### 4.2. Definición del Modelo en Ollama (Modelfile)
Una vez que tenemos el archivo binario (tutor_algoritmos.gguf), el archivo Modelfile actúa como la configuración final del sistema.



```
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>...
Análisis: Este bloque es vital. Phi-3 utiliza tokens especiales (<|system|>, <|user|>, <|assistant|>, <|end|>) para delimitar quién habla. Si no se configura esto exactamente así en el Modelfile, el modelo "alucina" y empieza a hablar consigo mismo o no sabe cuándo terminar de escribir.
```

PARAMETER temperature 0.7: Se eligió 0.7 en lugar de 0 (determinista) o 1 (creativo). Esto permite que el tutor tenga cierta variedad en sus explicaciones (creatividad) pero mantenga la precisión técnica del código.

PARAMETER num_ctx 4096: Se amplió la ventana de contexto para permitir que el modelo lea y analice fragmentos de código largos o mantenga una conversación extendida sin olvidar lo que se dijo al principio.

Prompt del Sistema (System Prompt)

```
SYSTEM "Eres un tutor experto en algoritmos... Tu objetivo es explicar conceptos complejos... usando analogías claras..."
```
Esta instrucción final "activa" el entrenamiento. Refuerza los patrones aprendidos durante el Fine-Tuning, asegurando que el modelo se mantenga en su rol de profesor paciente y no se desvíe a ser un asistente genérico.

## 5. Conclusión General del Proyecto

El modelo resultante no solo escribe código en Python, sino que explica el porqué detrás del código, utilizando analogías (como la caja de libros para O(1)) y pasos lógicos, satisfaciendo la necesidad planteada de una herramienta educativa personalizada.

Viabilidad: Al basarse en Phi-3.5 y cuantización, el tutor es una herramienta democrática que puede ejecutarse en ordenadores personales estándar, sin depender de costosas APIs en la nube.