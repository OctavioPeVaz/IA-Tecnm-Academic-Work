# Contextualización del Proyecto

## 1. Descripción del Objetivo y Alcance del Proyecto
### 1.1. Introducción y Contexto
El presente proyecto se sitúa en la intersección entre la filosofía contemporánea y la inteligencia artificial avanzada. Su propósito fundamental es utilizar herramientas tecnológicas de vanguardia para abordar interrogantes profundamente humanos que definen nuestra era actual. En concreto, el proyecto busca explorar si la Generación Z está experimentando una crisis de sentido derivada de la hiperconectividad y si, simultáneamente, existe una erosión gradual de la autonomía humana frente a la presencia de los algoritmos.

Para lograr esto, no se recurre únicamente a la especulación teórica, sino a la implementación técnica de un sistema RAG (Retrieval-Augmented Generation) utilizando el modelo de ollama phi4. Este sistema permite integrar análisis cualitativo y cuantitativo, minería de textos y recuperación semántica para generar interpretaciones fundamentadas sobre cómo la tecnología moldea la identidad y la libertad.


### 1.2. Objetivo General
El objetivo central es desarrollar un sistema de Inteligencia Artificial capaz de "leer", procesar y sintetizar grandes volúmenes de datos (artículos académicos, discursos en redes sociales y comentarios de plataformas digitales) para responder a una pregunta crítica: ¿Cómo se manifiesta la crisis de sentido en la Generación Z y qué papel juegan los algoritmos en la pérdida o transformación de su autonomía?.

El sistema RAG tiene la función específica de recuperar información contextual desde bases de datos vectorizadas para fundamentar las respuestas del modelo generativo, evitando alucinaciones y asegurando que las conclusiones estén ancladas en discursos reales y marcos teóricos sólidos.


### 1.3. Ejes de Investigación y Fundamentación Filosófica
El proyecto no es meramente técnico; está sostenido por una arquitectura teórica robusta que guía el análisis del RAG. Se divide en dos grandes problemáticas que el modelo debe ser capaz de identificar y analizar:

#### A. La Crisis de Sentido en la Generación Z
El primer eje investiga la hipótesis de que el exceso de información y la falta de proyectos compartidos han generado un vacío existencial. El sistema RAG está diseñado para detectar patrones discursivos relacionados con:

Existencialismo Digital (Sartre y Camus): Se busca identificar si las redes sociales actúan como un catalizador del "vacío existencial" o la náusea, donde la libertad de elección infinita en plataformas digitales conduce a la parálisis o la angustia.

La Posmodernidad y el Fin de los Metarrelatos (Lyotard): Se analiza si la Generación Z carece de grandes narrativas (religión, ideología política, progreso histórico) que den estructura a su vida, viviendo en un mundo fragmentado de "micro-relatos" efímeros.

Identidad Líquida (Zygmunt Bauman): El proyecto examina la volatilidad de los compromisos y la identidad. Se busca evidencia de cambios constantes en la auto-percepción y la dificultad para establecer vínculos duraderos, característicos de una "modernidad líquida".

La Sociedad del Cansancio (Byung-Chul Han): Un punto clave es la detección de discursos sobre el burnout juvenil. Se investiga si la presión por el rendimiento y la autoexplotación (presentarse siempre feliz y exitoso en redes) está generando ansiedad y depresión.


##### Ángulos de análisis específicos para el RAG:

¿Cómo influyen TikTok y sus algoritmos en la construcción del "Yo"?.

La tensión entre el narcisismo digital y la búsqueda de autenticidad.

La relación entre la ilusión de conexión digital y la soledad real.

#### B. Tecnología, IA y la Desaparición de la Autonomía Humana
El segundo eje cuestiona si estamos cediendo nuestra capacidad de decisión a sistemas automatizados. El RAG debe contrastar la percepción de libertad de los usuarios con la realidad de los mecanismos de control:

Biopoder y Vigilancia (Michel Foucault): Se analiza el entorno digital como un panóptico moderno donde la vigilancia algorítmica y el control de datos ejercen un poder sobre los cuerpos y las conductas de los usuarios.

La Tecnología como "Desocultamiento" (Martin Heidegger): Se explora la idea de que la tecnología no es neutral, sino que transforma al ser humano en un "recurso" o dato a ser procesado para la eficiencia, ocultando otras formas de ser.


Erosión del Espacio Público (Jürgen Habermas): Se investiga si las redes sociales han destruido el espacio público racional, sustituyéndolo por cámaras de eco donde no hay debate real, sino polarización.

##### Ángulos de análisis específicos para el RAG:

¿Decide el usuario qué ver o deciden los algoritmos de recomendación (Netflix, YouTube)?.


El impacto de las "burbujas de filtro" en el pensamiento crítico.


La capacidad de la IA para crear deseos, hábitos o creencias artificiales en los usuarios.


### 1.4. Justificación de la Metodología RAG
La elección de una arquitectura RAG (Retrieval-Augmented Generation) es estratégica para este proyecto. Dado que los modelos de lenguaje (LLMs) por sí solos pueden carecer de contexto específico sobre eventos recientes o datos de nicho, el RAG permite:


Recuperación Semántica: Buscar fragmentos relevantes en un corpus específico (libros filosóficos, tweets, comentarios de YouTube) que coincidan con la intención de la pregunta.

Contextualización: Inyectar estos fragmentos reales en el prompt del modelo phi4.

Interpretación Fundamentada: Obligar al modelo a generar respuestas ("interpretaciones") basadas estrictamente en la evidencia recuperada, permitiendo contrastar la teoría filosófica con la realidad empírica de los datos recolectados.



## 2. Generación, Estructuración y Procesamiento del Dataset
Para alimentar el sistema RAG y asegurar que el modelo phi4 pudiera generar respuestas contextualizadas y no alucinaciones, fue necesario construir una base de conocimientos robusta y heterogénea. Esta fase del proyecto se centró en la curación de datos provenientes de fuentes teóricas (filosofía) y fuentes empíricas (discurso digital real), tal como se especifica en los requisitos metodológicos del proyecto.

### 2.1. Selección del Formato de Almacenamiento: El Estándar CSV
Para la persistencia y manipulación de los datos se seleccionó el formato CSV (Comma-Separated Values). Esta decisión técnica se fundamenta en varios pilares clave para un flujo de trabajo de Inteligencia Artificial y Ciencia de Datos:

Interoperabilidad Universal: El formato CSV es agnóstico a la plataforma. Puede ser procesado nativamente por librerías de Python como pandas, csv y, crucialmente, por los cargadores de documentos de LangChain utilizados en el pipeline del RAG.

Eficiencia en la Segmentación (Chunking): En un sistema RAG, la unidad de información no es el archivo entero, sino fragmentos de texto. El CSV permite estructurar la información en filas (donde cada fila es un documento o fragmento), facilitando que el proceso de embedding trate cada entrada como un vector independiente sin necesidad de algoritmos de parsing complejos que requerirían formatos como PDF o HTML.

Transparencia y Depuración: Al ser texto plano, permite una inspección visual rápida para verificar que la limpieza de datos (eliminación de caracteres extraños, normalización) se haya realizado correctamente antes de pasar a la vectorización.

### 2.2. Composición del Corpus de Datos
El dataset final es un híbrido compuesto por tres fuentes principales, diseñadas para cubrir los ejes teóricos y los ángulos de investigación del proyecto.


#### A. Dataset Académico: academic_articles.csv
Este archivo constituye la base intelectual del sistema. Contiene fragmentos seleccionados de literatura académica y filosófica que responden a los ejes de análisis del proyecto.

Contenido: Se extrajeron y curaron textos de autores clave como:

- Jean-Paul Sartre y Albert Camus: Para fundamentar las respuestas sobre el vacío existencial y la angustia.

- Jean-François Lyotard: Para el análisis sobre la caída de los metarrelatos en la posmodernidad.

- Zygmunt Bauman: Textos sobre la "modernidad líquida" y la fragilidad de los vínculos humanos.

- Byung-Chul Han: Fragmentos sobre la sociedad del cansancio y el burnout.

- Foucault, Heidegger y Habermas: Para los análisis sobre biopoder, tecnificación del ser y la esfera pública.

Función: Este dataset actúa como el "experto filosófico" dentro del RAG, permitiendo que la IA contraste las opiniones de redes sociales con teoría formal.

#### B. Dataset Sintético de Redes Sociales (Twitter/X)
Origen: Proporcionado como recurso base del curso.

Naturaleza: Contiene datos obtenidos mediante web scraping de Twitter (X).

Objetivo: Representar el discurso rápido, efímero y a menudo polarizado que caracteriza a la comunicación digital actual. Sirve para detectar tendencias inmediatas y patrones lingüísticos breves.

#### C. Dataset Propio: YoutubeComments Scraper
Para obtener una visión profunda y cualitativa de la opinión real de la Generación Z, se desarrolló una herramienta propia de extracción de datos (scrapper) enfocada en YouTube. A diferencia de Twitter, los comentarios de YouTube suelen contener reflexiones más extensas y debates más profundos.

Videos Seleccionados para el Scrapeo: Se eligieron estratégicamente 4 videos virales que abordan directamente las problemáticas del proyecto:

- ¿Por qué la vida es tan difícil para la Generación Z y solo empeora? (Enfoque en crisis económica y existencial).

- El peligroso papel de la IA (Enfoque en autonomía y tecnología).

- La generación Z está condenada... (Enfoque en pesimismo generacional).

- La Cruda REALIDAD de la GEN Z (Enfoque en identidad y realidad social).

Funcionamiento Técnico del Scraper: El script desarrollado (youtubecomments scrapper) no se limitó a descargar texto indiscriminadamente. Se implementó lógica de limpieza para asegurar la calidad de los datos ("Garbage In, Garbage Out"):

Extracción Automatizada: El script itera sobre las URLs de los videos objetivo, accediendo a la sección de comentarios (ya sea mediante API o simulación de navegador).

Filtrado de Ruido: Se programó para ignorar elementos que no aportan valor semántico al modelo RAG, tales como:

Comentarios vacíos o compuestos solo por emojis.

Spam y enlaces promocionales.

Marcas de tiempo (timestamps) sin contexto.

Normalización: El texto extraído se procesó para eliminar caracteres especiales innecesarios y estandarizar el formato, generando un dataset limpio donde cada fila representa una opinión humana genuina sobre los temas de ansiedad, futuro, IA y soledad.

Este proceso de recolección y limpieza profunda  aseguró que, al crear los embeddings, el sistema capturara el sentimiento y la semántica real de la crisis generacional, en lugar de ruido digital.


## 3. Implementación del Sistema RAG con Ollama y LangChain
El núcleo tecnológico del proyecto reside en una arquitectura RAG (Retrieval-Augmented Generation) ejecutada localmente. A diferencia de un chat simple con una IA, este sistema permite al modelo consultar una base de datos privada (nuestro corpus de investigación) antes de generar una respuesta.

La implementación se divide en dos procesos lógicos gestionados por dos scripts de Python: la Ingestión de Datos (create_vectors.py) y la Generación de Respuestas (3_ollama_rag_informe.py).

### 3.1. Ingestión y Vectorización (create_vectors.py)
Este script es el responsable de transformar el lenguaje humano (los CSVs generados en la fase anterior) en lenguaje matemático (vectores) que la máquina puede procesar y buscar semánticamente.

#### A. Configuración del Modelo de Embeddings

```embeddings = OllamaEmbeddings(model="mxbai-embed-large")```

Decisión Técnica: Se seleccionó el modelo mxbai-embed-large corriendo sobre Ollama.

Justificación: A diferencia de modelos genéricos, mxbai está optimizado específicamente para tareas de recuperación (retrieval). Convierte cada comentario de YouTube y fragmento filosófico en un vector numérico de alta dimensionalidad, donde conceptos semánticamente similares (ej. "tristeza" y "vacío") se agrupan cerca en el espacio matemático.

#### B. Carga y Enriquecimiento de Metadatos
El script define una función crítica: load_csv_data.


```
def load_csv_data(filepath, text_column, metadata_columns, source_name):
    # ... lee el CSV ...
    doc = Document(
        page_content=str(row[text_column]),
        metadata=meta # Se adjunta autor, video_id y fuente
    )
```

Funcionamiento: No solo se lee el texto. El script encapsula cada fila del CSV en un objeto Document de LangChain y le adjunta metadatos (quién lo escribió, de qué video proviene o qué filósofo es). Esto es vital para que, en el informe final, la IA pueda citar si una idea viene de "Sartre" o del usuario "@usuario_genz".

#### C. Creación del Vector Store (ChromaDB)

```
vector_store = Chroma.from_documents(
    documents=all_documents,
    embedding=embeddings,
    persist_directory=db_location,
    collection_name="genz_research"
)
```

Persistencia: Se utiliza ChromaDB como base de datos vectorial persistente. Esto significa que los embeddings se guardan en el disco (./chroma_db).

Ventaja: No es necesario recalcular los vectores cada vez que se corre el sistema, ahorrando tiempo de cómputo significativo.

### 3.2. Motor de Recuperación y Generación (3_ollama_rag_informe.py)
Este script actúa como el "cerebro" del proyecto. Orquesta el flujo de información desde la pregunta hasta la respuesta final.

#### A. Inicialización del LLM (Phi-4)

```
model = OllamaLLM(model="phi4")
```

Modelo: Se utiliza Phi-4 de Microsoft, ejecutado localmente vía Ollama.

Justificación: Phi-4 es un modelo SLM (Small Language Model) con capacidades de razonamiento lógico sorprendentemente altas para su tamaño. Es ideal para ejecutar en hardware local sin sacrificar la coherencia necesaria para el análisis filosófico.

#### B. Diseño del Prompt
El script define una plantilla (chatPromptTemplate) que instruye estrictamente al modelo sobre su rol:
```
Eres un investigador experto en filosofía...
Tu objetivo es responder preguntas sintetizando...
1. MARCO TEÓRICO: Conceptos filosóficos...
2. EVIDENCIA EMPÍRICA: Datos de redes sociales...
Instrucciones:
- Utiliza la información proporcionada en el CONTEXTO...
- Siempre intenta conectar la teoría con la evidencia real...
```
Esta instrucción reduce las "alucinaciones", obligando al modelo a basarse únicamente en los datos recuperados.

#### C. El Bucle RAG (Retrieval loop)
El script itera sobre una lista predefinida de preguntas de investigación y ejecuta los siguientes pasos para cada una:

Recuperación (Retrieval):

```
docs = retriever.invoke(question)
```
El sistema busca en ChromaDB los fragmentos de texto (comentarios o teoría) que tengan mayor similitud semántica con la pregunta actual.

Inyección de Contexto: Los documentos recuperados se concatenan en un solo bloque de texto, preservando sus fuentes (ej. [YOUTUBE]: El futuro me da miedo).

Generación:

```
result = chain.invoke({"context": context_text, "question": question})
```

Se envía al modelo phi4 la pregunta original + el contexto recuperado. El modelo genera la respuesta analítica.

Finalmente, el script vuelca la respuesta generada y las fuentes utilizadas directamente en el archivo informe_final.md, automatizando la creación del reporte.

## 4. Interpretación de los Hallazgos Generados por la IA
El archivo informe_final.md contiene la síntesis producida por el modelo phi4 tras procesar las 20 preguntas de investigación contra nuestra base de datos vectorial. A continuación, se presenta un desglose detallado de las conclusiones a las que llegó el sistema, divididas por los ejes temáticos del proyecto.

### 4.1. Sobre el Vacío Existencial y la Crisis de Sentido
El sistema RAG identificó una correlación clara entre la teoría existencialista clásica y el discurso digital contemporáneo, pero con matices propios de la era digital.

El Lenguaje de la Crisis: Al analizar la Pregunta 1 (¿Qué expresiones utiliza la Gen Z...?), el modelo detectó que los usuarios no usan términos académicos como "náusea" (Sartre), sino expresiones de pesimismo y agotamiento. La IA señaló una "visión dualista" en los comentarios: una tensión constante entre el deseo de estar conectados y el dolor que esa conexión produce.

La Paradoja de la Conexión: Cruzando datos con las ideas de Byung-Chul Han, el informe destaca que la hiperconectividad no ha eliminado la soledad, sino que la ha transformado. La evidencia empírica recuperada sugiere que la "ilusión de compañía" digital genera un vacío mayor cuando la pantalla se apaga, confirmando la hipótesis de que la saturación de información impide la formación de vínculos profundos.

### 4.2. Identidad Líquida y Construcción del "Yo"
Respecto a la identidad (basado en Bauman y el análisis de redes sociales), el modelo arrojó conclusiones inquietantes sobre cómo los algoritmos moldean la autopercepción.

Identidad Fragmentada: El sistema encontró evidencia de que la identidad en la Generación Z es altamente volátil ("líquida"). Los comentarios analizados muestran que los jóvenes sienten presión para adaptar su personalidad a las tendencias virales de TikTok, lo que valida la teoría de que no hay "compromisos duraderos" con uno mismo.

Narcisismo vs. Autenticidad: El informe señala una lucha interna. Por un lado, el algoritmo premia el narcisismo digital (likes, visualizaciones); por otro, existe un discurso emergente de "búsqueda de autenticidad" donde los usuarios expresan cansancio por tener que "actuar" felicidad.

### 4.3. Autonomía Humana frente a la Inteligencia Artificial
En este eje (Foucault, Heidegger, Habermas), el análisis del RAG fue contundente respecto a la pérdida de libertad percibida.

Biopoder Algorítmico: El modelo interpretó las quejas sobre "adicción al scroll" y "recomendaciones predictivas" como una manifestación moderna del biopoder. La IA concluyó que los algoritmos no solo sugieren contenido, sino que gestionan la conducta, decidiendo qué deseos se activan en el usuario.

Erosión del Pensamiento Crítico: Basándose en los datos sobre "burbujas de filtro", el sistema concluyó que la autonomía intelectual está comprometida. Los usuarios admiten en los comentarios (dataset empírico) sentirse encerrados en bucles de opinión que refuerzan sus prejuicios, lo que Habermas identificaría como la destrucción del espacio público racional.

### 4.4. Estado Emocional: Miedos y Esperanzas (Conclusiones Finales)
El análisis de las últimas preguntas del informe (específicamente sobre el futuro y las emociones predominantes) revela un panorama complejo:

Prioridad del "Yo" (Individualismo Existencial): El RAG detectó un fuerte retorno al cuidado personal. Ante un mundo exterior caótico (crisis económica, colapso climático), la Generación Z vuelca su sentido de vida hacia el interior: "ser yo como prioridad". Esto no es solo egoísmo, sino un mecanismo de defensa existencial.

Deseo de Cambio vs. Parálisis: La IA identificó una contradicción fascinante. Existe un "Deseo de Cambio Positivo" (activismo, querer mejorar el mundo), pero a menudo choca con la parálisis generada por la sobreestimulación tecnológica. Los usuarios quieren actuar, pero el entorno digital los mantiene pasivos.

Predominio de la Ansiedad: Validando la teoría de la "Sociedad del Cansancio", el análisis de sentimientos del RAG confirmó que la ansiedad y la frustración son las emociones dominantes en el corpus, superando a la esperanza o la alegría.

### 4.5. Conclusión General
El sistema RAG con phi4 ha demostrado exitosamente que las preocupaciones filosóficas del siglo XX (Sartre, Foucault) están más vivas que nunca en el código del siglo XXI.

La conclusión final del informe generado es que la Generación Z sí está atravesando una crisis de sentido, exacerbada por una tecnología que ofrece conexión infinita pero sentido limitado. Sin embargo, no es una generación derrotada: el informe detecta una resistencia activa, un intento consciente de recuperar la autonomía y la autenticidad frente a la maquinaria algorítmica.