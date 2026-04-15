import os
import sys
import warnings
import logging
import glob
import re
from typing import List, Dict

# -----------------------------------------------------------------------------
# Configuración inicial: silenciar librerías externas para una salida limpia
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")

# Desactivar telemetría de Chroma, HuggingFace y otras herramientas
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Nivel de logging crítico + 1 para suprimir casi todo
_SILENT = logging.CRITICAL + 1
logging.root.setLevel(_SILENT)
for _h in logging.root.handlers[:]:
    logging.root.removeHandler(_h)
logging.root.addHandler(logging.NullHandler())

# Silenciar loggers específicos de librerías ruidosas
for _noisy in (
    "pypdf", "pypdf._cmap", "pypdf._page",
    "huggingface_hub", "huggingface_hub.utils._http",
    "transformers", "sentence_transformers",
    "langchain", "langchain_core", "chromadb"
):
    logging.getLogger(_noisy).setLevel(_SILENT)

# -----------------------------------------------------------------------------
# Importaciones necesarias para el RAG
# -----------------------------------------------------------------------------
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Para el LLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -----------------------------------------------------------------------------
# Configuración de directorios y variables globales
# -----------------------------------------------------------------------------
PERSIST_DIRECTORY = "./data_db"          # Donde se guarda la base vectorial
DATA_DIR = "data"                        # Carpeta con los documentos fuente
METADATA_FILE = os.path.join(DATA_DIR, "metadata.txt")   # Archivo con metadatos

# Modelo de embeddings (convierte texto a vectores)
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# Modelo LLM para generación de respuestas
# NOTA: Se utiliza Qwen2.5-1.5B-Instruct, que ofrece buen rendimiento en español
# con requisitos de hardware moderados (~3-4 GB de RAM/VRAM).
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # ← CAMBIADO según solicitud

# Parámetros de chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Número de fragmentos a recuperar y mostrar
TOP_K_RETRIEVAL = 10
TOP_K_DISPLAY = 5

# -----------------------------------------------------------------------------
# Funciones auxiliares para cargar metadatos desde metadata.txt
# -----------------------------------------------------------------------------
def parse_metadata_file(filepath: str) -> Dict[str, Dict[str, str]]:
    """
    Lee el archivo metadata.txt y extrae la información estructurada.
    Devuelve un diccionario donde la clave es el nombre del archivo (sin ruta)
    y el valor es otro diccionario con los campos: planta, sector, codigo, revision, etc.
    """
    if not os.path.exists(filepath):
        print(f"[WARN] Archivo de metadatos no encontrado: {filepath}")
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Cada bloque de metadatos está separado por líneas en blanco y comienza con "nombre del archivo:"
    blocks = re.split(r'\n\s*\n', content)
    metadata_dict = {}

    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        file_meta = {}
        filename = None
        for line in lines:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip().lower()
            val = val.strip()
            if 'nombre del archivo' in key:
                # El nombre puede venir con prefijo "nombre del archivo:"
                filename = val.replace("nombre del archivo:", "").strip()
                # Normalizar: quitar extensión si aparece dos veces y dejar solo nombre base
                filename = filename.split('.')[0] + '.txt'  # asumimos extensión .txt
            elif 'planta' in key:
                file_meta['planta'] = val
            elif 'sector' in key:
                file_meta['sector'] = val
            elif 'codigo de instructivo' in key or 'codigo' in key:
                file_meta['codigo'] = val
            elif 'revisión' in key:
                file_meta['revision'] = val
            elif 'fecha de revisión' in key or 'fecha de emisión' in key:
                if 'fecha' not in file_meta:  # priorizar fecha de revisión
                    file_meta['fecha'] = val
            elif 'aprobó' in key:
                file_meta['aprobado_por'] = val

        if filename:
            metadata_dict[filename] = file_meta

    return metadata_dict


def load_documents_with_metadata(data_dir: str, metadata_dict: Dict) -> List[Document]:
    """
    Carga únicamente archivos .txt desde data_dir.
    Para cada documento, adjunta los metadatos correspondientes del diccionario
    (si existe) y también la fuente (nombre de archivo).
    """
    raw_documents = []

    print("\n[PASO 1] Cargando archivos .txt desde la carpeta 'data'...")

    for file_path in glob.glob(os.path.join(data_dir, "*.txt")):
        if not os.path.isfile(file_path):
            continue

        # Ignorar el propio archivo de metadatos
        if os.path.basename(file_path) == "metadata.txt":
            continue

        try:
            # Cargar silenciosamente para evitar logs innecesarios
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout, sys.stderr = old_stdout, old_stderr
        except Exception as e:
            print(f"  [ERROR] No se pudo cargar {file_path}: {e}")
            continue

        # Obtener metadatos del archivo (si existen en metadata_dict)
        filename = os.path.basename(file_path)
        file_meta = metadata_dict.get(filename, {})

        # Agregar metadatos a cada documento cargado
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata.update(file_meta)

        raw_documents.extend(docs)
        print(f"  [OK] {len(docs):>4} documento(s)  ←  {filename}")

    print(f"[INFO] Total de documentos crudos cargados: {len(raw_documents)}")
    return raw_documents

# -----------------------------------------------------------------------------
# 1. Carga del modelo de embeddings (siempre necesario)
# -----------------------------------------------------------------------------
print("\n[INIT] Cargando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print(f"[OK] Modelo de embeddings '{EMBEDDING_MODEL_NAME}' cargado.\n")

# -----------------------------------------------------------------------------
# 2. Creación o carga de la base vectorial Chroma
# -----------------------------------------------------------------------------
if os.path.exists(PERSIST_DIRECTORY):
    print(f"{'─'*60}")
    print(f"[INFO] Base de datos encontrada en '{PERSIST_DIRECTORY}'")
    print(f"[INFO] Cargando DB existente... (se omite re-indexación)")
    print(f"{'─'*60}")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )
    print("[OK] Vector database cargada correctamente.\n")
else:
    print(f"{'─'*60}")
    print("[AVISO] No se encontró data_db. Construyendo desde cero...")
    print(f"{'─'*60}")

    # Cargar metadatos
    metadata_dict = parse_metadata_file(METADATA_FILE)

    # Cargar documentos con metadatos enriquecidos (solo TXT)
    raw_documents = load_documents_with_metadata(DATA_DIR, metadata_dict)

    # División en chunks (fragmentos)
    print("\n[PASO 2] Dividiendo documentos en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"[INFO] Fragmentos generados: {len(documents)}")
    print(f"[INFO] Tamaño de chunk: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")

    # Crear y persistir la base vectorial
    print("\n[PASO 3] Creando la base vectorial en disco...")
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    print(f"[OK] Base vectorial guardada en '{PERSIST_DIRECTORY}'")
    print(f"[OK] Total de chunks indexados: {len(documents)}")
    print(f"{'─'*60}\n")

# -----------------------------------------------------------------------------
# 3. Definición de la consulta del usuario
# -----------------------------------------------------------------------------
#Preguntas de recuperación directa ----------------------------------------------------------------------------------------
#query = "¿Cuál es el valor máximo de conductividad para el agua de reposición en las torres de enfriamiento?"

#Preguntas de integración ----------------------------------------------------------------------------------------

#query = "¿Qué significa el led verde encendido en el horno de enlozado y en las mesas de soldadura robótica?"


#Preguntas límite/trampa ----------------------------------------------------------------------------------------

query = "¿Cuál es el procedimiento para cambiar el aceite hidráulico de los robots Motoman?"


print("[PASO 4] Procesando consulta del usuario...")
print(f"[QUERY] {query}\n")

# -----------------------------------------------------------------------------
# 4. Búsqueda vectorial (recuperación de fragmentos relevantes)
# -----------------------------------------------------------------------------
print("[PASO 5] Realizando búsqueda por similitud vectorial...")
results = vectorstore.similarity_search_with_score(query, k=TOP_K_RETRIEVAL)
print(f"[INFO] Recuperados {len(results)} documentos candidatos.")

# Ordenar por score (distancia L2 ascendente)
results_sorted = sorted(results, key=lambda x: x[1])
seen_contents = set()
unique_results = []
for doc, score in results_sorted:
    content = doc.page_content.strip()
    if content not in seen_contents:
        seen_contents.add(content)
        unique_results.append((doc, score))
    if len(unique_results) >= TOP_K_DISPLAY:
        break

# Mostrar los fragmentos recuperados
print(f"\n{'='*70}")
print(f"  TOP {len(unique_results)} FRAGMENTOS RECUPERADOS ")
print(f"  Query: \"{query}\"")
print(f"{'='*70}\n")

for i, (doc, score) in enumerate(unique_results, 1):
    source = doc.metadata.get("source", "Desconocido")
    codigo = doc.metadata.get("codigo", "")
    revision = doc.metadata.get("revision", "")

    # Representación visual de similitud (estrellas)
    stars = max(1, 5 - round(score * 2))
    stars_str = "★" * stars + "☆" * (5 - stars)

    print(f"  #{i}  [{stars_str}]  Distancia: {score:.4f}  {'← mejor coincidencia' if i == 1 else ''}")
    print(f"  Fuente: {source}  |  {codigo} Rev. {revision}")
    print(f"  {'-'*66}")
    text = doc.page_content.strip()
    if len(text) > 700:
        text = text[:700] + " ... [truncado]"
    print(text)
    print()

print(f"{'='*70}\n")

# -----------------------------------------------------------------------------
# 5. Generación de respuesta con LLM
# -----------------------------------------------------------------------------
print("[PASO 6] Cargando modelo de lenguaje (LLM)...")
print(f"[INFO] Modelo: {LLM_MODEL_ID}")

# Cargar tokenizador y modelo
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    device_map="auto",          # Asigna automáticamente GPU/CPU
    torch_dtype=torch.float16,  # Usar FP16 para ahorrar memoria si hay GPU
    low_cpu_mem_usage=True
)

# Pipeline de generación
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,            # Baja temperatura para respuestas más precisas y técnicas
    do_sample=True,
    return_full_text=False
)

# Construir el contexto combinando los fragmentos recuperados
context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in unique_results])

# Prompt estructurado en español, instruyendo al modelo a usar solo el contexto
system_message = (
    "Eres un asistente técnico especializado en procedimientos de mantenimiento industrial de la empresa Longvie. "
    "Responde ÚNICAMENTE basándote en la información proporcionada en el CONTEXTO. "
    "Si la información no está en el contexto, indica que no puedes responder con los documentos disponibles. "
    "Sé claro, preciso y cita el código del instructivo cuando sea relevante."
)

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": f"CONTEXTO:\n{context_text}\n\nPREGUNTA:\n{query}"}
]

# Aplicar plantilla de chat del modelo
final_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Generar respuesta
print("[INFO] Generando respuesta... (puede tardar unos segundos)")
output = pipe(final_prompt)
response = output[0]['generated_text'].strip()

# Limpiar tokens residuales
response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

print(f"\n{'='*70}")
print("  RESPUESTA GENERADA POR EL LLM ")
print(f"{'='*70}")
print(f"  PREGUNTA: {query}")
print(f"  {'-'*66}")
print(response)
print(f"{'='*70}\n")