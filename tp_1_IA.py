import os
import sys
import warnings
import logging
import glob

# --- CONFIGURACIÓN DE SILENCIO (Para limpiar la consola) ---
warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

_SILENT = logging.CRITICAL + 1
logging.root.setLevel(_SILENT)

# --- IMPORTACIONES DE IA ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- CONFIGURACIÓN DE RUTAS ---
persist_directory = "./longvie_db" # Carpeta para la DB de Longvie
data_dir = "data"                  # Carpeta donde están tus 5 PDFs

# ------------------------------------------------------------
# 1. CARGAR MODELO DE EMBEDDINGS (Gratis y Local)
# ------------------------------------------------------------
print("\n[1/5] Cargando modelo de lenguaje matemático (Embeddings)...")
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# ------------------------------------------------------------
# 2. PROCESAMIENTO DE DOCUMENTOS (Carga e Indexación)
# ------------------------------------------------------------
if os.path.exists(persist_directory):
    print(f"[INFO] Cargando base de datos técnica desde {persist_directory}...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print(f"\n[2/5] Construyendo base de conocimientos desde los PDFs de Longvie...")
    raw_documents = []
    
    # Busca todos los PDF en la carpeta 'data'
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    for file_path in pdf_files:
        print(f"  -> Procesando: {os.path.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        raw_documents.extend(loader.load())

    # Fragmentación: Chunks de 700 para no perder precisión en los números
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # Creación de la base de datos vectorial
    vectorstore = Chroma.from_documents(
        documents, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    print(f"[OK] {len(documents)} fragmentos técnicos guardados en disco.")

# ------------------------------------------------------------
# 3. CONSULTA DEL USUARIO (Aquí puedes probar tus 18 preguntas)
# ------------------------------------------------------------
# Ejemplo: Pregunta de integración basada en tus archivos
query = "Compare el significado del Led Verde en el panel del horno de enlozado frente al panel de las mesas de soldadura robótica."

print(f"\n[3/5] Procesando pregunta técnica: {query}")

# ------------------------------------------------------------
# 4. BÚSQUEDA TÉCNICA (Retrieval)
# ------------------------------------------------------------
# Traemos los 7 fragmentos más relevantes (k=7)
results = vectorstore.similarity_search_with_score(query, k=3)
context_text = "\n\n".join([doc.page_content for doc, score in results])

# ------------------------------------------------------------
# 5. GENERACIÓN DE RESPUESTA (LLM Qwen2.5 - Gratis y Local)
# ------------------------------------------------------------
print("\n[4/5] Generando respuesta basada en manuales oficiales...")

model_id = "Qwen/Qwen2.5-1.5B-Instruct" # Modelo ultra-ligero para tu PC
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=400,
    temperature=0.2, # Baja temperatura para que sea PRECISO y no "creativo"
    do_sample=True,
    return_full_text=False
)

# Prompt adaptado a la consigna de Longvie
messages = [
    {
        "role": "system", 
        "content": "Eres un Asistente Técnico de Planta de Longvie S.A. Responde utilizando ÚNICAMENTE el contexto proporcionado. Si la información no está en el contexto, responde únicamente que no tienes evidencia suficiente."
    },
    {
        "role": "user", 
        "content": f"CONTEXTO TÉCNICO:\n{context_text}\n\nPREGUNTA:\n{query}"
    }
]

final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output = pipe(final_prompt)
response = output[0]['generated_text'].strip()

print(f"\n{'='*70}")
print(" RESPUESTA DEL ASISTENTE TÉCNICO LONGVIE ")
print(f"{'='*70}")
print(f"PREGUNTA: {query}")
print(f"{'-'*66}")
print(response)
print(f"{'='*70}\n")
