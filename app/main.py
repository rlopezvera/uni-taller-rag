"""
app/main.py — FastAPI demo para el curso GenAI Multimodal
Instructor: Rodrigo López Vera | Revolut Perú

Este archivo es para la DEMO del instructor. Los estudiantes NO lo construyen.
Objetivo: mostrar cómo el pipeline RAG del notebook se convierte en una API deployable.

Endpoints:
  GET  /           — health check (modelos activos + doc counts)
  POST /ingest     — recibe documento (texto o imagen), lo indexa en ChromaDB
  POST /query      — pregunta + imagen opcional → RAGResponse (multipart)
  POST /query/json — pregunta en JSON puro → RAGResponse (sin imagen)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÓMO PROBAR — GUÍA RÁPIDA PARA ESTUDIANTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PASO 1 — Levanta la API
  export GOOGLE_API_KEY="AIza..."
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

PASO 2 — Ingesta documentos (necesario antes del primer query)

  # Texto: circular SBS
  curl -X POST http://localhost:8000/ingest \\
       -F "file=@data/circulares_sbs/circular_B_2244_2024.md" \\
       -F "source_id=circular_B_2244_2024" \\
       -F "date=2024-03" \\
       -F "doc_type=text"

  # Imagen: voucher de pago
  curl -X POST http://localhost:8000/ingest \\
       -F "file=@data/images/voucher_yape_001.png" \\
       -F "source_id=voucher_yape_001" \\
       -F "date=2024-06" \\
       -F "doc_type=image"

PASO 3 — Consultas

  # Query simple (JSON)
  curl -X POST http://localhost:8000/query/json \\
       -H "Content-Type: application/json" \\
       -d '{"question": "¿Qué es una operación sospechosa?"}'

  # Query con filtro de fecha
  curl -X POST http://localhost:8000/query/json \\
       -H "Content-Type: application/json" \\
       -d '{"question": "Obligaciones del oficial de cumplimiento", "date_filter": "2024-01", "n_results": 5}'

  # Query multimodal (pregunta + voucher)
  curl -X POST http://localhost:8000/query \\
       -F "question=¿Esta transferencia requiere reporte a la UIF?" \\
       -F "image=@data/images/voucher_bbva_internacional_003.png"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÓMO PROBAR DESDE PYTHON / GOOGLE COLAB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  import httpx

  BASE = "http://localhost:8000"   # reemplaza con la URL pública si aplica

  # 1. Health check
  print(httpx.get(f"{BASE}/").json())

  # 2. Ingestar circular SBS
  with open("data/circulares_sbs/circular_B_2244_2024.md", "rb") as f:
      r = httpx.post(f"{BASE}/ingest",
                     files={"file": f},
                     data={"source_id": "circular_B_2244_2024",
                           "date": "2024-03", "doc_type": "text"})
  print(r.json())  # {"status": "ok", "chunks_indexed": N, "collection": "circulares_sbs"}

  # 3. Ingestar voucher
  with open("data/images/voucher_yape_001.png", "rb") as f:
      r = httpx.post(f"{BASE}/ingest",
                     files={"file": f},
                     data={"source_id": "voucher_yape_001",
                           "date": "2024-06", "doc_type": "image"})
  print(r.json())  # {"status": "ok", "chunks_indexed": 1, "collection": "vouchers_financieros"}

  # 4. Query solo texto
  r = httpx.post(f"{BASE}/query/json",
                 json={"question": "¿Cuál es el umbral para reportar operaciones sospechosas?"})
  print(r.json())

  # 5. Query multimodal
  with open("data/images/voucher_bbva_internacional_003.png", "rb") as f:
      r = httpx.post(f"{BASE}/query",
                     data={"question": "¿Esta operación requiere reporte a la UIF?"},
                     files={"image": f}, timeout=30)
  print(r.json())

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SWAGGER UI — documentación interactiva en el navegador
  http://localhost:8000/docs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import io
import json
import os
import re
from contextlib import asynccontextmanager
from typing import Optional

import chromadb
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL = "gemini-3.1-flash-lite-preview"
EMBED_MODEL = "gemini-embedding-2"
CHROMA_PATH = "./chroma_db"

# Estos se inicializan en el lifespan para no bloquear el import
gemini_client: genai.Client = None
chroma_client: chromadb.PersistentClient = None
text_collection = None
image_collection = None


# ---------------------------------------------------------------------------
# Lifespan: inicialización al arrancar la app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa clientes al arrancar. Se ejecuta una sola vez."""
    global gemini_client, chroma_client, text_collection, image_collection

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no encontrada en variables de entorno.")

    # Inicializar Gemini
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    print("[startup] Gemini client inicializado.")

    # Inicializar ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    text_collection = chroma_client.get_or_create_collection("circulares_sbs")
    image_collection = chroma_client.get_or_create_collection("vouchers_financieros")
    print(f"[startup] ChromaDB: {text_collection.count()} docs en circulares_sbs.")
    print(f"[startup] ChromaDB: {image_collection.count()} docs en vouchers_financieros.")

    yield  # La app corre entre yield y el bloque de cleanup

    # Cleanup (opcional aquí, ChromaDB persiste solo)
    print("[shutdown] Cerrando app.")


app = FastAPI(
    title="GenAI Compliance API",
    description="API de consulta normativa SBS con RAG multimodal",
    version="1.0.0",
    lifespan=lifespan
)


# ---------------------------------------------------------------------------
# Schemas de request / response
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    date_filter: Optional[str] = None   # ej: "2024-01" para solo normas >= esa fecha
    n_results: int = 3


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence_note: str


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    collection: str


# ---------------------------------------------------------------------------
# Utilidades internas (mismas funciones que en el notebook)
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Genera embeddings en batch con text-embedding-004."""
    result = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts
    )
    return [e.values for e in result.embeddings]


def parse_md_to_articles(md_text: str, source_id: str, date: str) -> list[dict]:
    """Divide un MD de SPIJ por artículo. Un artículo = un chunk."""
    chunks = []
    pattern = re.compile(r'^#{1,3}\s*Art[ií]culo\s+[\w]+', re.MULTILINE | re.IGNORECASE)
    matches = list(pattern.finditer(md_text))

    if not matches:
        return [{"text": md_text.strip(), "source": source_id, "article": "completo", "date": date}]

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        chunk_text = md_text[start:end].strip()
        article_num = match.group(0).strip().split()[-1]
        chunks.append({"text": chunk_text, "source": source_id, "article": article_num, "date": date})

    return chunks


def retrieve_chunks(
    query: str,
    collection,
    n_results: int = 3,
    where: Optional[dict] = None
) -> list[dict]:
    """Retrieval semántico contra una colección ChromaDB."""
    query_emb = embed_texts([query])[0]

    kwargs = {
        "query_embeddings": [query_emb],
        "n_results": min(n_results, max(collection.count(), 1)),
        "include": ["documents", "metadatas", "distances"]
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


def build_rag_response(question: str, chunks: list[dict], image: Optional[Image.Image] = None) -> RAGResponse:
    """
    Construye el prompt de augmentation y llama a Gemini.
    Retorna RAGResponse estructurado.
    """
    # Construir contexto normativo
    context = "\n\n---\n\n".join([
        f"[{c['metadata']['source']} | Artículo {c['metadata']['article']}]\n{c['text']}"
        for c in chunks
    ])

    sources = [
        f"{c['metadata']['source']} · Art. {c['metadata']['article']}"
        for c in chunks
    ]

    augmented_prompt = f"""
Eres un analista de compliance del sistema financiero peruano.
Responde ÚNICAMENTE basándote en la normativa SBS proporcionada.
Cita la circular y el artículo exactos. Si la normativa no cubre la pregunta, dilo.

=== NORMATIVA SBS RECUPERADA ===
{context}

=== PREGUNTA DEL ANALISTA ===
{question}
"""

    # Contenido: imagen (si hay) + prompt
    contents = [augmented_prompt]
    if image is not None:
        contents = [image, augmented_prompt]

    response = gemini_client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=600,
            response_mime_type="application/json",
            response_schema=RAGResponse
        )
    )

    parsed = json.loads(response.text)
    if not parsed.get("sources"):
        parsed["sources"] = sources

    return RAGResponse(**parsed)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "text_docs": text_collection.count(),
        "image_docs": image_collection.count()
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    # Opción A: subir como archivo
    file: Optional[UploadFile] = File(default=None),
    # Opción B: texto directo como form field
    text_content: Optional[str] = Form(default=None),
    source_id: str = Form(default="documento_subido"),
    date: str = Form(default="2024-01"),
    doc_type: str = Form(default="text")  # "text" o "image"
):
    """
    Indexa un documento en ChromaDB.

    Acepta:
    - Archivo MD/TXT + metadata (source_id, date)
    - Texto directo como form field
    - Imagen (PNG/JPG) → Gemini la describe → embed la descripción

    Ejemplo con httpx:
        files = {"file": open("circular.md", "rb")}
        data = {"source_id": "circular_SBS_B_2244_2024", "date": "2024-03"}
        r = httpx.post("/ingest", files=files, data=data)
    """

    if doc_type == "image":
        # -- Indexación de imagen --
        if file is None:
            raise HTTPException(400, "Se requiere un archivo de imagen para doc_type='image'")

        raw = await file.read()
        image = Image.open(io.BytesIO(raw))

        # Gemini describe la imagen
        describe_prompt = """
Describe este documento financiero de forma detallada.
Incluye: tipo de documento, entidad, montos, monedas, fechas, partes involucradas,
concepto y estado. Responde en texto plano, sin markdown.
"""
        desc_response = gemini_client.models.generate_content(
            model=MODEL,
            contents=[image, describe_prompt],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        description = desc_response.text

        # Embed y guardar en colección de imágenes
        embedding = embed_texts([description])[0]
        image_collection.upsert(
            ids=[source_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[{"source": source_id, "date": date, "type": "image", "description": description[:500]}]
        )

        return IngestResponse(status="ok", chunks_indexed=1, collection="vouchers_financieros")

    else:
        # -- Indexación de texto (MD/TXT) --
        if file is not None:
            raw = await file.read()
            content = raw.decode("utf-8")
        elif text_content:
            content = text_content
        else:
            raise HTTPException(400, "Se requiere file o text_content para doc_type='text'")

        # Chunking por artículo
        chunks = parse_md_to_articles(content, source_id, date)

        if not chunks:
            raise HTTPException(400, "No se encontraron artículos en el documento.")

        # Embed e indexar en batches
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = embed_texts(texts)
            ids = [f"{c['source']}__art_{c['article']}" for c in batch]
            metadatas = [{"source": c["source"], "article": c["article"], "date": c["date"]} for c in batch]
            text_collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        return IngestResponse(status="ok", chunks_indexed=len(chunks), collection="circulares_sbs")


@app.post("/query", response_model=RAGResponse)
async def query_endpoint(
    # Soportar tanto JSON puro como multipart con imagen
    question: str = Form(...),
    date_filter: Optional[str] = Form(default=None),
    n_results: int = Form(default=3),
    image: Optional[UploadFile] = File(default=None)
):
    """
    Consulta el pipeline RAG multimodal.

    Parámetros:
        question    : pregunta del analista
        date_filter : ej '2024-01' para solo normas posteriores
        n_results   : cuántos chunks recuperar (default: 3)
        image       : voucher/imagen opcional (multipart)

    Ejemplo con httpx (solo texto):
        r = httpx.post("/query", data={"question": "¿Cuál es el umbral de reporte?"})

    Ejemplo con imagen:
        files = {"image": open("voucher.png", "rb")}
        data = {"question": "¿Requiere reporte esta operación?"}
        r = httpx.post("/query", files=files, data=data)
    """

    if text_collection.count() == 0:
        raise HTTPException(
            503,
            "La colección está vacía. Ingesta documentos primero con POST /ingest."
        )

    # Retrieval de texto normativo
    where = {"date": {"$gte": date_filter}} if date_filter else None
    chunks = retrieve_chunks(question, text_collection, n_results=n_results, where=where)

    if not chunks:
        raise HTTPException(404, "No se encontraron fragmentos relevantes para la query.")

    # Procesar imagen si se incluyó
    pil_image = None
    if image is not None:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw))

    # Construir respuesta RAG
    result = build_rag_response(question, chunks, image=pil_image)
    return result


@app.post("/query/json")
async def query_json(request: QueryRequest):
    """
    Endpoint alternativo que acepta JSON puro (sin imagen).
    Más cómodo para testing con curl o httpx.

    Ejemplo:
        r = httpx.post("/query/json", json={"question": "¿Qué es una operación sospechosa?"})
    """
    if text_collection.count() == 0:
        raise HTTPException(503, "Colección vacía. Ingesta documentos primero.")

    where = {"date": {"$gte": request.date_filter}} if request.date_filter else None
    chunks = retrieve_chunks(request.question, text_collection, n_results=request.n_results, where=where)

    if not chunks:
        raise HTTPException(404, "No se encontraron fragmentos relevantes.")

    return build_rag_response(request.question, chunks)
