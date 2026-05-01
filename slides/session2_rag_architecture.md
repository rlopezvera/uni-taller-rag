# Sesión 2 — Slides
## Multimodal RAG: del caso práctico al pipeline completo

**Rodrigo López Vera** | Head of Data, Revolut Perú  
Curso GenAI Multimodal · Mayo 2026

---

## Debrief de ayer: ¿qué dolor sentiste?

En el caso práctico de la sesión 1 construimos esto:

```python
# Para CADA query:
chunk_embeddings = embed_all_chunks()  # llama a la API N veces
similarities = [cosine_sim(query_emb, c) for c in chunk_embeddings]
top_k = sort_and_pick(similarities)
response = gemini(top_k + query)
```

**¿Qué pasa con 15,000 artículos de SPIJ?**
- 15,000 llamadas a la API de embeddings por query
- `np.argsort` sobre 15,000 vectores de 768 dims: O(n) cada vez
- Sin metadata: no sabes de qué circular vino cada fragmento
- Sin persistencia: pierdes todo al reiniciar el runtime

**Hoy:** ChromaDB resuelve todo esto en <20 líneas.

---

## Arquitectura RAG — el pipeline completo

```
INDEXACIÓN (una sola vez, offline):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Circulares SPIJ (MD/PDF)
        │
        ▼
  parse_md_to_articles()       ← chunk por artículo
        │
        ▼
  text-embedding-004           ← 768 dims por chunk
        │
        ▼
  ChromaDB.upsert()            ← guarda texto + embedding + metadata
        │
        ▼
  {source, article, date}      ← metadata para filtrar después


CONSULTA (cada query, en tiempo real):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pregunta del analista
        │
        ▼
  text-embedding-004           ← embed la query
        │
        ▼
  ChromaDB.query()             ← HNSW index → top-k en O(log n)
        │
        ▼
  Chunks + metadata            ← [fuente, artículo, fecha]
        │
        ▼
  Prompt de augmentation       ← contexto normativo + pregunta
        │
        ▼
  gemini-2.0-flash             ← genera respuesta con citación
        │
        ▼
  {"answer", "sources", "confidence_note"}
```

---

## ¿Por qué ChromaDB y no [inserta tu favorito]?

| Vector store | Dónde vive | Cuándo usarlo |
|--------------|------------|---------------|
| **ChromaDB** | Local / Docker | Desarrollo, cursos, prototipos. Lo de hoy. |
| **Pinecone** | SaaS managed | Producción sin infraestructura propia |
| **Vertex AI Vector Search** | GCP | Producción en GCP, >1M vectores, latencia baja |
| **pgvector** | PostgreSQL | Ya tienes Postgres, quieres simplicidad |
| **Weaviate** | Self-hosted / SaaS | Hybrid search, multitenancy |
| **FAISS** | En memoria | Si necesitas velocidad pura y no te importa persistencia |

**Regla práctica:** ChromaDB para desarrollo, luego migras el código de indexación/retrieval.  
La función `collection.query()` es casi idéntica en todos.

---

## Dónde falla naive RAG

**1. Chunking malo**
```
❌ Split every 512 tokens → corta un artículo en la mitad
✓ Split por artículo → la unidad semántica natural de SPIJ
```

**2. Sin metadata**
```
❌ Solo el texto → no sabes de qué norma viene
✓ {source, article, date} → puedes filtrar: "solo 2024"
```

**3. Top-k sin reranking**
```
❌ Embedding similarity es aproximada
✓ Cross-encoder reranking: pasa query + chunk → score exacto
  (lo mencionamos, hoy no lo construimos)
```

**4. Imágenes sin descripción**
```
❌ text-embedding-004 no entiende pixels
✓ Gemini describe → embed la descripción → guardar referencia a imagen original
```

---

## Multimodal RAG — la solución real

`text-embedding-004` solo acepta texto. ¿Cómo indexamos vouchers?

```python
# Paso 1: Gemini describe la imagen
description = gemini.generate_content([image, "Describe este documento..."])
# → "Voucher BBVA, transferencia SWIFT, USD 15,000, Banco Santander Uruguay..."

# Paso 2: Embed la descripción (no la imagen)
embedding = embed_texts([description])

# Paso 3: Guardar con metadata que referencia la imagen original
image_collection.upsert(
    ids=["voucher_bbva_001"],
    embeddings=[embedding],
    documents=[description],
    metadatas=[{
        "type": "swift_internacional",
        "date": "2024-04-28",
        "amount_usd": "15000",
        "image_path": "data/images/voucher_bbva_003.png"  # ← la referencia
    }]
)
```

**En retrieval:** recuperas la descripción + la referencia a la imagen original.  
Si necesitas mostrar la imagen en la respuesta, la cargas desde el path.

Esto es exactamente lo que usamos cuando no tienes acceso a multimodal embeddings propietarios (Vertex AI Multimodal Embeddings cuesta más y tiene peor latencia para la mayoría de casos).

---

## FastAPI — de notebook a API deployable

El código del notebook se convierte en una API con mínimo cambio:

```python
# En el notebook:
result = rag_query(question, image=voucher_img)

# En FastAPI:
@app.post("/query", response_model=RAGResponse)
async def query_endpoint(question: str = Form(...), image: UploadFile = File(None)):
    pil_image = Image.open(BytesIO(await image.read())) if image else None
    chunks = retrieve_chunks(question, collection)
    return build_rag_response(question, chunks, image=pil_image)
```

**Dos endpoints:**
- `POST /ingest` — sube un documento, lo chunka, lo embeddea, lo indexa
- `POST /query` — recibe pregunta + imagen opcional, retorna JSON estructurado

**El response:**
```json
{
  "answer": "Sí, requiere reporte. Según el Artículo 4 de la Circular B-2244-2024...",
  "sources": ["circular_SBS_B_2244_2024 · Art. 4", "circular_SBS_B_2244_2024 · Art. 5"],
  "confidence_note": "La respuesta está respaldada por normativa vigente de 2024."
}
```

---

## Lo que el reto pide

```
RETO: Financial Profile Interrogator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
  - Perfil de cliente (texto)
  - Pregunta de compliance
  - Voucher (imagen, opcional)

Output:
  - Respuesta con artículo exacto citado
  - Lista de fuentes
  - Nota de confianza

Query de ejemplo:
  "Este cliente tiene 3 transferencias internacionales en 48 horas
   por un total de S/. 45,000. ¿Requiere reporte a la UIF?"

El stub ya corre. Tu trabajo: extenderlo (elige una opción):
  A — Logging de queries a CSV
  B — Filtro por tipo de operación
  C — Reranking simple por longitud
  D — Comparar respuesta con y sin imagen
```

---

## Lo que haría diferente en producción (BCP / Revolut)

| Componente | Hoy | Producción |
|------------|-----|------------|
| Vector store | ChromaDB local | Vertex AI Vector Search |
| Retrieval | Top-k dense | Hybrid: dense + BM25 |
| Reranking | Ninguno | Cross-encoder (Cohere Rerank o custom) |
| Imágenes | Descripción en texto | Vertex AI Multimodal Embeddings |
| API | FastAPI local | Cloud Run + autoscaling |
| Observabilidad | Ninguna | LangSmith + dashboards de latencia y costo |
| Evaluación | Manual, a ojo | Panel de revisores + métricas automáticas (RAGAS) |

**El aprendizaje más importante de BCP:**  
El LLM es la parte más fácil. El chunking, la metadata y el feedback loop con usuarios son donde se gana o se pierde.

**En Revolut:**  
Evaluamos la calidad del retrieval semanalmente. Si el top-1 no es correcto en >20% de las queries de un topic, revisamos el chunking, no el LLM.

---

## Cierre

**Sesión 1 construiste:**
- Llamadas al SDK con control de parámetros
- Extracción estructurada de texto e imágenes
- Similitud coseno a mano → viste el dolor

**Sesión 2 construiste:**
- Pipeline RAG completo con ChromaDB
- Retrieval con metadata filtering
- Multimodal: imagen → descripción → embed
- Stub de sistema de compliance extensible

**Lo que llevas:**
- La abstracción mental: chunk → embed → store → retrieve → augment → generate
- El conocimiento de que el 80% del trabajo está antes del LLM
- Código que funciona y puedes deployar

---

*github.com/rlopezvera98/genai-multimodal-course*
