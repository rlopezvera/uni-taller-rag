# GenAI Multimodal Course — Google Gen AI SDK

Materiales del curso de posgrado en Data/IA.  
**Instructor:** Rodrigo López Vera — Data, Revolut Perú

---

## Antes de empezar: crea tu API key

1. Ve a **[aistudio.google.com](https://aistudio.google.com)**
2. Inicia sesión con tu cuenta de Google
3. Panel izquierdo → **"Get API key"** → **"Create API key"** → "Create API key in new project"
4. Copia la key (empieza con `AIza...`)
5. En Google Colab, haz clic en el ícono 🔑 (Secrets) en el panel izquierdo:
   - Nombre: `GOOGLE_API_KEY`
   - Valor: pega tu key
   - Activa el toggle **"Notebook access"**

**Problemas frecuentes:**
- `401 API_KEY_INVALID`: la key no está bien copiada o no activaste "Notebook access"
- `429 RESOURCE_EXHAUSTED`: límite gratuito. Espera 1 minuto o usa otra cuenta Google

---

## Cómo usar los notebooks

### Opción A — Abrir en Google Colab (recomendado)

1. Ve al repositorio en GitHub
2. Abre el notebook que quieras
3. Haz clic en el badge **"Open in Colab"** o copia la URL del notebook y reemplaza `github.com` por `colab.research.google.com/github`

### Opción B — Ejecutar localmente

```bash
# Paso 0 — entorno virtual
git clone https://github.com/rlopezvera98/genai-multimodal-course.git
cd genai-multimodal-course
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Paso 1 — dependencias
pip install -r requirements.txt

# Paso 2 — abrir los notebooks
jupyter notebook
```

Si ejecutas localmente, exporta tu API key como variable de entorno:
```bash
export GOOGLE_API_KEY="AIza..."
```
Y reemplaza `userdata.get('GOOGLE_API_KEY')` por `os.environ.get('GOOGLE_API_KEY')` en los notebooks.

---

## Estructura del repo

```
genai-multimodal-course/
├── notebooks/
│   ├── 00_setup_check.ipynb              ← Abre esto primero
│   ├── session1_genai_sdk_fundamentals.ipynb
│   └── session2_multimodal_rag.ipynb
├── data/
│   ├── circulares_sbs/                   ← Circulares SBS en MD (por artículo)
│   ├── images/                           ← Vouchers ficticios
│   └── transactions/                     ← Transacciones de ejemplo en JSON
├── app/
│   └── main.py                           ← FastAPI demo (solo instructor)
├── slides/
│   ├── session1_landscape.md
│   └── session2_rag_architecture.md
└── requirements.txt
```

---

## Dependencias

```
google-genai>=1.0.0    # SDK de Google para Gemini
chromadb>=0.5.0        # Vector store local
fastapi>=0.110.0       # API demo (instructor)
uvicorn>=0.29.0        # Servidor ASGI
httpx>=0.27.0          # Cliente HTTP para testing
numpy>=1.26.0          # Operaciones vectoriales
Pillow>=10.0.0         # Procesamiento de imágenes
pydantic>=2.0.0        # Schemas y validación
```

---

## Modelos usados

| Modelo | Uso |
|--------|-----|
| `gemini-3.1-flash-lite-preview` | Generación de texto, descripción de imágenes, extracción estructurada |
| `gemini-embedding-2` | Embeddings para indexación y retrieval semántico |
---

## Sesión 1 — Fundamentos del SDK

**Bloque A (120 min):**
- Landscape: modelos generativos vs discriminativos, familia Gemini
- Setup API key + primeras llamadas
- Parámetros: temperature, top_p, max_output_tokens, system_instruction
- Outputs estructurados con response_schema

**Ejercicio 1:** Extracción de entidades de transacciones financieras

**Bloque B (100 min):**
- Inputs multimodales: imágenes y PDFs
- Embeddings con text-embedding-004
- Similitud coseno en numpy

**Caso práctico:** Sistema de compliance sin RAG — similitud coseno manual

---

## Sesión 2 — Multimodal RAG

**Bloque A (120 min):**
- Arquitectura RAG: chunk → embed → store → retrieve → augment → generate
- ChromaDB: indexación con metadata, retrieval, filtros

**Ejercicio:** 5 queries + evaluación de retrieval

**Bloque B (100 min):**
- Multimodal RAG: imagen → descripción → embed
- Pipeline end-to-end con función `rag_query()`
- Demo FastAPI: endpoints `/ingest` y `/query`

**Reto final:** Financial Profile Interrogator

---

## ADICIONAL: Levantar la API demo

```bash
export GOOGLE_API_KEY="AIza..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Probar la API — Guía para estudiantes

Una vez que la API esté corriendo en `http://localhost:8000`, pruebala:

### Swagger UI (más fácil, desde el navegador)

Abre en tu navegador:

```
http://localhost:8000/docs
```

Verás todos los endpoints documentados e interactivos. Puedes ejecutar cada llamada directamente desde ahí sin escribir código.

---


### Flujo completo recomendado para la demo

```
1. Levanta la API              →  uvicorn app.main:app --reload
2. Ingesta las 3 circulares    →  POST /ingest (x3)
3. Ingesta los 5 vouchers      →  POST /ingest?doc_type=image (x5)
4. Verifica el índice          →  GET /  (debería mostrar text_docs > 0)
5. Haz queries de texto        →  POST /query/json
6. Haz queries multimodales    →  POST /query con imagen
```

---

*Mayo 2026 · Posgrado Data/IA*
