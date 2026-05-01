# Sesión 1 — Slides
## GenAI Multimodal con Google Gen AI SDK

**Rodrigo López Vera** | Data, Revolut Perú  
Economista PUCP · MicroMasters MIT en Data Science  
+5 años en ML aplicado a finanzas

---

## Quién soy (versión rápida)

- **Ahora:** Data en Revolut Perú. Único de data en la operación local.
- **Antes:** Sub Gerente AI Engineer en BCP → primera solución GenAI pro-code del banco (agente con Unity Catalog, redujo análisis de pricing de 1-3h a <60 segundos).
- **Antes de eso:** Consultor Senior ML en EY FSO → equipos de data en 4 países. 

No vengo a enseñarles teoría. Vengo a mostrarles lo que funciona.
---

## El problema con los LLMs si vienes de ML clásico

**Lo que ya sabes (ML discriminativo):**
```
X → f(X) → ŷ
```
- Función determinista (con un seed)
- Optimizas una métrica clara
- El output es una categoría o un número

**Lo que cambia con LLMs (modelos generativos):**
```
prompt → P(token | tokens_anteriores) → secuencia de tokens
```
- La generación es **estocástica** — el mismo input puede dar outputs distintos
- No optimizas una métrica directamente: usas RLHF, DPO, etc.
- El output es texto que puede representar cualquier cosa: JSON, código, prosa

**Para nosotros (usuarios de la API):** la diferencia práctica es `temperature`.  
`temperature=0` → determinista. `temperature=1` → creativo. En compliance: siempre 0.

---

## ¿Qué significa "multimodal"?

Un modelo multimodal puede procesar y/o generar **múltiples tipos de datos**:

| Modalidad | Ejemplo en fintech |
|-----------|-------------------|
| Texto | Transacciones, circulares SBS, contratos |
| Imagen | Vouchers, estados de cuenta, capturas de Yape |
| Audio | Llamadas de atención al cliente, verificación de voz |
| PDF | Circulares de SPIJ, informes financieros |
| Video | *(no cubrimos hoy)* |

**A nivel técnico:** todo se tokeniza. Un PDF de 10 páginas son ~5,000-8,000 tokens.  
Un voucher de 480x400 pixels son ~300-500 tokens.  
Los tokens tienen costo.

---

## Familia Gemini — cuándo usar qué

| Modelo | Velocidad | Costo | Úsalo cuando... |
|--------|-----------|-------|-----------------|
| Gemini 2.0 Flash | Muy rápida | ~$0.075/1M tokens input | Producción, extracción de datos, RAG, agentes |
| Gemini 2.5 Flash | Rápida | Algo más que 2.0 | Necesitas mejor razonamiento sin pagar Pro |
| Gemini 1.5 Pro | Lenta | ~$1.25/1M tokens input | Análisis de documentos muy largos, tareas complejas |
| Gemini 2.5 Pro | Lenta | Caro | Benchmarks, demos, cuando la calidad es crítica |

**En este curso:** `gemini-3.1-flash-lite-preview`. En producción en Revolut: `lo-que-el-presupuesto-aguante`.  
Pro lo usamos solo para análisis de contratos >100 páginas donde el contexto importa.

---

## El ecosistema Google — por qué el SDK directamente

| Opción | Para quién | Trade-off |
|--------|------------|-----------|
| **AI Studio** (aistudio.google.com) | Explorar, testear prompts | No es para producción, límites bajos |
| **Google Gen AI SDK** (`google-genai`) | Developers, producción | Lo que usamos hoy |
| **Vertex AI SDK** | Empresas grandes, GCP enterprise | Más control, más setup, IAM, precios por volumen |
| **LangChain / LlamaIndex** | Quieres abstracciones sobre el SDK | Más magic, menos control sobre el loop de tokens |

**Nuestra posición:** SDK directamente. Sin abstracciones innecesarias.  
Cuando entiendes el SDK, entiendes LangChain. Al revés no siempre.

---

## Setup rápido — lo que vamos a correr

```python
# 1. Instalar
pip install google-genai

# 2. Crear cliente
from google import genai
client = genai.Client(api_key="AIza...")

# 3. Generar texto
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="¿Qué es SWIFT?"
)
print(response.text)

# 4. Generar con parámetros
from google.genai import types
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Dame 3 señales de lavado de activos",
    config=types.GenerateContentConfig(
        temperature=0.0,
        system_instruction="Eres analista de compliance del BCP."
    )
)
```

Eso es todo. Sin `ChatOpenAI()`, sin chains, sin configuración de 50 parámetros.

---

## Qué vamos a construir en las próximas 4 horas

```
[Bloque A — 120 min]
  → SDK setup + primeras llamadas
  → Parámetros: temperature, system_instruction, response_schema
  → [EJERCICIO 1] Extraer JSON de transacciones financieras

  → Inputs multimodales: imágenes + PDFs
  
[Break — 20 min]

[Bloque B — 100 min]
  → [EJERCICIO 2] Voucher → JSON estructurado
  → Embeddings: text-embedding-004
  → Similitud coseno en numpy
  
  → [CASO PRÁCTICO] Compliance sin RAG:
     10 fragmentos SBS + query → top-3 manual → Gemini → respuesta
     
  → Reflexión: ¿por qué esto no escala? → preview sesión 2
```

**Abrir ahora:** `00_setup_check.ipynb` → confirmar que el entorno funciona.

---

## Una nota sobre el ritmo

Este no es un curso donde el instructor habla 3 horas y tú copias.

El 70% del tiempo vas a estar corriendo código.  
Cuando algo no funciona → es una pregunta válida, no un error tuyo.  
Cuando algo funciona → intenta romperlo.

Las respuestas de Gemini son no-deterministas con T>0. No existe "la respuesta correcta" en los ejercicios. Existe "funciona, entiendo por qué, y sé cuándo usar T=0".

---

*Abrir el notebook*
