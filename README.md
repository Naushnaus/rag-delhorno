# RAG DEL HORNO · asistente de menú con LangChain + Claude

> Retrieval-Augmented Generation over a real restaurant menu (Tuxtla Gutiérrez, Mexico). Python, LangChain, local embeddings, Chroma vector store and the Claude API called from my own code, with a 20-question evaluation set.

Asistente que contesta preguntas de clientes sobre el menú de **DEL HORNO · Carnitas, Chamorros y Lechones**, un negocio real de Tuxtla. Responde solo con lo que dice el menú, suma cuentas y, cuando no tiene un dato (horario, precio del lechón), lo dice y manda al WhatsApp en lugar de inventarlo.

## Arquitectura

```
data/menu_delhorno.md
        │  1. INGESTA (ingest.py)
        ▼
MarkdownHeaderTextSplitter ── un chunk por producto (+ RecursiveCharacterTextSplitter de respaldo)
        │
        ▼
Embeddings locales (FastEmbed · paraphrase-multilingual-MiniLM-L12-v2, 384 dimensiones, corre en la PC)
        │
        ▼
Chroma (vector store persistido en ./vectorstore, distancia coseno)

pregunta del cliente (ask.py)
        │  2. RECUPERACIÓN
        ▼
búsqueda híbrida: vectores (Chroma) + BM25 (palabras exactas) → fusión RRF → top-k (k = 4)
        │  3. GENERACIÓN
        ▼
API de Claude (SDK oficial anthropic) con prompt de sistema + <contexto> → respuesta
        │
        ▼
respuesta + fuentes + tokens + costo en USD → logs/consultas.jsonl
```

| Pieza | Archivo | Qué hace |
|---|---|---|
| Configuración | `rag/config.py` | Lee `.env`; modelo, top-k, tamaño de chunk, tabla de precios |
| Embeddings | `rag/embeddings.py` | Adaptador propio de FastEmbed a la interfaz `Embeddings` de LangChain |
| Vector store | `rag/vectores.py` | Abre la colección de Chroma |
| Ingesta | `ingest.py` | Carga, trocea, vectoriza y guarda (idempotente: borra y recarga) |
| Búsqueda híbrida | `rag/hibrida.py` | BM25 propio + Reciprocal Rank Fusion |
| Recuperación + generación | `rag/generador.py` | Busca, arma el contexto, llama a Claude, maneja errores, calcula costo |
| CLI | `ask.py` | Una pregunta o modo conversación; `--ver-contexto` muestra los chunks usados |
| Evaluación | `evaluar.py` + `tests/preguntas_prueba.json` | 20 preguntas con respuesta esperada; % de aciertos de recuperación y de respuesta |

### Decisiones y por qué

- **Chunk por producto.** El menú ya viene ordenado por secciones; cortar por encabezado deja cada precio junto a su producto y evita que un chunk mezcle precios de dos platillos.
- **Embeddings locales.** Cero costo por consulta y nada sale de la PC en la etapa de búsqueda. El modelo es multilingüe porque los clientes escriben en español coloquial ("¿a cómo el taco?").
- **Búsqueda híbrida (vectores + BM25).** En la primera prueba real, solo con vectores, "¿cuánto cuesta el chamorro?" no trajo la sección Chamorro: el modelo de embeddings no conoce bien nombres como "chamorro" o "lechón" y se fue por "cuesta/precio". Claude respondió bien ("no tengo ese dato") en lugar de inventar, pero la respuesta era inútil. Se agregó BM25 (premia la palabra exacta) y se fusionaron los dos rankings con RRF. Recuperación con las 20 preguntas: **16/20 → 20/20** con top-4.
- **Claude por SDK oficial, no por un wrapper.** Control directo de reintentos (`max_retries=3` para 429/5xx), timeout, errores tipados (llave inválida, modelo inexistente, límite de uso) y lectura de `usage` para el costo por llamada.
- **Respaldo ante rechazos.** Con Opus/Sonnet se activa `fallbacks="default"`: si un filtro de seguridad rechaza la petición, el servidor la reintenta con otro modelo. Si aun así hay `stop_reason == "refusal"`, se contesta un mensaje seguro.
- **Anti-alucinación.** El prompt de sistema prohíbe inventar; la evaluación tiene preguntas trampa (horario no publicado, lechón sin precio, pizza que no venden).
- **Escala.** Este menú cabe completo en el contexto del modelo; el RAG se justifica cuando el catálogo crece (por ejemplo, un inventario de refacciones con miles de piezas). La arquitectura es la misma: solo cambia lo que hay en `data/`.

## Cómo correrlo (Windows, PowerShell)

Requisitos: Python 3.10 a 3.13 y una API key de Anthropic.

```powershell
cd C:\NEO\JARVIS\proyecto-encore\e2-rag-delhorno
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # luego abre .env y pega tu ANTHROPIC_API_KEY
python ingest.py                # la primera vez descarga el modelo de embeddings (~220 MB)
python ask.py "¿cuánto cuesta el chamorro?"
python ask.py --ver-contexto "si pido 3 tacos con queso y 2 aguas, ¿cuánto es?"
python evaluar.py --solo-recuperacion   # gratis
python evaluar.py                       # completo, llama a Claude 20 veces
```

## Evaluación

`tests/preguntas_prueba.json` tiene 20 casos. Cada uno dice qué sección debe recuperarse y qué debe (o no debe) aparecer en la respuesta. `evaluar.py` reporta:

- **Recuperación:** en cuántas preguntas el top-k trajo la sección correcta.
- **Respuestas correctas:** en cuántas la respuesta contiene lo esperado y no contiene lo prohibido.
- **Costo total** de la corrida en USD.

Resultados: ver `resultados/` después de correrlo (no se suben al repo; se reportan aquí).

| Fecha | Modelo | Recuperación | Respuestas | Costo |
|---|---|---|---|---|
| 2026-10-15 | solo vectores | 16/20 (80 %) | — | — |
| 2026-10-15 | híbrida (vectores + BM25) | 20/20 (100 %) | _pendiente_ | _pendiente_ |

## Configuración (`.env`)

| Variable | Defecto | Para qué |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Tu llave. Nunca se sube (está en `.gitignore`). |
| `MODELO_CLAUDE` | `claude-opus-5-5` | Puedes probar `claude-haiku-5-5` (mucho más barato) y comparar con `evaluar.py`. |
| `ESFUERZO` | `low` | Cuánto razona el modelo antes de contestar. Para un menú, `low` basta. |
| `TOP_K` | `4` | Cuántos chunks se mandan como contexto. |

## Datos

`data/menu_delhorno.md` se armó con el menú impreso de octubre 2026 y los precios que declaró el negocio el 3 y 9 de octubre de 2026. Lo que el negocio no ha publicado (horario, formas de pago, precio del lechón) está marcado como no publicado a propósito.

---
Hecho por PERMANENC.IA Market Solutions · Tuxtla Gutiérrez, Chiapas.
