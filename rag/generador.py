"""Pasos 2 y 3 del RAG: recuperar los chunks relevantes y generar la respuesta con la API de Claude."""
import json
import time
from datetime import datetime

import anthropic
from langchain_core.documents import Document

from . import config
from .hibrida import BM25, fusionar
from .vectores import abrir_vectorstore

SISTEMA = """Eres el asistente de pedidos de DEL HORNO (carnitas, chamorros y lechones) en Tuxtla Gutiérrez.
Respondes clientes por WhatsApp en español de México: corto, amable y directo.

Reglas:
- Usa SOLO la información dentro de <contexto>. No inventes productos, precios, horarios ni promociones.
- Si el dato no está en el contexto, dilo con claridad y pide confirmarlo por WhatsApp al 961 277 8843.
- Si el cliente pide varios productos, suma el total con los precios del contexto y muestra la cuenta.
- Precios siempre en pesos mexicanos con signo $.
- El lechón entero es solo por encargo y su precio no está publicado: nunca des una cifra."""


class ErrorRAG(Exception):
    """Error con mensaje listo para mostrar al usuario."""


def recuperar(pregunta: str, k: int = config.TOP_K, vectorstore=None):
    """Devuelve [(Document, puntaje)] con búsqueda híbrida: vectores + BM25 fusionados con RRF."""
    vs = vectorstore or abrir_vectorstore()
    todos = vs.get(include=["documents", "metadatas"])  # el menú es chico: BM25 se arma en memoria
    if not todos["ids"]:
        return []
    docs = {i: Document(page_content=t, metadata=m) for i, t, m in zip(todos["ids"], todos["documents"], todos["metadatas"])}

    # Ranking 1: similitud de significado (Chroma, distancia coseno)
    candidatos = min(len(docs), max(k * 3, 10))
    por_vector = [d.id for d, _ in vs.similarity_search_with_score(pregunta, k=candidatos)]

    # Ranking 2: coincidencia de palabras (BM25); solo cuentan los que tienen al menos una palabra en común
    ids = list(docs)
    bm25 = BM25([docs[i].page_content for i in ids]).puntajes(pregunta)
    por_palabra = [ids[j] for j in sorted(range(len(ids)), key=lambda j: -bm25[j]) if bm25[j] > 0]

    return [(docs[i], puntaje) for i, puntaje in fusionar([por_vector, por_palabra])[:k]]


def armar_contexto(fragmentos) -> str:
    bloques = []
    for doc, _ in fragmentos:
        bloques.append(f'<fragmento seccion="{doc.metadata.get("seccion")}">\n{doc.page_content}\n</fragmento>')
    return "\n".join(bloques)


def costo_usd(modelo: str, uso) -> float | None:
    precios = config.PRECIOS_USD_POR_MTOK.get(modelo)
    if not precios:
        return None
    return (uso.input_tokens * precios[0] + uso.output_tokens * precios[1]) / 1_000_000


def llamar_claude(pregunta: str, contexto: str):
    cliente = anthropic.Anthropic(max_retries=3, timeout=60.0)  # reintenta 429, 5xx y cortes de red
    peticion = dict(
        model=config.MODELO_CLAUDE,
        max_tokens=config.MAX_TOKENS,
        system=SISTEMA,
        output_config={"effort": config.ESFUERZO},
        messages=[{
            "role": "user",
            "content": f"<contexto>\n{contexto}\n</contexto>\n\nPregunta del cliente: {pregunta}",
        }],
    )
    if config.MODELO_CLAUDE in config.MODELOS_CON_RESPALDO:
        # Si un filtro de seguridad rechaza la petición, el servidor la reintenta con otro modelo.
        peticion.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    try:
        return cliente.beta.messages.create(**peticion)
    except anthropic.AuthenticationError:
        raise ErrorRAG("La API key no es válida. Revisa ANTHROPIC_API_KEY en tu archivo .env.")
    except anthropic.PermissionDeniedError:
        raise ErrorRAG("Tu cuenta no tiene permiso para este modelo. Cambia MODELO_CLAUDE en .env.")
    except anthropic.NotFoundError:
        raise ErrorRAG(f"El modelo '{config.MODELO_CLAUDE}' no existe. Revisa MODELO_CLAUDE en .env.")
    except anthropic.RateLimitError:
        raise ErrorRAG("Límite de uso alcanzado (429) aun después de 3 reintentos. Espera un minuto.")
    except anthropic.APIConnectionError:
        raise ErrorRAG("No hay conexión con la API de Claude. Revisa tu internet.")
    except anthropic.APIStatusError as e:
        raise ErrorRAG(f"La API respondió {e.status_code}: {e.message}")


def responder(pregunta: str, k: int = config.TOP_K, vectorstore=None) -> dict:
    inicio = time.perf_counter()
    fragmentos = recuperar(pregunta, k, vectorstore)
    respuesta = llamar_claude(pregunta, armar_contexto(fragmentos))

    if respuesta.stop_reason == "refusal":
        texto = "No puedo responder eso. Para cualquier duda escríbenos al 961 277 8843."
    else:
        texto = "".join(b.text for b in respuesta.content if b.type == "text").strip()

    resultado = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "pregunta": pregunta,
        "respuesta": texto,
        "fuentes": [{"seccion": d.metadata.get("seccion"), "puntaje": round(s, 4)} for d, s in fragmentos],
        "modelo": respuesta.model,
        "tokens_entrada": respuesta.usage.input_tokens,
        "tokens_salida": respuesta.usage.output_tokens,
        "costo_usd": costo_usd(config.MODELO_CLAUDE, respuesta.usage),
        "segundos": round(time.perf_counter() - inicio, 2),
        "stop_reason": respuesta.stop_reason,
    }
    registrar(resultado)
    return resultado


def registrar(resultado: dict):
    config.CARPETA_LOGS.mkdir(exist_ok=True)
    with open(config.CARPETA_LOGS / "consultas.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(resultado, ensure_ascii=False) + "\n")
