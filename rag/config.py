"""Configuración central. Todo lo que cambia entre equipos vive en .env, no en el código."""
import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

# Rutas
CARPETA_DATOS = RAIZ / "data"
CARPETA_VECTORES = RAIZ / "vectorstore"
CARPETA_LOGS = RAIZ / "logs"
COLECCION = "menu_delhorno"

# Embeddings locales (corren en tu PC, sin costo por llamada)
MODELO_EMBEDDINGS = os.getenv(
    "MODELO_EMBEDDINGS", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Troceo (chunking)
TAMANO_CHUNK = int(os.getenv("TAMANO_CHUNK", "800"))
SOLAPE_CHUNK = int(os.getenv("SOLAPE_CHUNK", "80"))

# Recuperación
TOP_K = int(os.getenv("TOP_K", "4"))

# Claude
MODELO_CLAUDE = os.getenv("MODELO_CLAUDE", "claude-opus-5-5")
ESFUERZO = os.getenv("ESFUERZO", "low")  # low | medium | high
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# Precio por millón de tokens en USD (entrada, salida). Fuente: tabla pública de Anthropic, oct 2026.
PRECIOS_USD_POR_MTOK = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Modelos que aceptan el respaldo automático del servidor cuando un filtro de seguridad rechaza la petición
MODELOS_CON_RESPALDO = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"}
