"""Embeddings locales con FastEmbed (ONNX), envueltos en la interfaz Embeddings de LangChain.

Se escribe el adaptador a mano (en vez de usar langchain-community, que está en retiro)
para que LangChain y Chroma lo usen como cualquier otro proveedor de embeddings.
"""
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding

from . import config


class EmbeddingsLocales(Embeddings):
    def __init__(self, modelo: str = config.MODELO_EMBEDDINGS):
        # La primera vez descarga el modelo (~220 MB) y lo deja en caché.
        self._modelo = TextEmbedding(model_name=modelo)

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._modelo.embed(textos)]

    def embed_query(self, texto: str) -> list[float]:
        return next(iter(self._modelo.query_embed(texto))).tolist()
