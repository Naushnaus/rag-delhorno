"""Acceso al vector store local (Chroma persistido en ./vectorstore)."""
from langchain_chroma import Chroma

from . import config
from .embeddings import EmbeddingsLocales


def abrir_vectorstore(embeddings=None) -> Chroma:
    return Chroma(
        collection_name=config.COLECCION,
        embedding_function=embeddings or EmbeddingsLocales(),
        persist_directory=str(config.CARPETA_VECTORES),
        collection_metadata={"hnsw:space": "cosine"},
    )
