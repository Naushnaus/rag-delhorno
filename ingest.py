"""Paso 1 del RAG: ingesta.

Lee los .md de ./data, los corta por secciones (un producto = un chunk), calcula
embeddings locales y los guarda en Chroma. Correr cada vez que cambie el menú.

    python ingest.py
"""
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag import config
from rag.vectores import abrir_vectorstore


def cargar_documentos() -> list[Document]:
    docs = []
    for ruta in sorted(config.CARPETA_DATOS.glob("*.md")):
        docs.append(Document(page_content=ruta.read_text(encoding="utf-8"), metadata={"fuente": ruta.name}))
    if not docs:
        raise SystemExit(f"No hay archivos .md en {config.CARPETA_DATOS}")
    return docs


def trocear(docs: list[Document]) -> list[Document]:
    # 1) Corte semántico: cada encabezado "## " es una sección (un producto o un bloque de datos).
    por_seccion = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "negocio"), ("##", "seccion")],
        strip_headers=False,
    )
    # 2) Red de seguridad: si una sección es muy larga, se parte por tamaño con solape.
    por_tamano = RecursiveCharacterTextSplitter(
        chunk_size=config.TAMANO_CHUNK, chunk_overlap=config.SOLAPE_CHUNK
    )
    chunks = []
    for doc in docs:
        for sec in por_seccion.split_text(doc.page_content):
            sec.metadata["fuente"] = doc.metadata["fuente"]
            sec.metadata.setdefault("seccion", "General")
            # El nombre del negocio va dentro del texto para que cada chunk tenga contexto propio.
            if not sec.page_content.startswith("# "):
                sec.page_content = f"DEL HORNO · {sec.page_content}"
            chunks.extend(por_tamano.split_documents([sec]))
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def main():
    docs = cargar_documentos()
    chunks = trocear(docs)
    vs = abrir_vectorstore()
    vs.reset_collection()  # ingesta idempotente: se borra la colección y se vuelve a cargar
    vs.add_documents(chunks, ids=[f"chunk-{c.metadata['chunk_id']}" for c in chunks])
    print(f"Documentos: {len(docs)} · Chunks: {len(chunks)} · Guardado en {config.CARPETA_VECTORES}")
    for c in chunks:
        print(f"  [{c.metadata['chunk_id']:>2}] {c.metadata['seccion']} ({len(c.page_content)} caracteres)")


if __name__ == "__main__":
    main()
