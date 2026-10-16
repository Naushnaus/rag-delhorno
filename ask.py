"""Consulta por línea de comandos.

    python ask.py "¿cuánto cuesta el chamorro?"
    python ask.py                      (modo conversación; salir con Enter vacío)
    python ask.py --ver-contexto "..."  (muestra qué fragmentos recuperó)
"""
import argparse
import sys

from rag import config
from rag.generador import ErrorRAG, responder
from rag.vectores import abrir_vectorstore


def imprimir(r: dict, ver_contexto: bool):
    print(f"\n{r['respuesta']}\n")
    if ver_contexto:
        for f in r["fuentes"]:
            print(f"  · {f['seccion']}  (puntaje RRF {f['puntaje']})")
    costo = f"${r['costo_usd']:.5f} USD" if r["costo_usd"] is not None else "costo n/d"
    print(f"  [{r['modelo']} · {r['tokens_entrada']}+{r['tokens_salida']} tokens · {costo} · {r['segundos']} s]\n")


def main():
    p = argparse.ArgumentParser(description="Pregúntale al menú de DEL HORNO")
    p.add_argument("pregunta", nargs="*", help="La pregunta. Si no pones nada, entra en modo conversación.")
    p.add_argument("--k", type=int, default=config.TOP_K, help="Cuántos fragmentos recuperar")
    p.add_argument("--ver-contexto", action="store_true", help="Muestra los fragmentos usados")
    args = p.parse_args()

    vs = abrir_vectorstore()
    if vs._collection.count() == 0:
        sys.exit("El vector store está vacío. Corre primero:  python ingest.py")

    preguntas = [" ".join(args.pregunta)] if args.pregunta else None
    try:
        if preguntas:
            imprimir(responder(preguntas[0], args.k, vs), args.ver_contexto)
            return
        print("DEL HORNO · asistente del menú (Enter vacío para salir)")
        while (q := input("> ").strip()):
            imprimir(responder(q, args.k, vs), args.ver_contexto)
    except ErrorRAG as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
