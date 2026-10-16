"""Evaluación con las 20 preguntas de prueba (tests/preguntas_prueba.json).

    python evaluar.py --solo-recuperacion   (gratis: revisa que el buscador traiga la sección correcta)
    python evaluar.py                       (completo: llama a Claude y revisa la respuesta; cuesta centavos)

Cada corrida guarda el detalle en ./resultados/.
"""
import argparse
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

from rag import config
from rag.generador import ErrorRAG, recuperar, responder
from rag.vectores import abrir_vectorstore

RAIZ = Path(__file__).resolve().parent


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y sin espacios/guiones, para comparar '961-277-8843' con '961 277 8843'."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    for ch in " -,":
        t = t.replace(ch, "")
    return t


def revisar_respuesta(caso: dict, respuesta: str) -> list[str]:
    r = normalizar(respuesta)
    fallas = [f"falta '{x}'" for x in caso.get("debe_contener", []) if normalizar(x) not in r]
    fallas += [f"no debía decir '{x}'" for x in caso.get("no_debe_contener", []) if normalizar(x) in r]
    return fallas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solo-recuperacion", action="store_true")
    p.add_argument("--k", type=int, default=config.TOP_K)
    args = p.parse_args()

    casos = json.loads((RAIZ / "tests" / "preguntas_prueba.json").read_text(encoding="utf-8"))
    vs = abrir_vectorstore()
    if vs._collection.count() == 0:
        sys.exit("El vector store está vacío. Corre primero:  python ingest.py")

    detalle, ok_recup, ok_resp, costo = [], 0, 0, 0.0
    for caso in casos:
        fila = {"id": caso["id"], "pregunta": caso["pregunta"]}

        secciones = [d.metadata.get("seccion") for d, _ in recuperar(caso["pregunta"], args.k, vs)]
        esperada = caso.get("seccion_esperada")
        fila["recuperacion_ok"] = esperada is None or esperada in secciones
        fila["secciones_recuperadas"] = secciones
        ok_recup += fila["recuperacion_ok"]

        if not args.solo_recuperacion:
            try:
                r = responder(caso["pregunta"], args.k, vs)
            except ErrorRAG as e:
                sys.exit(f"Error: {e}")
            fallas = revisar_respuesta(caso, r["respuesta"])
            fila.update(respuesta=r["respuesta"], fallas=fallas, respuesta_ok=not fallas)
            ok_resp += not fallas
            costo += r["costo_usd"] or 0

        marca = "OK " if fila["recuperacion_ok"] and fila.get("respuesta_ok", True) else "XX "
        extra = "" if fila.get("respuesta_ok", True) else f"  ← {', '.join(fila['fallas'])}"
        print(f"{marca}{caso['id']:>2}. {caso['pregunta']}{extra}")
        detalle.append(fila)

    n = len(casos)
    resumen = {"fecha": datetime.now().isoformat(timespec="seconds"), "modelo": config.MODELO_CLAUDE,
               "top_k": args.k, "recuperacion": f"{ok_recup}/{n}"}
    print(f"\nRecuperación: {ok_recup}/{n} ({ok_recup / n:.0%}) con top-{args.k}")
    if not args.solo_recuperacion:
        resumen.update(respuestas=f"{ok_resp}/{n}", costo_usd=round(costo, 5))
        print(f"Respuestas correctas: {ok_resp}/{n} ({ok_resp / n:.0%}) · costo total ${costo:.4f} USD")

    salida = RAIZ / "resultados"
    salida.mkdir(exist_ok=True)
    archivo = salida / f"eval-{datetime.now():%Y%m%d-%H%M}.json"
    archivo.write_text(json.dumps({"resumen": resumen, "detalle": detalle}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detalle guardado en {archivo}")


if __name__ == "__main__":
    main()
