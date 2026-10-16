"""Búsqueda híbrida: vectores (significado) + BM25 (palabras exactas), unidas con Reciprocal Rank Fusion.

Por qué: el modelo de embeddings entiende "¿a cómo?" = "precio", pero confunde productos
con nombres raros ("chamorro", "lechón"). BM25 premia la palabra exacta. RRF combina los
dos rankings sin tener que calibrar sus puntajes, que vienen en escalas distintas.
"""
import math
import re
import unicodedata
from collections import Counter

PALABRAS_VACIAS = set(
    "a al algo algun alguna como con cual cuanto cuanta cuantos cuantas cuesta cuestan de del el ella en es esta "
    "este hay la las lo los me mi mas no o para pero por que quiero se si sin su sus te tienen tiene un una uno unos "
    "y ya yo hola favor sale precio precios vale valen manejan venden".split()
)


def tokenizar(texto: str) -> list[str]:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    tokens = []
    for w in re.findall(r"[a-z0-9]+", t):
        if w in PALABRAS_VACIAS:
            continue
        # Stemming mínimo para plural español: tacos→taco, lechones→lechon, aguas→agua
        if len(w) > 4 and w.endswith("es") and not w.endswith("ues"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        tokens.append(w)
    return tokens


class BM25:
    def __init__(self, textos: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenizar(t) for t in textos]
        self.k1, self.b = k1, b
        self.promedio = sum(map(len, self.docs)) / max(len(self.docs), 1)
        n = len(self.docs)
        df = Counter(tok for d in self.docs for tok in set(d))
        self.idf = {tok: math.log(1 + (n - f + 0.5) / (f + 0.5)) for tok, f in df.items()}

    def puntajes(self, consulta: str) -> list[float]:
        q = tokenizar(consulta)
        salida = []
        for d in self.docs:
            tf = Counter(d)
            s = 0.0
            for tok in q:
                if tok in tf:
                    f = tf[tok]
                    s += self.idf[tok] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.promedio))
            salida.append(s)
        return salida


def fusionar(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: cada lista aporta 1/(k + posición)."""
    total = Counter()
    for ranking in rankings:
        for pos, doc_id in enumerate(ranking, start=1):
            total[doc_id] += 1 / (k + pos)
    return total.most_common()
