"""Entrena el clasificador auxiliar de postura con la verdad del set de práctica.

Uso: python entrenar_postura.py --datos datos/practica
Genera modelos/postura.joblib. Es solo un voto para orientar cada narrativa;
la postura comentario por comentario sale del grafo de conversación.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from qn import texto as tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"))
    ap.add_argument("--salida", type=Path, default=Path("modelos/postura.joblib"))
    a = ap.parse_args()
    c = pd.read_csv(a.datos / "comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(a.datos / "verdad" / "comentarios.csv.gz", dtype=str, keep_default_na=False)
    c = c.merge(v, on="comentario_id")
    c = c[(c["narrativa_id"] != "ninguna") & (c["respuesta_a"] == "")]
    c["b"] = c["texto"].map(tx.base)
    P, S = tx.aprender_muletillas(set(c["b"]))
    nuc = tx.nucleo_fn(P, S)
    c["core"] = c["b"].map(nuc).map(tx.sin_acentos)
    g = c.groupby("core")["postura"].agg(lambda x: x.mode()[0]).reset_index()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(g["core"])
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X, g["postura"] == "rechaza")
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((vec, clf), a.salida)
    print(f"Modelo guardado en {a.salida} ({len(g)} núcleos)")


if __name__ == "__main__":
    main()
