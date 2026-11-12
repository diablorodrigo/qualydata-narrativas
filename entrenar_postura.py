"""Entrena el clasificador auxiliar de postura con la verdad del set de práctica.

Uso: python entrenar_postura.py --datos datos/practica
Genera modelos/postura.joblib y un respaldo portable en modelos/postura.json
(independiente de la versión de scikit-learn). Es solo un voto para orientar
cada narrativa; la postura comentario por comentario sale del grafo de conversación.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from qn import texto as tx


def _ngrams(t: str) -> set[str]:
    w = [x for x in t.split() if len(x) >= 2]
    return set(w) | {f"{a} {b}" for a, b in zip(w, w[1:])}


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

    pos_c, neg_c = Counter(), Counter()
    n_pos = n_neg = 0
    for core_txt, post_lbl in zip(g["core"], g["postura"]):
        grams = _ngrams(core_txt)
        if post_lbl == "rechaza":
            n_pos += 1
            for gr in grams:
                pos_c[gr] += 1
        else:
            n_neg += 1
            for gr in grams:
                neg_c[gr] += 1
    weights = {}
    for gr in set(pos_c) | set(neg_c):
        if pos_c[gr] + neg_c[gr] < 3:
            continue
        w = math.log(((pos_c[gr] + 1.0) / (n_pos + 2.0)) / ((neg_c[gr] + 1.0) / (n_neg + 2.0)))
        if abs(w) >= 0.25:
            weights[gr] = round(w, 4)
    json_path = a.salida.with_suffix(".json")
    json_path.write_text(json.dumps({"weights": weights}, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo guardado en {a.salida} y respaldo portable en {json_path} ({len(g)} núcleos)")


if __name__ == "__main__":
    main()
