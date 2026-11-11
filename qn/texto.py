"""Normalización de texto y aprendizaje automático de muletillas.

La gente agrega muletillas ("che,", "la verdad que", "qué bronca", emojis) a la
misma idea. En vez de una lista fija, las aprendemos del propio set: un prefijo o
sufijo es muletilla si al sacarlo queda un texto que también aparece solo,
para muchos textos distintos. Así funciona con cualquier tema.
"""
import re
import unicodedata
from collections import defaultdict

_KEEP = set(" ,.:;?¿!¡'")


def base(texto: str) -> str:
    t = unicodedata.normalize("NFC", str(texto or "")).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w[\w.]*", " ", t)
    t = re.sub(r"#\w+", " ", t)
    t = "".join(ch if (ch.isalnum() or ch in _KEEP) else " " for ch in t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.;:")
    return t


def sin_acentos(t: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", t) if unicodedata.category(ch) != "Mn")


def hashtags(texto: str) -> list[str]:
    return [h.lower() for h in re.findall(r"#(\w+)", str(texto or ""))]


def aprender_muletillas(textos: set[str], max_palabras: int = 5):
    pref, suf = defaultdict(set), defaultdict(set)
    for t in textos:
        w = t.split(" ")
        for k in range(1, min(max_palabras, len(w) - 1) + 1):
            resto = " ".join(w[k:]).strip(" ,.;:")
            if resto in textos:
                pref[" ".join(w[:k])].add(resto)
            resto = " ".join(w[:-k]).strip(" ,.;:")
            if resto in textos:
                suf[" ".join(w[-k:])].add(resto)
    umbral = max(15, int(0.0002 * len(textos)))
    P = sorted([k for k, v in pref.items() if len(v) >= umbral], key=len, reverse=True)
    S = sorted([k for k, v in suf.items() if len(v) >= umbral], key=len, reverse=True)
    return P, S


def nucleo_fn(P, S):
    def nucleo(t: str) -> str:
        for _ in range(4):
            old = t
            for p in P:
                if t.startswith(p + " "):
                    t = t[len(p) + 1:].strip(" ,.;:")
                    break
            for s in S:
                if t.endswith(" " + s):
                    t = t[: -len(s) - 1].strip(" ,.;:")
                    break
            if t == old:
                break
        return t
    return nucleo
