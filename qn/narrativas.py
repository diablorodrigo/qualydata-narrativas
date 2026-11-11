"""Tarea A: descubrir narrativas, ruido, postura y fecha de origen.

Idea central: dos señales complementarias.
1. Significado: embeddings multilingües locales de cada "núcleo" de texto
   (texto sin muletillas, deduplicado), para juntar variantes y typos.
2. Conversación: una respuesta que discute a otra pertenece a la misma
   narrativa con la postura opuesta. Eso arma un grafo de núcleos donde las
   comunidades son narrativas y la bipartición de cada comunidad es la postura.
"""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import ia
from . import texto as tx

MODELO_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STOP = set(tx.sin_acentos("""a al algo algun alguna algunos ante asi aca ahi ahora aun cada como con contra cual cuando de del desde
donde dos el ella ellos en entre era es esa ese eso esta estan este esto estos fue ha hace hacen hasta hay la las le les lo los
mas me mi mientras mucho muy nada ni no nos o otra otro para pero poco por porque que q se sea ser si sin sobre solo son su sus
tambien tan tanto te tiene todo todos tu un una uno unos y ya va van vez ver x pa yo hoy dia bien mal eso esa este nunca siempre""").split())


def embeber(textos: list[str], cache: Path, hilos: int | None = None, log=print) -> np.ndarray:
    from fastembed import TextEmbedding
    modelo = TextEmbedding(MODELO_EMB, cache_dir=str(cache), threads=hilos)
    E = np.asarray(list(modelo.embed(textos, batch_size=256)), dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return E


def _louvain(G: nx.Graph, seed: int = 0, resolution: float = 1.0):
    return [set(c) for c in nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=resolution)]


def _dos_colores(G: nx.Graph, nodos: set) -> dict:
    """Bipartición de un grafo de 'contradicciones' (aristas = posturas opuestas)."""
    sub = G.subgraph(nodos)
    lado = {}
    for comp in nx.connected_components(sub):
        raiz = max(comp, key=lambda n: sub.degree(n, weight="weight"))
        lado[raiz] = 0
        for u, v in nx.bfs_edges(sub, raiz):
            lado[v] = 1 - lado[u]
    for _ in range(15):  # refinamiento: cada nodo toma el lado opuesto a la mayoría ponderada de vecinos
        cambios = 0
        for n in sub.nodes:
            s = [0.0, 0.0]
            for m, d in sub[n].items():
                s[lado[m]] += d.get("weight", 1.0)
            nuevo = 0 if s[1] > s[0] else (1 if s[0] > s[1] else lado[n])
            if nuevo != lado[n]:
                lado[n] = nuevo
                cambios += 1
        if not cambios:
            break
    return lado


def _origen_robusto(ts: np.ndarray, ventana_h: float = 3.0, frac: float = 0.03, minimo: int = 3) -> float:
    """Detección de arranque: primer momento en que el volumen en una ventana de
    `ventana_h` horas supera `frac` del pico de la narrativa. Después se toma el
    primer comentario del grupo dentro de las 3 horas previas a ese arranque.
    Así un comentario suelto mal asignado días antes no adelanta el origen."""
    ts = np.sort(ts)
    if len(ts) == 0:
        return float("nan")
    j = np.searchsorted(ts, ts + ventana_h * 3600, side="right")
    cuenta = j - np.arange(len(ts))
    umbral = max(minimo, frac * cuenta.max())
    ok = np.where(cuenta >= umbral)[0]
    if not len(ok):
        return float(ts[0])
    t_on = ts[ok[0]]
    previos = ts[(ts >= t_on - 3 * 3600) & (ts <= t_on)]
    return float(previos[0])


def _frases_clave(nucleos_por_grupo: dict, k: int = 6) -> dict:
    """c-TF-IDF simple: términos frecuentes en el grupo y raros en el resto."""
    tf = {}
    df = Counter()
    for g, items in nucleos_por_grupo.items():
        cnt = Counter()
        for t, w in items:
            toks = [x for x in re.findall(r"[a-zñ0-9]+", tx.sin_acentos(t)) if x not in STOP and len(x) > 2]
            grams = set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])}
            for x in grams:
                cnt[x] += w
        tf[g] = cnt
        for x in cnt:
            df[x] += 1
    G = len(nucleos_por_grupo)
    out = {}
    for g, cnt in tf.items():
        tot = sum(cnt.values()) or 1
        sc = {x: (c / tot) * math.log(1 + G / df[x]) * (1.3 if " " in x else 1.0) for x, c in cnt.items()}
        elegidos = []
        for x, _ in sorted(sc.items(), key=lambda z: -z[1]):
            if any(x in e or e in x for e in elegidos):
                continue
            elegidos.append(x)
            if len(elegidos) == k:
                break
        out[g] = elegidos
    return out


class ModeloPostura:
    """Clasificador liviano apoya/rechaza (TF-IDF + regresión logística), entrenado
    con la verdad del set de práctica. Solo se usa como un voto más para decidir
    qué lado de cada narrativa es el que la afirma."""

    def __init__(self, ruta: Path):
        self.ok = False
        try:
            import joblib
            self.vec, self.clf = joblib.load(ruta)
            self.ok = True
        except Exception:
            pass

    def p_rechaza(self, textos: list[str]) -> np.ndarray:
        if not self.ok or not textos:
            return np.full(len(textos), 0.5)
        return self.clf.predict_proba(self.vec.transform([tx.sin_acentos(t) for t in textos]))[:, 1]


def detectar(df: pd.DataFrame, cache: Path, modelo_postura: Path, cuentas_coord: set | None = None,
             log=print, hilos: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    N = len(df)
    df = df.copy()
    df["b"] = df["texto"].map(tx.base)
    P, S = tx.aprender_muletillas(set(df["b"].unique()))
    nucleo = tx.nucleo_fn(P, S)
    mapa = {b: nucleo(b) for b in df["b"].unique()}
    df["core"] = df["b"].map(mapa)
    log(f"  muletillas aprendidas: {len(P)} prefijos, {len(S)} sufijos; {df['core'].nunique():,} núcleos únicos")

    es_resp = df["respuesta_a"].notna()
    st = df.groupby("core").agg(n=("core", "size"), nresp=("respuesta_a", lambda x: x.notna().sum()))
    st["palabras"] = [len(k.split()) for k in st.index]
    st["rr"] = st["nresp"] / st["n"]
    acuerdo = set(st.index[(st.palabras <= 5) & (st.rr >= 0.3) & (st.nresp >= 3)])
    vacio = {""} | set(st.index[st.palabras == 0])
    log(f"  respuestas cortas de acompañamiento detectadas: {len(acuerdo)} núcleos")

    # --- Grafo de conversación entre núcleos (respuestas que discuten) ---
    por_id = df.set_index("comentario_id")
    r = df.loc[es_resp, ["core", "respuesta_a"]].copy()
    r["pcore"] = r["respuesta_a"].map(por_id["core"])
    r = r[~r["core"].isin(acuerdo | vacio) & ~r["pcore"].isin(acuerdo | vacio) & (r["core"] != r["pcore"])]
    W = Counter((min(a, b), max(a, b)) for a, b in zip(r["core"], r["pcore"]))
    G = nx.Graph()
    G.add_weighted_edges_from([(a, b, w) for (a, b), w in W.items()])

    # --- Embeddings de núcleos únicos ---
    candidatos = [k for k in st.index if k not in vacio and k not in acuerdo]
    E = embeber(candidatos, cache, hilos=hilos, log=log)
    idx = {k: i for i, k in enumerate(candidatos)}
    log(f"  embeddings: {len(candidatos):,} núcleos x {E.shape[1]} dims ({MODELO_EMB.split('/')[-1]})")

    # Hubs: núcleos que discuten con todos (no aportan a la estructura). Se sacan y se rehace.
    comms = _louvain(G)
    for _ in range(2):
        com_de = {n: i for i, c in enumerate(comms) for n in c}
        hubs = []
        for n in G.nodes:
            tot = sum(d["weight"] for d in G[n].values())
            dentro = sum(d["weight"] for m, d in G[n].items() if com_de[m] == com_de[n])
            if tot >= 10 and dentro / tot < 0.5:
                hubs.append(n)
        if not hubs:
            break
        G.remove_nodes_from(hubs)
        comms = _louvain(G)
    # Peso de cada comunidad en comentarios
    peso = [int(st.loc[list(c), "n"].sum()) for c in comms]
    n_contenido = int(df["core"].isin(candidatos).sum())
    minimo = max(60, int(0.004 * n_contenido))
    grandes = [c for c, w in zip(comms, peso) if w >= minimo]
    log(f"  grafo de conversación: {G.number_of_nodes():,} núcleos, {G.number_of_edges():,} aristas; "
        f"{len(comms)} comunidades, {len(grandes)} narrativas (mínimo {minimo} comentarios)")

    narr_de, lado_de = {}, {}
    for i, c in enumerate(sorted(grandes, key=lambda c: -st.loc[list(c), "n"].sum())):
        lados = _dos_colores(G, c)
        for n in c:
            narr_de[n] = i
            lado_de[n] = lados[n]
    K = len(grandes)

    # --- Núcleos fuera del grafo: vecino más cercano por significado o ruido ---
    asignados = [k for k in candidatos if k in narr_de]
    if asignados:
        A = E[[idx[k] for k in asignados]]
        # tasa de participación en conversación de los núcleos narrativos
        part = df[df["core"].isin(asignados)]
        enganche = set(r["core"]) | set(r["pcore"])
        rho = float(np.clip((st.loc[asignados, "nresp"].sum() * 2) / max(1, st.loc[asignados, "n"].sum()), 0.02, 0.9))
        resto = [k for k in candidatos if k not in narr_de]
        R = E[[idx[k] for k in resto]] if resto else np.zeros((0, E.shape[1]), np.float32)
        sims = []
        for s in range(0, len(resto), 2048):
            M = R[s:s + 2048] @ A.T
            j = M.argmax(1)
            sims.extend(zip(j, M[np.arange(len(j)), j]))
        UMBRAL = 0.80
        nuevo_ruido = 0
        for k, (j, sim) in zip(resto, sims):
            n_k = int(st.loc[k, "n"])
            # Ruido: núcleos frecuentes que nunca entran en una discusión (muy improbable para una narrativa).
            if k not in enganche and (1 - rho) ** n_k < 0.01:
                nuevo_ruido += 1
                continue
            if sim >= UMBRAL:
                vec = asignados[j]
                narr_de[k] = narr_de[vec]
                lado_de[k] = lado_de[vec]
        log(f"  núcleos asignados por similitud: {sum(1 for k in resto if k in narr_de):,}; ruido por falta de conversación: {nuevo_ruido:,}")

    # --- Orientación: qué lado afirma la narrativa (apoya) ---
    mp = ModeloPostura(modelo_postura)
    df["narr"] = df["core"].map(narr_de)
    df["lado"] = df["core"].map(lado_de)
    top = df[~es_resp & df["narr"].notna()]
    orient = {}
    coord = cuentas_coord or set()
    usar_ia = ia.disponible()
    tokens_ia, titulos_ia = 0, {}
    for g in range(K):
        sub = top[top["narr"] == g]
        votos = 0.0
        feats = {}
        s0, s1 = sub[sub["lado"] == 0], sub[sub["lado"] == 1]
        if len(s0) == 0 or len(s1) == 0:
            orient[g] = 0
            continue
        allg = df[df["narr"] == g]
        rr0 = allg[allg["lado"] == 0]["respuesta_a"].notna().mean()
        rr1 = allg[allg["lado"] == 1]["respuesta_a"].notna().mean()
        c0 = s0["core"].value_counts()
        c1 = s1["core"].value_counts()
        p0 = float(np.average(mp.p_rechaza(list(c0.index)), weights=c0.values))
        p1 = float(np.average(mp.p_rechaza(list(c1.index)), weights=c1.values))
        feats = {
            "tamano": np.sign(len(s0) - len(s1)),
            "largo": np.sign(s0["core"].str.split().str.len().mean() - s1["core"].str.split().str.len().mean()),
            "me_gusta": np.sign(s0["me_gusta"].mean() - s1["me_gusta"].mean()),
            "respuestas": np.sign(rr1 - rr0),
            "clasificador": np.sign(p1 - p0) * 1.01,
        }
        if coord:
            k0 = s0["cuenta_id"].isin(coord).sum()
            k1 = s1["cuenta_id"].isin(coord).sum()
            if k0 + k1 >= 10:
                feats["campana"] = np.sign(k0 - k1) * 1.5
        votos = sum(feats.values())
        orient[g] = 0 if votos >= 0 else 1  # lado que apoya
        if usar_ia:
            lado, titulo, tok = ia.orientar(list(c0.index[:6]), list(c1.index[:6]))
            tokens_ia += tok
            if lado is not None:
                orient[g] = lado
            if titulo:
                titulos_ia[g] = titulo
    df["postura"] = np.where(df["narr"].notna(), np.where(df["lado"] == df["narr"].map(orient), "apoya", "rechaza"), "")

    # --- Reglas de respuestas (en orden temporal) ---
    df = df.sort_values(["ts", "comentario_id"]).reset_index(drop=True)
    narr = df["narr"].to_numpy(dtype=object)
    post = df["postura"].to_numpy(dtype=object)
    pos = {c: i for i, c in enumerate(df["comentario_id"])}
    cores = df["core"].to_numpy(dtype=object)
    resp = df["respuesta_a"].to_numpy(dtype=object)
    for i in range(len(df)):
        pa = resp[i]
        if pa is None or (isinstance(pa, float) and math.isnan(pa)):
            continue
        j = pos.get(pa)
        if j is None or narr[j] is None or (isinstance(narr[j], float) and math.isnan(narr[j])):
            continue
        propio = narr[i] if not (isinstance(narr[i], float) and math.isnan(narr[i])) else None
        if cores[i] in acuerdo:
            narr[i], post[i] = narr[j], post[j]
        elif propio == narr[j] and post[i]:
            pass
        else:
            narr[i] = narr[j]
            post[i] = "rechaza" if post[j] == "apoya" else "apoya"
    df["narr"], df["postura"] = narr, post
    # Respuestas cortas o vacías sueltas (sin padre narrativo) quedan como ruido.
    df.loc[df["narr"].isna(), "postura"] = ""

    etiqueta = {g: f"n{g + 1:02d}" for g in range(K)}
    df["narrativa"] = df["narr"].map(lambda g: etiqueta.get(g, "ninguna") if g == g and g is not None else "ninguna")
    df.loc[df["narrativa"] == "ninguna", "postura"] = ""

    # --- Tabla de narrativas ---
    filas = []
    items = {}
    for g in range(K):
        sub = df[df["narr"] == g]
        ap = sub[(sub["postura"] == "apoya") & ~sub["core"].isin(acuerdo)]
        items[g] = list(ap["core"].value_counts().head(40).items()) or list(sub["core"].value_counts().head(40).items())
    claves = _frases_clave(items)
    for g in range(K):
        sub = df[df["narr"] == g]
        ap = sub[(sub["postura"] == "apoya") & ~sub["core"].isin(acuerdo)]["core"].value_counts()
        largos = [k for k in ap.index if len(k.split()) >= 5]
        rep = largos[0] if largos else (ap.index[0] if len(ap) else sub["core"].value_counts().index[0])
        titulo = titulos_ia.get(g) or (rep[:1].upper() + rep[1:])
        fiables = sub[sub["respuesta_a"].isna() & (sub["core"].str.split().str.len() >= 4)]
        t0 = _origen_robusto(fiables["ts"].to_numpy(dtype=float))
        if not np.isfinite(t0):
            t0 = float(sub["ts"].min())
        filas.append({
            "narrativa": etiqueta[g], "titulo": titulo, "frases_clave": "|".join(claves.get(g, [])),
            "fecha_origen": pd.Timestamp(t0, unit="s", tz="UTC").tz_convert("-03:00").isoformat(),
            "comentarios": len(sub), "apoya": int((sub["postura"] == "apoya").sum()),
            "rechaza": int((sub["postura"] == "rechaza").sum()),
        })
    narrativas = pd.DataFrame(filas, columns=["narrativa", "titulo", "frases_clave", "fecha_origen", "comentarios", "apoya", "rechaza"])
    info = {"muletillas_prefijo": P[:20], "muletillas_sufijo": S[:20], "nucleos": int(df["core"].nunique()),
            "narrativas": K, "tokens_ia": tokens_ia, "ia": usar_ia, "ruido": int((df["narrativa"] == "ninguna").sum())}
    return df, narrativas, info
