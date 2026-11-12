"""Tarea B: detectar cuentas coordinadas y agruparlas por operación.

1. Sincronía: para cada par de cuentas contamos en cuántas publicaciones distintas
   comentaron con menos de 10 minutos de diferencia. Cada coincidencia pesa menos
   cuanto más concurrida estaba esa publicación en ese momento (veinte comentarios
   en diez minutos es normal en un posteo viral y raro en uno que nadie mira).
   El total se normaliza por la actividad de cada cuenta.
2. Lo esperable: armamos un mundo nulo barajando las cuentas dentro de cada
   publicación (mismos horarios, misma actividad, cero coordinación) y fijamos los
   umbrales como percentiles altos de ese mundo nulo. Así el umbral se adapta solo
   a cada set.
3. Respuestas entre cuentas que además coinciden en horario refuerzan el vínculo
   (operaciones que se responden entre sí).
4. Comunidades (Louvain) sobre ese grafo = operaciones. Se valida cada grupo:
   tamaño mínimo y densidad interna.
5. Refinamiento por narrativa y ráfagas: una vez detectadas las narrativas y
   posturas, se consolidan las ventanas de ataque de cada operación sobre su
   (narrativa, postura) dominante para rescatar cuentas con actividad de
   cobertura y podar falsos positivos virales.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

VENTANA_S = 600


def _puntajes(con, tabla: str) -> pd.DataFrame:
    con.execute(f"""create or replace temp table _dens as
        select c1.publicacion_id p, c1.cuenta_id a, c1.ts, count(distinct c2.cuenta_id) m
        from {tabla} c1 join {tabla} c2 on c1.publicacion_id = c2.publicacion_id and abs(c1.ts - c2.ts) <= {VENTANA_S}
        group by all""")
    con.execute(f"""create or replace temp table _pares as
        select x1.a a, x2.a b, x1.p, max(1.0 / ln(1 + greatest(x1.m, x2.m))) w
        from _dens x1 join _dens x2 on x1.p = x2.p and x1.a < x2.a and abs(x1.ts - x2.ts) <= {VENTANA_S}
        group by 1, 2, 3""")
    con.execute(f"create or replace temp table _act as select cuenta_id a, count(distinct publicacion_id) np from {tabla} group by 1")
    # Orden fijo y redondeo: DuckDB agrega en paralelo y no garantiza el orden de las filas
    # ni el último bit de las sumas; así el grafo (y Louvain) queda idéntico entre corridas.
    return con.execute("""select s.a, s.b, s.n, round(s.sw / sqrt(n1.np * n2.np), 10) sinc
        from (select a, b, count(*) n, sum(w) sw from _pares group by 1, 2 having count(*) >= 2) s
        join _act n1 on n1.a = s.a join _act n2 on n2.a = s.b
        order by s.a, s.b""").df()


def _grafo(m: pd.DataFrame, q_alto: float, q_bajo: float) -> nx.Graph:
    fuerte = m[m["sinc"] >= q_alto]
    respaldo = m[(m["nr"] >= 1) & (m["sinc"] >= q_bajo)]
    aristas = pd.concat([fuerte, respaldo]).drop_duplicates(["a", "b"])
    G = nx.Graph()
    for a, b, s, nr in zip(aristas["a"], aristas["b"], aristas["sinc"], aristas["nr"]):
        G.add_edge(a, b, weight=float(s / q_alto) + 0.5 * float(min(nr, 3)))
    return G


def _comunidades(G: nx.Graph, semilla: int) -> list:
    """Comunidades Louvain con densidad interna mínima."""
    if not G.number_of_nodes():
        return []
    out = []
    for c in nx.community.louvain_communities(G, weight="weight", seed=semilla):
        if len(c) >= 3 and nx.density(G.subgraph(c)) >= 0.1:
            out.append(c)
    return out


def detectar(con, log=print, semilla: int = 0) -> tuple[pd.DataFrame, dict]:
    obs = _puntajes(con, "comentarios")
    # Mundo nulo: barajar cuentas dentro de cada publicación
    base = con.execute("select publicacion_id, cuenta_id, ts from comentarios order by publicacion_id, comentario_id").df()
    rng = np.random.default_rng(semilla)
    orden = np.lexsort((rng.random(len(base)), base["publicacion_id"].to_numpy()))
    nulo = base[["publicacion_id", "ts"]].copy()
    nulo["cuenta_id"] = base["cuenta_id"].to_numpy()[orden]
    con.register("_nulo_df", nulo)
    con.execute("create or replace temp table _nulo as select * from _nulo_df")
    nul = _puntajes(con, "_nulo")
    con.execute("drop table _nulo")
    q_alto = float(nul["sinc"].quantile(0.9995)) if len(nul) else 0.1
    q_bajo = float(nul["sinc"].quantile(0.99)) if len(nul) else 0.05

    resp = con.execute("""select least(c1.cuenta_id, c2.cuenta_id) a, greatest(c1.cuenta_id, c2.cuenta_id) b, count(*) nr
        from comentarios c1 join comentarios c2 on c1.respuesta_a = c2.comentario_id
        where c1.cuenta_id <> c2.cuenta_id group by all order by a, b""").df()
    m = obs.merge(resp, on=["a", "b"], how="left").fillna({"nr": 0})
    G = _grafo(m, q_alto, q_bajo)
    # Tamaño de grupo que aparece por azar: mismo procedimiento sobre el mundo nulo.
    Gn = _grafo(nul.merge(resp, on=["a", "b"], how="left").fillna({"nr": 0}), q_alto, q_bajo)
    tam_azar = max([len(c) for c in _comunidades(Gn, semilla)] or [0])
    tam_min = max(5, tam_azar + 1)
    grupos = {}
    detalle = []
    if G.number_of_nodes():
        k = 0
        for c in sorted(_comunidades(G, semilla), key=len, reverse=True):
            if len(c) < tam_min:
                continue
            dens = nx.density(G.subgraph(c))
            k += 1
            et = f"g{k:02d}"
            for a in c:
                grupos[a] = et
            detalle.append({"grupo": et, "cuentas": len(c), "densidad": round(dens, 3)})
    cuentas = con.execute("select cuenta_id from cuentas").df()
    cuentas["grupo"] = cuentas["cuenta_id"].map(grupos).fillna("")
    info = {"umbral_sincronia": round(q_alto, 4), "umbral_respaldo": round(q_bajo, 4),
            "pares_observados": int(len(obs)), "tamano_azar": tam_azar, "aristas": int(G.number_of_edges()), "grupos": detalle}
    log(f"  pares con co-ocurrencia: {len(obs):,}; umbral (p99.95 del mundo nulo) {q_alto:.4f}; "
        f"grupo máximo por azar {tam_azar}; {len(detalle)} operaciones, {len(grupos)} cuentas marcadas")
    return cuentas, info


def _rafagas(tss: np.ndarray) -> list[tuple[int, int]]:
    """Agrupa timestamps ordenados de una publicación en ráfagas separadas por > 1800 s."""
    if len(tss) < 2:
        return []
    out = []
    ini = prev = int(tss[0])
    cnt = 1
    for t in tss[1:]:
        ti = int(t)
        if ti - prev <= 1800:
            prev = ti
            cnt += 1
        else:
            if cnt >= 2:
                out.append((ini - 300, prev + 300))
            ini = prev = ti
            cnt = 1
    if cnt >= 2:
        out.append((ini - 300, prev + 300))
    return out


def refinar_con_narrativas(cuentas: pd.DataFrame, com: pd.DataFrame, info_c: dict, log=print) -> tuple[pd.DataFrame, dict]:
    """Consolida cada operación coordinada usando las ráfagas de su (narrativa, postura)
    dominante. Rescata cuentas con actividad de cobertura (que comentan dentro de las
    ráfagas de la operación pero tienen muchos comentarios de cobertura en otros temas)
    y descarta grupos falsos positivos que no empujan ninguna narrativa."""
    if not info_c.get("grupos"):
        return cuentas, info_c

    por_id_acc = dict(zip(com["comentario_id"], com["cuenta_id"]))
    por_id_nar = dict(zip(com["comentario_id"], com["narrativa"]))
    acc_total_coms = Counter(com["cuenta_id"])

    # Pre-indexar respuestas entre cuentas por narrativa
    resp_rows = com.loc[com["respuesta_a"].notna(), ["cuenta_id", "respuesta_a", "narrativa"]]
    resp_pares = []
    for a, pa, nar_a in resp_rows.itertuples(index=False):
        b = por_id_acc.get(pa)
        if b and a != b:
            resp_pares.append((a, b, nar_a, por_id_nar.get(pa, "ninguna")))

    validos = com[com["narrativa"] != "ninguna"][["publicacion_id", "cuenta_id", "ts", "narrativa", "postura"]]

    op_meta = []
    for g_info in info_c["grupos"]:
        et = g_info["grupo"]
        seed_set = set(cuentas.loc[cuentas["grupo"] == et, "cuenta_id"])
        sub_seed = validos[validos["cuenta_id"].isin(seed_set)]
        if len(sub_seed) < 15:
            continue
        top_np = sub_seed.groupby(["narrativa", "postura"]).size().sort_values(ascending=False, kind="stable")
        (nar, pos), cnt_top = top_np.index[0], int(top_np.iloc[0])
        if cnt_top < 15 or cnt_top / len(sub_seed) < 0.25:
            continue

        seed_np = sub_seed[(sub_seed["narrativa"] == nar) & (sub_seed["postura"] == pos)]
        windows = {}
        for pid, grp in seed_np.groupby("publicacion_id"):
            bs = _rafagas(np.sort(grp["ts"].to_numpy(dtype=np.int64)))
            if bs:
                windows[pid] = bs
        if sum(len(bs) for bs in windows.values()) < 3:
            continue
        op_meta.append({"et": et, "nar": nar, "pos": pos, "windows": windows, "accounts": seed_set, "densidad": g_info["densidad"]})

    # Si Louvain partió una misma campaña en dos subgrupos con igual (nar, pos) y ventanas superpuestas, unirlos
    merged = []
    used = set()
    for i, m1 in enumerate(op_meta):
        if i in used:
            continue
        acc_u = set(m1["accounts"])
        win_u = dict(m1["windows"])
        for j in range(i + 1, len(op_meta)):
            if j in used:
                continue
            m2 = op_meta[j]
            if m1["nar"] == m2["nar"] and m1["pos"] == m2["pos"]:
                inter = set(win_u) & set(m2["windows"])
                if len(inter) / max(1, min(len(win_u), len(m2["windows"]))) >= 0.5:
                    acc_u |= m2["accounts"]
                    for pid, bs in m2["windows"].items():
                        win_u[pid] = win_u.get(pid, []) + bs
                    used.add(j)
        m1_copy = dict(m1)
        m1_copy["accounts"] = acc_u
        m1_copy["windows"] = win_u
        merged.append(m1_copy)

    # Expandir cuentas con cobertura (en 2 pasadas) y podar falsos positivos por operación
    nuevos_grupos = {}
    nuevo_detalle = []
    for idx_g, m in enumerate(merged, start=1):
        et = f"g{idx_g:02d}"
        cur_set = set(m["accounts"])
        nar, pos = m["nar"], m["pos"]
        sub_np = validos[(validos["narrativa"] == nar) & (validos["postura"] == pos)]
        tot_np = Counter(sub_np["cuenta_id"])

        for _ in range(2):
            seed_np = sub_np[sub_np["cuenta_id"].isin(cur_set)]
            windows = {}
            for pid, grp in seed_np.groupby("publicacion_id"):
                bs = _rafagas(np.sort(grp["ts"].to_numpy(dtype=np.int64)))
                if bs:
                    windows[pid] = bs
            if sum(len(bs) for bs in windows.values()) < 3:
                break

            hits = Counter()
            for pid, grp in sub_np.groupby("publicacion_id"):
                if pid not in windows:
                    continue
                bs = windows[pid]
                ts_arr = grp["ts"].to_numpy(dtype=np.int64)
                acc_arr = grp["cuenta_id"].to_numpy()
                for w0, w1 in bs:
                    for aid in set(acc_arr[(ts_arr >= w0) & (ts_arr <= w1)]):
                        hits[aid] += 1

            rep_g = Counter()
            for a, b, nar_a, nar_b in resp_pares:
                if b in cur_set and nar_a == nar:
                    rep_g[a] += 1
                if a in cur_set and nar_b == nar:
                    rep_g[b] += 1

            kept_seed = {a for a in cur_set if hits[a] >= 2 or rep_g[a] >= 2}
            rescued = {
                a for a, h in hits.items()
                if (h >= 3 and (h / max(1, tot_np[a])) >= 0.6 and (h >= 4 or rep_g[a] >= 1 or (h / max(1, tot_np[a])) >= 0.75))
                or (h >= 2 and h == tot_np[a] and acc_total_coms[a] <= 6)
            }
            cur_set = kept_seed | rescued

        final_set = cur_set - set(nuevos_grupos)
        for a in final_set:
            nuevos_grupos[a] = et
        nuevo_detalle.append({"grupo": et, "cuentas": len(final_set), "densidad": m["densidad"]})

    cuentas = cuentas.copy()
    cuentas["grupo"] = cuentas["cuenta_id"].map(nuevos_grupos).fillna("")
    info_c = dict(info_c)
    info_c["grupos"] = nuevo_detalle
    log(f"  refinamiento con narrativas: {len(nuevo_detalle)} operaciones, {len(nuevos_grupos)} cuentas marcadas")
    return cuentas, info_c
