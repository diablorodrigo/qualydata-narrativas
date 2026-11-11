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
"""
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
    return con.execute("""select s.a, s.b, s.n, s.sw / sqrt(n1.np * n2.np) sinc
        from (select a, b, count(*) n, sum(w) sw from _pares group by 1, 2 having count(*) >= 2) s
        join _act n1 on n1.a = s.a join _act n2 on n2.a = s.b""").df()


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
    base = con.execute("select publicacion_id, cuenta_id, ts from comentarios order by publicacion_id").df()
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
        where c1.cuenta_id <> c2.cuenta_id group by all""").df()
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
