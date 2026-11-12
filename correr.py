"""Pipeline completo QualyData con un solo comando.

    python correr.py --datos datos/final --salida entrega
    python correr.py --datos datos/practica --salida salida/practica
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import pandas as pd

from qn import coordinacion, ingesta, narrativas, tablero

RAIZ = Path(__file__).resolve().parent


def hardware() -> str:
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        ram_txt = f"{ram:.0f} GB RAM"
    except (ValueError, OSError, AttributeError):
        ram_txt = "RAM s/d"
    return f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPU, {ram_txt}, sin GPU"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("datos/final"))
    ap.add_argument("--salida", type=Path, default=Path("entrega"))
    ap.add_argument("--trabajo", type=Path, default=None, help="Carpeta para DuckDB/Parquet (por defecto trabajo/<nombre del set>)")
    ap.add_argument("--hilos", type=int, default=None)
    ap.add_argument("--sin-tablero", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    trabajo = a.trabajo or (RAIZ / "trabajo" / a.datos.resolve().name)
    a.salida.mkdir(parents=True, exist_ok=True)
    tiempos = {}

    print("1/4 Ingesta en DuckDB")
    con = ingesta.cargar(a.datos, trabajo)
    tiempos["ingesta"] = round(time.time() - t0, 1)

    print("2/4 Coordinación")
    t = time.time()
    cuentas, info_c = coordinacion.detectar(con)
    tiempos["coordinacion"] = round(time.time() - t, 1)
    coord = set(cuentas.loc[cuentas["grupo"] != "", "cuenta_id"])

    print("3/4 Narrativas")
    t = time.time()
    df = con.execute("select comentario_id, publicacion_id, cuenta_id, ts, texto, respuesta_a, me_gusta from comentarios").df()
    com, narr, info_n = narrativas.detectar(df, RAIZ / ".cache" / "fastembed", RAIZ / "modelos" / "postura.joblib",
                                            cuentas_coord=coord, hilos=a.hilos)
    tiempos["narrativas"] = round(time.time() - t, 1)

    t_ref = time.time()
    cuentas, info_c = coordinacion.refinar_con_narrativas(cuentas, com, info_c)
    tiempos["coordinacion"] = round(tiempos["coordinacion"] + (time.time() - t_ref), 1)
    coord = set(cuentas.loc[cuentas["grupo"] != "", "cuenta_id"])

    print("4/4 Escritura")
    com[["comentario_id", "narrativa", "postura"]].sort_values("comentario_id").to_csv(a.salida / "comentarios.csv", index=False)
    narr[["narrativa", "titulo", "frases_clave", "fecha_origen"]].to_csv(a.salida / "narrativas.csv", index=False)
    cuentas[["cuenta_id", "grupo"]].sort_values("cuenta_id").to_csv(a.salida / "cuentas.csv", index=False)
    # Persistimos el resultado también en DuckDB para consultas y el tablero.
    con.register("_com", com[["comentario_id", "narrativa", "postura", "core"]])
    con.execute("create or replace table etiquetas as select * from _com")
    con.register("_cue", cuentas)
    con.execute("create or replace table grupos as select * from _cue")
    con.register("_nar", narr)
    con.execute("create or replace table narrativas as select * from _nar")

    seg = round(time.time() - t0, 1)
    n = len(com)
    from qn import ia
    usd_ia = info_n["tokens_ia"] / 1e6 * ia.PRECIO_USD_POR_MILLON
    costo = {
        "comentarios_procesados": n,
        "segundos": seg,
        "tokens_ia": info_n["tokens_ia"],
        "usd_ia": round(usd_ia, 4),
        "usd_por_mil_comentarios": round(usd_ia / max(1, n) * 1000, 6),
        "hardware": hardware(),
        "almacenamiento": "DuckDB local sobre Parquet (zstd)",
        "notas": ("IA solo para orientar y titular cada narrativa (1 consulta por narrativa). " if info_n["ia"] else "Sin APIs pagas. ") + "Embeddings locales (MiniLM multilingüe, ONNX en CPU), grafo de conversación y "
                 "detección de coordinación contra un mundo nulo barajado. Tiempos por etapa: "
                 + ", ".join(f"{k} {v}s" for k, v in tiempos.items()),
    }
    (a.salida / "costo.json").write_text(json.dumps(costo, ensure_ascii=False, indent=2), encoding="utf-8")
    if not a.sin_tablero:
        tablero.generar(con, a.salida / "tablero.html", info_c, info_n)
    con.close()
    print(f"Listo en {seg}s: {info_n['narrativas']} narrativas, {(com['narrativa'] == 'ninguna').sum():,} comentarios de ruido, "
          f"{len(info_c['grupos'])} operaciones coordinadas ({len(coord)} cuentas). Salida en {a.salida}/")


if __name__ == "__main__":
    main()
