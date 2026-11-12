"""Ingesta robusta de los CSV a DuckDB (persistido como Parquet).

Tolera: archivos .csv o .csv.gz, encoding UTF-8 o Latin-1, filas rotas,
fechas inválidas o vacías, ids duplicados, textos nulos y respuestas a
comentarios inexistentes.
"""
from __future__ import annotations

from pathlib import Path

import duckdb


def _buscar(datos: Path, nombre: str) -> Path | None:
    for cand in (datos / f"{nombre}.csv.gz", datos / f"{nombre}.csv", datos / f"{nombre}.gz"):
        if cand.exists():
            return cand
    return None


def _encoding(path: Path) -> str:
    """UTF-8 si todo el archivo decodifica bien; si no, Latin-1."""
    import codecs
    import gzip
    dec = codecs.getincrementaldecoder("utf-8")()
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as f:
            while chunk := f.read(1 << 20):
                dec.decode(chunk)
            dec.decode(b"", final=True)
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _leer(con, path: Path, tabla: str):
    """Lee todo como texto para no fallar con tipos sucios."""
    base = "all_varchar=true, header=true, ignore_errors=true, null_padding=true, strict_mode=false, quote='\"'"
    enc = _encoding(path)
    con.execute(f"create or replace table {tabla} as select * from read_csv('{path.as_posix()}', {base}, encoding='{enc}')")


def _col(con, tabla: str, nombre: str, default: str = "NULL") -> str:
    cols = {r[0].strip().lower(): r[0] for r in con.execute(f"describe {tabla}").fetchall()}
    return f'"{cols[nombre]}"' if nombre in cols else default


def _ts(expr: str) -> str:
    # Acepta ISO con zona, sin zona (se asume -03:00) y formatos dd/mm/aaaa comunes.
    e = f"trim({expr})"
    return (f"coalesce(try_cast({e} as timestamptz), try_cast(({e} || '-03:00') as timestamptz), "
            f"try_cast(try_strptime({e}, ['%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y']) as timestamptz))")


def cargar(datos: Path, trabajo: Path, log=print) -> duckdb.DuckDBPyConnection:
    trabajo.mkdir(parents=True, exist_ok=True)
    db = trabajo / "qualydata.duckdb"
    if db.exists():
        db.unlink()
    con = duckdb.connect(str(db))
    con.execute("set preserve_insertion_order=false")
    con.execute(f"set temp_directory='{(trabajo / 'tmp').as_posix()}'")
    con.execute("set TimeZone='America/Argentina/Buenos_Aires'")

    pc = _buscar(datos, "comentarios")
    if pc is None:
        raise SystemExit(f"No encuentro comentarios.csv(.gz) en {datos}")
    _leer(con, pc, "raw_comentarios")
    c = lambda n, d="NULL": _col(con, "raw_comentarios", n, d)
    con.execute(f"""
        create or replace table comentarios as
        with base as (
            select trim({c('comentario_id')}) comentario_id,
                   coalesce(nullif(trim({c('publicacion_id')}), ''), 'sin_publicacion') publicacion_id,
                   coalesce(nullif(trim({c('cuenta_id')}), ''), 'sin_cuenta') cuenta_id,
                   {_ts(c('fecha'))} fecha,
                   coalesce({c('texto')}, '') texto,
                   nullif(trim({c('respuesta_a')}), '') respuesta_a,
                   coalesce(try_cast({c('me_gusta', "'0'")} as integer), 0) me_gusta,
                   row_number() over () rn
            from raw_comentarios
            where nullif(trim({c('comentario_id')}), '') is not null
        ), dedup as (
            select * exclude (k) from (select *, row_number() over (partition by comentario_id order by rn) k from base) where k = 1
        )
        select * from dedup
    """)
    # Fechas faltantes: se imputan con la mediana de su publicación (o la global).
    con.execute("""
        update comentarios set fecha = coalesce(
            (select to_timestamp(median(epoch(c2.fecha))) from comentarios c2
             where c2.publicacion_id = comentarios.publicacion_id and c2.fecha is not null),
            (select to_timestamp(median(epoch(fecha))) from comentarios where fecha is not null))
        where fecha is null
    """)
    con.execute("update comentarios set respuesta_a = null where respuesta_a is not null and respuesta_a not in (select comentario_id from comentarios)")
    con.execute("alter table comentarios add column ts bigint")
    con.execute("update comentarios set ts = epoch(fecha)::bigint")

    pa = _buscar(datos, "cuentas")
    if pa is not None:
        _leer(con, pa, "raw_cuentas")
        a = lambda n, d="NULL": _col(con, "raw_cuentas", n, d)
        con.execute(f"""create or replace table cuentas_raw as
            select trim({a('cuenta_id')}) cuenta_id, {a('usuario')} usuario,
                   try_cast(try_cast(trim({a('fecha_creacion')}) as timestamp) as date) fecha_creacion,
                   try_cast({a('seguidores')} as bigint) seguidores, try_cast({a('siguiendo')} as bigint) siguiendo
            from raw_cuentas where nullif(trim({a('cuenta_id')}), '') is not null""")
    else:
        con.execute("create or replace table cuentas_raw (cuenta_id varchar, usuario varchar, fecha_creacion date, seguidores bigint, siguiendo bigint)")
    con.execute("""create or replace table cuentas as
        select u.cuenta_id, any_value(r.usuario) usuario, any_value(r.fecha_creacion) fecha_creacion,
               any_value(r.seguidores) seguidores, any_value(r.siguiendo) siguiendo
        from (select cuenta_id from cuentas_raw union select distinct cuenta_id from comentarios) u
        left join cuentas_raw r using (cuenta_id) group by u.cuenta_id""")

    pp = _buscar(datos, "publicaciones")
    if pp is not None:
        _leer(con, pp, "raw_pub")
        p = lambda n, d="NULL": _col(con, "raw_pub", n, d)
        con.execute(f"""create or replace table publicaciones as select trim({p('publicacion_id')}) publicacion_id,
            trim({p('fuente_id')}) fuente_id, {_ts(p('fecha'))} fecha, coalesce({p('titular')}, '') titular from raw_pub""")
    else:
        con.execute("create or replace table publicaciones (publicacion_id varchar, fuente_id varchar, fecha timestamptz, titular varchar)")

    for t in ("comentarios", "cuentas", "publicaciones"):
        con.execute(f"copy {t} to '{(trabajo / (t + '.parquet')).as_posix()}' (format parquet, compression zstd)")
    for t in [r[0] for r in con.execute("show tables").fetchall() if r[0].startswith("raw_") or r[0] == "cuentas_raw"]:
        con.execute(f"drop table {t}")
    n = con.execute("select count(*), count(distinct cuenta_id), count(distinct publicacion_id) from comentarios").fetchone()
    log(f"  {n[0]:,} comentarios, {n[1]:,} cuentas, {n[2]:,} publicaciones cargados en DuckDB")
    return con
