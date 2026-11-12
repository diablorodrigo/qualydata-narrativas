"""Tablero estático (un solo HTML, sin dependencias) para la demo y el pitch.

Responde en un minuto: qué narrativas circulan, desde cuándo y quién las empuja.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

COLORES = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d", "#4d7c0f",
           "#7c3aed", "#b45309", "#0f766e", "#e11d48", "#1d4ed8", "#a16207", "#15803d", "#6d28d9"]


def _spark(serie: list[int], ancho=360, alto=46, color="#2563eb", marca: int | None = None) -> str:
    if not serie:
        return ""
    mx = max(serie) or 1
    n = len(serie)
    pts = " ".join(f"{i * ancho / max(1, n - 1):.1f},{alto - 2 - v * (alto - 6) / mx:.1f}" for i, v in enumerate(serie))
    area = f"0,{alto} {pts} {ancho},{alto}"
    m = ""
    if marca is not None:
        x = marca * ancho / max(1, n - 1)
        m = f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{alto}" stroke="#111" stroke-dasharray="3,2"/>'
    return (f'<svg width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}"><polygon points="{area}" fill="{color}22"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/>{m}</svg>')


def _barra(ap: int, re: int) -> str:
    tot = max(1, ap + re)
    a = 100 * ap / tot
    return (f'<div class="bar"><span style="width:{a:.1f}%;background:#16a34a"></span>'
            f'<span style="width:{100 - a:.1f}%;background:#dc2626"></span></div>'
            f'<small>{a:.0f}% apoya · {100 - a:.0f}% rechaza</small>')


def _horas(h: list[int]) -> str:
    mx = max(h) or 1
    barras = "".join(f'<rect x="{i * 7}" y="{30 - 28 * v / mx:.1f}" width="6" height="{28 * v / mx:.1f}" fill="#9333ea"/>'
                     for i, v in enumerate(h))
    return f'<svg width="168" height="32">{barras}</svg>'


def generar(con, destino: Path, info_c: dict, info_n: dict):
    e = lambda q: con.execute(q).df()
    rango = e("select min(ts) t0, max(ts) t1, count(*) n from comentarios").iloc[0]
    t0 = int(rango.t0)
    dias = int((rango.t1 - t0) // 86400) + 1
    horas_tot = int((rango.t1 - t0) // 3600) + 1
    nar = e("select * from narrativas")
    vol = e(f"""select e.narrativa, cast((c.ts - {t0}) / 21600 as int) b, count(*) n
               from comentarios c join etiquetas e using (comentario_id) where e.narrativa <> 'ninguna' group by all""")
    coord = e("""select e.narrativa, g.grupo, e.postura, count(*) n from comentarios c join etiquetas e using (comentario_id)
                 join grupos g on g.cuenta_id = c.cuenta_id where g.grupo <> '' group by all""")
    nb = horas_tot // 6 + 1
    filas = []
    for i, r in nar.iterrows():
        serie = [0] * nb
        for b, n in vol[vol.narrativa == r.narrativa][["b", "n"]].itertuples(index=False):
            if 0 <= b < nb:
                serie[b] = int(n)
        t_or = pd.Timestamp(r.fecha_origen)
        b_or = int((t_or.timestamp() - t0) // 21600)
        emergente = (t_or.timestamp() - t0) > 86400
        cc = coord[coord.narrativa == r.narrativa]
        empuje = ""
        if len(cc):
            g = cc.groupby("grupo").n.sum().sort_values(ascending=False)
            tot_g = coord.groupby("grupo").n.sum()
            empuje = " ".join(f'<span class="tag">{html.escape(k)} · {v} com.</span>' for k, v in g.items()
                              if v >= 10 and v >= 0.15 * tot_g[k])
        ej_ap = e(f"select c.texto from comentarios c join etiquetas e using (comentario_id) where e.narrativa = '{r.narrativa}' and e.postura = 'apoya' and c.respuesta_a is null group by 1 order by count(*) desc limit 2").texto.tolist()
        ej_re = e(f"select c.texto from comentarios c join etiquetas e using (comentario_id) where e.narrativa = '{r.narrativa}' and e.postura = 'rechaza' group by 1 order by count(*) desc limit 2").texto.tolist()
        det_html = ""
        if ej_ap or ej_re:
            ap_str = "<br>".join(f'🟢 «{html.escape(x[:110])}»' for x in ej_ap)
            re_str = "<br>".join(f'🔴 «{html.escape(x[:110])}»' for x in ej_re)
            det_html = f'<details style="margin-top:4px"><summary style="cursor:pointer;font-size:12px;color:#2563eb">Ejemplos apoya vs. rechaza</summary><small style="display:block;margin-top:4px;color:#334155">{ap_str}<br>{re_str}</small></details>'
        filas.append(f"""<tr><td><b>{r.narrativa}</b></td><td><div class="tit">{html.escape(str(r.titulo))}</div>
            <small class="kw">{html.escape(str(r.frases_clave)).replace('|', ' · ')}</small>{det_html}</td>
            <td>{t_or.strftime('%d/%m %H:%M')}{' <span class="new">emergente</span>' if emergente else ''}</td>
            <td>{_spark(serie, color=COLORES[i % len(COLORES)], marca=b_or)}</td>
            <td style="text-align:right">{int(r.comentarios):,}</td><td>{_barra(int(r.apoya), int(r.rechaza))}</td><td>{empuje}</td></tr>""")

    grp = e("""select g.grupo, count(distinct g.cuenta_id) cuentas,
               median(date_diff('day', cu.fecha_creacion, cast(to_timestamp((select min(ts) from comentarios)) as date))) antig_dias,
               median(cu.seguidores) seguidores
               from grupos g join cuentas cu using (cuenta_id) where g.grupo <> '' group by 1 order by 1""")
    gfilas = []
    for _, g in grp.iterrows():
        cm = e(f"""select c.texto, c.ts, c.publicacion_id, e.narrativa, e.postura from comentarios c join grupos g using (cuenta_id)
                   join etiquetas e using (comentario_id) where g.grupo = '{g.grupo}'""")
        h = [0] * 24
        for hh in pd.to_datetime(cm.ts, unit="s", utc=True).dt.tz_convert("-03:00").dt.hour:
            h[hh] += 1
        nn = cm[cm.narrativa != "ninguna"]
        principal = ""
        if len(nn):
            top = nn.groupby(["narrativa", "postura"]).size().sort_values(ascending=False)
            (na, po), k = top.index[0], top.iloc[0]
            tit = nar.set_index("narrativa").titulo.get(na, "")
            rasgos = []
            if not pd.isna(g.antig_dias):
                rasgos.append("Cuentas nuevas" if g.antig_dias <= 15 else ("Cuentas antiguas + cobertura" if k / len(nn) < 0.75 else "Cuentas antiguas"))
            if sum(h[0:6]) / max(1, len(cm)) >= 0.5:
                rasgos.append("Ráfaga nocturna")
            if cm.texto.value_counts().head(5).sum() / max(1, len(cm)) >= 0.35:
                rasgos.append("Plantilla repetida")
            else:
                rasgos.append("Textos variados / respuestas")
            badges = " ".join(f'<span class="tag">{r_txt}</span>' for r_txt in rasgos)
            principal = f"<b>{na}</b> ({po}, {100 * k / len(nn):.0f}%)<br><small>{html.escape(str(tit))}</small><br>{badges}"
        ej = cm.texto.value_counts().head(3).index.tolist()
        gfilas.append(f"""<tr><td><b>{g.grupo}</b></td><td style="text-align:right">{int(g.cuentas)}</td>
            <td style="text-align:right">{len(cm)}</td><td style="text-align:right">{cm.publicacion_id.nunique()}</td>
            <td style="text-align:right">{'' if pd.isna(g.antig_dias) else int(g.antig_dias)}</td>
            <td style="text-align:right">{'' if pd.isna(g.seguidores) else int(g.seguidores)}</td>
            <td>{principal}</td><td>{_horas(h)}</td>
            <td><small>{'<br>'.join('«' + html.escape(x[:110]) + '»' for x in ej)}</small></td></tr>""")

    ruido = int(e("select count(*) from etiquetas where narrativa = 'ninguna'").iloc[0, 0])
    costo = {}
    cp = destino.parent / "costo.json"
    if cp.exists():
        costo = json.loads(cp.read_text(encoding="utf-8"))
    kpis = [("Comentarios", f"{int(rango.n):,}"), ("Días", dias), ("Narrativas", len(nar)),
            ("Ruido", f"{100 * ruido / max(1, rango.n):.0f}%"), ("Operaciones coordinadas", len(grp)),
            ("Cuentas coordinadas", int(grp.cuentas.sum()) if len(grp) else 0),
            ("Tiempo de proceso", f"{costo.get('segundos', '?')} s"), ("Costo IA", f"USD {costo.get('usd_ia', 0)}")]
    kp = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{k}</div></div>' for k, v in kpis)
    doc = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>QualyData · Narrativas y coordinación</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#f8fafc;color:#0f172a;margin:0;padding:24px 32px}}
h1{{margin:0 0 4px;font-size:26px}} h2{{margin:32px 0 8px;font-size:20px}} .sub{{color:#475569;margin-bottom:18px}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap}} .kpi{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;min-width:120px}}
.kpi .v{{font-size:24px;font-weight:700}} .kpi .l{{color:#64748b;font-size:13px}}
table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden}}
th,td{{padding:8px 10px;border-bottom:1px solid #f1f5f9;vertical-align:middle;font-size:14px}} th{{background:#f1f5f9;text-align:left;font-size:13px;color:#334155}}
.tit{{font-weight:600}} .kw{{color:#64748b}} .bar{{display:flex;height:10px;width:160px;border-radius:5px;overflow:hidden}} .bar span{{display:block}}
.tag{{display:inline-block;background:#f3e8ff;color:#6b21a8;border-radius:6px;padding:2px 6px;margin:1px;font-size:12px}}
.new{{background:#fef3c7;color:#92400e;border-radius:6px;padding:1px 6px;font-size:12px}} .nota{{color:#64748b;font-size:13px;margin-top:8px}}
</style></head><body>
<h1>¿Qué circula, desde cuándo y quién lo empuja?</h1>
<div class="sub">Desafío QualyData · Hackathon Argentina Space 2026 · datos sintéticos</div>
<div class="kpis">{kp}</div>
<h2>Narrativas</h2>
<table><tr><th></th><th>Narrativa (frase más repetida del lado que la afirma)</th><th>Origen</th><th>Volumen cada 6 h (línea punteada: origen)</th><th>Comentarios</th><th>Postura</th><th>Empujada por</th></tr>
{''.join(filas)}</table>
<div class="nota">Una narrativa es emergente si su origen cae después del primer día del período.</div>
<h2>Operaciones coordinadas</h2>
<table><tr><th>Grupo</th><th>Cuentas</th><th>Comentarios</th><th>Publicaciones</th><th>Antigüedad mediana (días)</th><th>Seguidores (mediana)</th><th>Qué empujan</th><th>Horario (0 a 23 h)</th><th>Textos más repetidos</th></tr>
{''.join(gfilas)}</table>
<div class="nota">Un grupo se marca cuando sus cuentas coinciden en las mismas publicaciones con minutos de diferencia muchas más veces que lo esperable por azar
(umbral = percentil 99,95 de un mundo nulo barajado: {info_c.get('umbral_sincronia')}; grupo máximo que aparece por azar: {info_c.get('tamano_azar')} cuentas).</div>
</body></html>"""
    destino.write_text(doc, encoding="utf-8")
