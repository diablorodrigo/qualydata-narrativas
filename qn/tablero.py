"""Tablero estático interactivo (un solo HTML, sin dependencias externas) para la demo y el pitch.

Responde en un minuto: qué narrativas circulan, desde cuándo y quién las empuja.
Incluye filtros en 1 clic (emergentes / impulsadas por campaña), buscador instantáneo,
ejemplos desplegables (apoya vs. rechaza), vínculo interactivo entre narrativas y
operaciones, y gráfico comparativo contra el mundo nulo barajado.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

COLORES = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d", "#4d7c0f",
           "#7c3aed", "#b45309", "#0f766e", "#e11d48", "#1d4ed8", "#a16207", "#15803d", "#6d28d9"]


def _spark(serie: list[int], t0: int, ancho=360, alto=48, color="#2563eb", marca: int | None = None) -> str:
    if not serie:
        return ""
    mx = max(serie) or 1
    n = len(serie)
    pts = " ".join(f"{i * ancho / max(1, n - 1):.1f},{alto - 3 - v * (alto - 8) / mx:.1f}" for i, v in enumerate(serie))
    area = f"0,{alto} {pts} {ancho},{alto}"
    m = ""
    if marca is not None and 0 <= marca < n:
        x = marca * ancho / max(1, n - 1)
        m = (f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{alto}" stroke="#0f172a" stroke-width="1.5" stroke-dasharray="3,2"/>'
             f'<circle cx="{x:.1f}" cy="{alto - 3 - serie[marca] * (alto - 8) / mx:.1f}" r="3" fill="#0f172a"/>')
    # Barras invisibles para tooltip nativo por ventana de 6h
    w_bar = max(2.0, ancho / max(1, n))
    hover = []
    for i, v in enumerate(serie):
        ts_b = pd.Timestamp(t0 + i * 21600, unit="s", tz="UTC").tz_convert("-03:00").strftime("%d/%m %H:%M")
        bx = i * ancho / max(1, n - 1) - w_bar / 2
        hover.append(f'<rect x="{max(0.0, bx):.1f}" y="0" width="{w_bar:.1f}" height="{alto}" fill="transparent"><title>{ts_b} hs: {v:,} comentarios</title></rect>')
    return (f'<svg width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}" style="overflow:visible">'
            f'<polygon points="{area}" fill="{color}20"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>{m}{"".join(hover)}</svg>')


def _barra(ap: int, re: int) -> str:
    tot = max(1, ap + re)
    a = 100 * ap / tot
    return (f'<div class="bar"><span style="width:{a:.1f}%;background:#16a34a" title="Apoya: {ap:,} ({a:.1f}%)"></span>'
            f'<span style="width:{100 - a:.1f}%;background:#dc2626" title="Rechaza: {re:,} ({100 - a:.1f}%)"></span></div>'
            f'<small><b>{a:.0f}%</b> apoya · <b>{100 - a:.0f}%</b> rechaza</small>')


def _horas(h: list[int]) -> str:
    mx = max(h) or 1
    barras = "".join(
        f'<rect x="{i * 7}" y="{30 - 28 * v / mx:.1f}" width="6" height="{max(1.0, 28 * v / mx) if v else 0:.1f}" fill="{"#7c3aed" if 0 <= i < 6 else "#9333ea"}" rx="1">'
        f'<title>{i:02d}:00 a {i:02d}:59 hs: {v:,} comentarios</title></rect>'
        for i, v in enumerate(h)
    )
    return f'<svg width="168" height="32">{barras}</svg>'


def _comparador_nulo(tam_azar: int, umbral_sinc: float, grp_df: pd.DataFrame) -> str:
    mx = max([tam_azar] + ([int(grp_df.cuentas.max())] if len(grp_df) else [10])) or 10
    w_az = min(100.0, 100.0 * tam_azar / mx)
    barras = [
        f'<div class="null-row"><div class="null-lbl">Máximo grupo por azar (Mundo Nulo)</div>'
        f'<div class="null-track"><div class="null-fill" style="width:{w_az:.1f}%;background:#94a3b8"></div></div>'
        f'<div class="null-val">{tam_azar} cuentas (umbral de descarte)</div></div>'
    ]
    for _, g in grp_df.iterrows():
        c = int(g.cuentas)
        w = min(100.0, 100.0 * c / mx)
        mult = f"{c / max(1, tam_azar):.1f}× el azar"
        barras.append(
            f'<div class="null-row"><div class="null-lbl"><b>Operación {html.escape(str(g.grupo))}</b></div>'
            f'<div class="null-track"><div class="null-fill" style="width:{w:.1f}%;background:#7c3aed"></div>'
            f'<div class="null-line" style="left:{w_az:.1f}%" title="Límite del azar ({tam_azar})"></div></div>'
            f'<div class="null-val"><b>{c} cuentas</b> <span class="kw">({mult})</span></div></div>'
        )
    return (
        f'<div class="null-card">'
        f'<div class="null-head"><div><b>Prueba contra el Mundo Nulo (permutación de cuentas por publicación)</b>'
        f'<div class="kw">Barajamos las cuentas conservando horarios y volumen exactos. Umbral de sincronía p99,95 = <b>{umbral_sinc}</b>. '
        f'Por azar nunca se forma un grupo mayor a <b>{tam_azar} cuentas</b>.</div></div></div>'
        f'{"".join(barras)}</div>'
    )


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
    n_emergentes = 0
    n_impulsadas = 0
    for i, r in nar.iterrows():
        serie = [0] * nb
        for b, n in vol[vol.narrativa == r.narrativa][["b", "n"]].itertuples(index=False):
            if 0 <= b < nb:
                serie[b] = int(n)
        t_or = pd.Timestamp(r.fecha_origen)
        b_or = int((t_or.timestamp() - t0) // 21600)
        emergente = (t_or.timestamp() - t0) > 86400
        if emergente:
            n_emergentes += 1
        cc = coord[coord.narrativa == r.narrativa]
        empuje = ""
        tiene_coord = False
        if len(cc):
            g = cc.groupby("grupo").n.sum().sort_values(ascending=False)
            tot_g = coord.groupby("grupo").n.sum()
            tags = [
                f'<a href="#op-{html.escape(k)}" class="tag" onclick="resaltar(\'op-{html.escape(k)}\')">🎯 {html.escape(k)} · {v} com.</a>'
                for k, v in g.items() if v >= 10 and v >= 0.15 * tot_g[k]
            ]
            if tags:
                tiene_coord = True
                n_impulsadas += 1
                empuje = " ".join(tags)
        ej_ap = e(f"select c.texto, count(*) n from comentarios c join etiquetas e using (comentario_id) where e.narrativa = '{r.narrativa}' and e.postura = 'apoya' and c.respuesta_a is null group by 1 order by 2 desc limit 2")
        ej_re = e(f"select c.texto, count(*) n from comentarios c join etiquetas e using (comentario_id) where e.narrativa = '{r.narrativa}' and e.postura = 'rechaza' group by 1 order by 2 desc limit 2")
        det_html = ""
        if len(ej_ap) or len(ej_re):
            ap_str = "<br>".join(f'🟢 <b>Apoya:</b> «{html.escape(str(x)[:110])}» <span class="kw">({int(n)})</span>' for x, n in ej_ap.itertuples(index=False))
            re_str = "<br>".join(f'🔴 <b>Rechaza:</b> «{html.escape(str(x)[:110])}» <span class="kw">({int(n)})</span>' for x, n in ej_re.itertuples(index=False))
            det_html = f'<details class="ej"><summary>Ver ejemplos (apoya vs. rechaza)</summary><div class="ej-box">{ap_str}<br>{re_str}</div></details>'
        pico = max(serie) if serie else 0
        et_orig = '<br><span class="new">⚡ emergente</span>' if emergente else '<br><small class="kw">desde el inicio</small>'
        et_emp = empuje or '<span class="kw">Orgánica</span>'
        filas.append(
            f'<tr id="nar-{r.narrativa}" class="nar-row" data-emergente="{1 if emergente else 0}" data-coord="{1 if tiene_coord else 0}">'
            f'<td><span class="nar-id" style="background:{COLORES[i % len(COLORES)]}18;color:{COLORES[i % len(COLORES)]}">{r.narrativa}</span></td>'
            f'<td><div class="tit">{html.escape(str(r.titulo))}</div>'
            f'<small class="kw">{html.escape(str(r.frases_clave)).replace("|", " · ")}</small>{det_html}</td>'
            f'<td style="white-space:nowrap"><b>{t_or.strftime("%d/%m %H:%M")}</b>{et_orig}</td>'
            f'<td>{_spark(serie, t0, color=COLORES[i % len(COLORES)], marca=b_or)}<div class="kw" style="font-size:11px">Pico: {pico:,} com./6h</div></td>'
            f'<td style="text-align:right;font-weight:600">{int(r.comentarios):,}</td>'
            f'<td>{_barra(int(r.apoya), int(r.rechaza))}</td>'
            f'<td>{et_emp}</td></tr>'
        )

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
                rasgos.append("Ráfaga nocturna (00-05h)")
            if cm.texto.value_counts().head(5).sum() / max(1, len(cm)) >= 0.35:
                rasgos.append("Plantilla repetida")
            else:
                rasgos.append("Textos variados / respuestas")
            badges = " ".join(f'<span class="tag">{r_txt}</span>' for r_txt in rasgos)
            principal = (f'<a href="#nar-{na}" onclick="resaltar(\'nar-{na}\')" style="color:#1e40af;text-decoration:none">'
                         f'<b>{na}</b> ({po}, {100 * k / len(nn):.0f}%)</a><br>'
                         f'<small>{html.escape(str(tit))}</small><br>{badges}')
        ej = cm.texto.value_counts().head(3).index.tolist()
        gfilas.append(
            f'<tr id="op-{g.grupo}"><td><span class="tag" style="font-weight:700;font-size:13px">{g.grupo}</span></td>'
            f'<td style="text-align:right;font-weight:700">{int(g.cuentas)}</td>'
            f'<td style="text-align:right">{len(cm):,}</td><td style="text-align:right">{cm.publicacion_id.nunique()}</td>'
            f'<td style="text-align:right">{"" if pd.isna(g.antig_dias) else f"{int(g.antig_dias)} d"}</td>'
            f'<td style="text-align:right">{"" if pd.isna(g.seguidores) else int(g.seguidores)}</td>'
            f'<td>{principal}</td><td>{_horas(h)}</td>'
            f'<td><small>{"<br>".join("«" + html.escape(x[:110]) + "»" for x in ej)}</small></td></tr>'
        )

    ruido = int(e("select count(*) from etiquetas where narrativa = 'ninguna'").iloc[0, 0])
    costo = {}
    cp = destino.parent / "costo.json"
    if cp.exists():
        costo = json.loads(cp.read_text(encoding="utf-8"))
    kpis = [
        ("Comentarios", f"{int(rango.n):,}", f"en {dias} días", "#2563eb"),
        ("Narrativas", f"{len(nar)}", f"{n_emergentes} emergentes · {n_impulsadas} con campaña", "#0891b2"),
        ("Ruido filtrado", f"{100 * ruido / max(1, rango.n):.0f}%", f"{ruido:,} comentarios", "#64748b"),
        ("Operaciones", f"{len(grp)}", f"{int(grp.cuentas.sum()) if len(grp) else 0} cuentas coordinadas", "#7c3aed"),
        ("Tiempo total", f"{costo.get('segundos', '?')} s", "DuckDB + MiniLM ONNX CPU", "#16a34a"),
        ("Costo IA", f"USD {costo.get('usd_ia', 0)}", f"{costo.get('tokens_ia', 0)} tokens", "#16a34a"),
    ]
    kp = "".join(
        f'<div class="kpi" style="border-top:3px solid {col}"><div class="v">{v}</div><div class="l">{k}</div><div class="s">{sub}</div></div>'
        for k, v, sub, col in kpis
    )
    comp_nulo = _comparador_nulo(int(info_c.get("tamano_azar", 4)), float(info_c.get("umbral_sincronia", 0.088)), grp)
    doc = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>QualyData · Narrativas y coordinación</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#f8fafc;color:#0f172a;margin:0;padding:24px 36px;line-height:1.4}}
.top-bar{{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:16px}}
h1{{margin:0 0 4px;font-size:26px;letter-spacing:-0.02em}} h2{{margin:28px 0 10px;font-size:19px;display:flex;align-items:center;gap:8px}}
.sub{{color:#475569;font-size:14px}}
.pills{{display:flex;gap:6px;flex-wrap:wrap}} .pill{{background:#e2e8f0;color:#1e293b;border-radius:999px;padding:4px 10px;font-size:12px;font-weight:600}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin-bottom:22px}}
.kpi{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;box-shadow:0 1px 2px rgba(15,23,42,0.03)}}
.kpi .v{{font-size:24px;font-weight:700;color:#0f172a}} .kpi .l{{color:#334155;font-size:13px;font-weight:600;margin-top:2px}} .kpi .s{{color:#64748b;font-size:11px;margin-top:2px}}
.toolbar{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px}}
.btns{{display:flex;gap:6px;flex-wrap:wrap}}
.btn{{background:#fff;border:1px solid #cbd5e1;color:#334155;padding:6px 12px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all .15s}}
.btn:hover{{background:#f1f5f9}} .btn.active{{background:#0f172a;color:#fff;border-color:#0f172a}}
.search{{padding:6px 12px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px;min-width:240px}}
table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;box-shadow:0 1px 2px rgba(15,23,42,0.03)}}
th,td{{padding:9px 11px;border-bottom:1px solid #f1f5f9;vertical-align:middle;font-size:13.5px}}
th{{background:#f1f5f9;text-align:left;font-size:12.5px;color:#334155;text-transform:uppercase;letter-spacing:0.03em}}
tr.nar-row:hover,tr[id^="op-"]:hover{{background:#f8fafc}}
.nar-id{{display:inline-block;padding:3px 8px;border-radius:6px;font-weight:700;font-size:12.5px}}
.tit{{font-weight:600;font-size:14px;color:#0f172a}} .kw{{color:#64748b}}
.bar{{display:flex;height:10px;width:160px;border-radius:5px;overflow:hidden;margin-bottom:3px}} .bar span{{display:block}}
.tag{{display:inline-block;background:#f3e8ff;color:#6b21a8;border:1px solid #e9d5ff;border-radius:6px;padding:2px 7px;margin:1px;font-size:12px;text-decoration:none;font-weight:500}}
a.tag:hover{{background:#e9d5ff}}
.new{{display:inline-block;background:#fef3c7;color:#92400e;border:1px solid #fde68a;border-radius:6px;padding:1px 6px;font-size:11.5px;font-weight:600;margin-top:2px}}
.nota{{color:#64748b;font-size:12.5px;margin-top:8px}}
.ej{{margin-top:5px}} .ej summary{{cursor:pointer;font-size:12px;color:#2563eb;user-select:none;font-weight:500}}
.ej-box{{margin-top:4px;padding:7px 9px;background:#f8fafc;border-left:3px solid #94a3b8;border-radius:4px;font-size:12px;color:#334155;line-height:1.45}}
.null-card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 18px;margin-bottom:12px;box-shadow:0 1px 2px rgba(15,23,42,0.03)}}
.null-head{{margin-bottom:10px;font-size:13.5px}}
.null-row{{display:grid;grid-template-columns:230px 1fr 190px;align-items:center;gap:12px;margin:6px 0;font-size:13px}}
.null-track{{position:relative;height:12px;background:#f1f5f9;border-radius:6px;overflow:hidden}}
.null-fill{{height:100%;border-radius:6px}}
.null-line{{position:absolute;top:0;bottom:0;width:2px;background:#0f172a}}
@keyframes flash{{0%{{background:#fef08a}}100%{{background:transparent}}}}
.destacado{{animation:flash 2.2s ease-out}}
</style></head><body>
<div class="top-bar">
  <div>
    <h1>¿Qué circula, desde cuándo y quién lo empuja?</h1>
    <div class="sub">Desafío QualyData · Hackathon Argentina Space 2026 · Análisis de narrativas y coordinación a escala</div>
  </div>
  <div class="pills">
    <span class="pill">🗄️ DuckDB + Parquet (zstd)</span>
    <span class="pill">🧠 Embeddings MiniLM ONNX (CPU)</span>
    <span class="pill">🕸️ Grafo Bipartito + Mundo Nulo</span>
  </div>
</div>
<div class="kpis">{kp}</div>

<h2>1. Narrativas detectadas</h2>
<div class="toolbar">
  <div class="btns">
    <button class="btn active" onclick="filtrar('todas', this)">Todas ({len(nar)})</button>
    <button class="btn" onclick="filtrar('emergentes', this)">⚡ Solo Emergentes ({n_emergentes})</button>
    <button class="btn" onclick="filtrar('coord', this)">🎯 Impulsadas por Operaciones ({n_impulsadas})</button>
    <button class="btn" onclick="filtrar('organicas', this)">💬 Solo Orgánicas ({len(nar) - n_impulsadas})</button>
    <button class="btn" onclick="toggleEjemplos()">📂 Expandir/Contraer ejemplos</button>
  </div>
  <input type="search" id="buscador" class="search" placeholder="Buscar narrativa o palabra clave..." oninput="buscarTexto(this.value)">
</div>
<table><thead><tr><th>ID</th><th>Narrativa (título y frases clave c-TF-IDF)</th><th>Origen</th><th>Evolución cada 6 h (línea punteada = origen)</th><th style="text-align:right">Volumen</th><th>Balance de Postura</th><th>Impulsada por</th></tr></thead>
<tbody id="tabla-nar">{''.join(filas)}</tbody></table>
<div class="nota">💡 Pasá el mouse por las barras y curvas para ver el detalle por ventana de 6 horas. Hacé clic en una operación (🎯) para saltar a su radiografía.</div>

<h2>2. Operaciones coordinadas y contraste contra el azar</h2>
{comp_nulo}
<table><thead><tr><th>Grupo</th><th style="text-align:right">Cuentas</th><th style="text-align:right">Comentarios</th><th style="text-align:right">Publicaciones</th><th style="text-align:right">Antigüedad med.</th><th style="text-align:right">Seguidores med.</th><th>Narrativa que empujan y huella detectada</th><th>Actividad por hora (0 a 23 h)</th><th>Textos representativos</th></tr></thead>
<tbody>{''.join(gfilas)}</tbody></table>
<script>
let abiertos = false;
function filtrar(modo, btn) {{
  document.querySelectorAll('.btns .btn').forEach(b => {{ if (b.textContent.indexOf('Expandir') === -1) b.classList.remove('active'); }});
  btn.classList.add('active');
  document.querySelectorAll('tr.nar-row').forEach(tr => {{
    const em = tr.getAttribute('data-emergente') === '1';
    const co = tr.getAttribute('data-coord') === '1';
    let ok = true;
    if (modo === 'emergentes') ok = em;
    else if (modo === 'coord') ok = co;
    else if (modo === 'organicas') ok = !co;
    tr.style.display = ok ? '' : 'none';
  }});
}}
function buscarTexto(q) {{
  q = q.toLowerCase().trim();
  document.querySelectorAll('tr.nar-row').forEach(tr => {{
    tr.style.display = (!q || tr.innerText.toLowerCase().indexOf(q) !== -1) ? '' : 'none';
  }});
}}
function toggleEjemplos() {{
  abiertos = !abiertos;
  document.querySelectorAll('details.ej').forEach(d => d.open = abiertos);
}}
function resaltar(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  el.style.display = '';
  el.classList.remove('destacado');
  void el.offsetWidth;
  el.classList.add('destacado');
}}
</script>
</body></html>"""
    destino.write_text(doc, encoding="utf-8")
