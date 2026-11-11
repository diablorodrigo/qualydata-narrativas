"""IA opcional y barata: una sola consulta por narrativa (no por comentario).

Se activa solo si están definidas las variables de entorno:
    QN_IA_URL    endpoint compatible con OpenAI, p. ej. https://api.x.ai/v1/chat/completions
    QN_IA_CLAVE  la clave (nunca va al repo)
    QN_IA_MODELO nombre del modelo
Para cada narrativa le mostramos 6 frases de cada lado y le pedimos: qué lado
afirma la narrativa y un título corto. Si algo falla, el pipeline sigue sin IA.
"""
import json
import os
import urllib.request

PRECIO_USD_POR_MILLON = float(os.environ.get("QN_IA_USD_MILLON", "0.5"))


def disponible() -> bool:
    return bool(os.environ.get("QN_IA_URL") and os.environ.get("QN_IA_CLAVE") and os.environ.get("QN_IA_MODELO"))


def orientar(lado0: list[str], lado1: list[str]) -> tuple[int | None, str | None, int]:
    """Devuelve (lado que afirma 0/1 o None, título, tokens usados)."""
    prompt = (
        "Dos grupos de comentarios discuten la misma afirmación. Un grupo la AFIRMA (denuncia, reclama o "
        "sostiene) y el otro la REFUTA o la niega.\n"
        f"Grupo A:\n- " + "\n- ".join(lado0[:6]) + "\n"
        f"Grupo B:\n- " + "\n- ".join(lado1[:6]) + "\n"
        'Respondé solo JSON: {"afirma": "A" o "B", "titulo": "la afirmación en una frase corta"}'
    )
    body = json.dumps({"model": os.environ["QN_IA_MODELO"], "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(os.environ["QN_IA_URL"], data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {os.environ['QN_IA_CLAVE']}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read())
        txt = out["choices"][0]["message"]["content"]
        tokens = int(out.get("usage", {}).get("total_tokens", 0))
        j = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        lado = {"A": 0, "B": 1}.get(str(j.get("afirma", "")).strip().upper()[:1])
        return lado, (j.get("titulo") or None), tokens
    except Exception:
        return None, None, 0
