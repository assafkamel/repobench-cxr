"""Shareable research cases (no PHI). Store is a local JSON dir."""
import hashlib, json, os, time
from pathlib import Path

CASES = Path(os.environ.get("CASES_DIR", "cases"))
CASES.mkdir(exist_ok=True)


def _slug(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def save_case(name, params, rec_meta, overlay_png_bytes, author="", notes=""):
    payload = dict(schema=1, name=name, params=params, meta=rec_meta,
                   author=author[:120], notes=notes[:2000], created=time.time())
    slug = _slug(payload)
    d = CASES / slug
    d.mkdir(exist_ok=True)
    (d / "case.json").write_text(json.dumps(payload, indent=2))
    (d / "overlay.png").write_bytes(overlay_png_bytes)
    return slug


def load_case(slug):
    d = CASES / slug
    if not (d / "case.json").exists():
        return None
    payload = json.loads((d / "case.json").read_text())
    payload["overlay_png"] = (d / "overlay.png").read_bytes()
    return payload


def list_cases():
    out = []
    for p in sorted(CASES.glob("*/case.json")):
        j = json.loads(p.read_text())
        out.append({k: j.get(k) for k in ("schema", "name", "author", "notes", "created")}
                   | {"slug": p.parent.name})
    return out
