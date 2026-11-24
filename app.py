"""Thorax Lab: local UI + shareable cases.  python app.py --data path/to/JSRT [--masks path/to/SCR]"""
import argparse, base64, os
import cv2, numpy as np
from flask import Flask, jsonify, request, send_from_directory
import detector, jsrt, cases
import dl_model

app = Flask(__name__, static_folder="static")
RECS, MASKDIR = {}, None
DL_CHECKPOINT = "fasterrcnn_jsrt.pth"

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/list")
def api_list():
    keys = ("name", "label", "subtlety", "size_mm", "age", "sex")
    return jsonify([{k: r[k] for k in keys} for r in RECS.values()])


@app.route("/api/detect")
def api_detect():
    a = request.args
    name = a.get("name", "")
    rec = RECS.get(name)
    if not rec:
        return jsonify(error=f"unknown image '{name}'"), 404
    img = jsrt.load(rec["path"], a.get("invert", "auto"))
    p = {k: (a[k] if k == "rank_mode" else float(a[k]))
     for k in detector.DEFAULTS if k in a}
    src = a.get("mask", "auto")
    mask = jsrt.scr_mask(rec["name"], MASKDIR) if src == "scr" and MASKDIR else None
    out = detector.detect(img, mask=mask, use_mask=src != "none", **p)
    gt = jsrt.gt_of(rec, p.get("mm_per_px", detector.MM_PER_PX))
    cands = detector.mark_hits(out["cands"], gt)
    view = out["enh"] if a.get("view", "enh") == "enh" else img
    jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 95,
                                     cv2.IMWRITE_JPEG_OPTIMIZE, 1])[1].tobytes()
    cnts, _ = cv2.findContours(out["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [cv2.approxPolyDP(c, 1.0, True)[:, 0, :].tolist() for c in cnts]
    return jsonify(image="data:image/jpeg;base64," + base64.b64encode(jpg).decode(),
                   cands=cands, gt=gt, contours=polys,
                   meta={k: rec[k] for k in rec if k != "path"},
                   mm_per_px=p.get("mm_per_px", detector.MM_PER_PX))


# ── shareable cases ─────────────────────────────────────────────────────────
PUBLIC_PREFIXES = ("JPCLN", "JPCNN", "JSRT")


@app.route("/api/cases", methods=["GET"])
def api_cases():
    return jsonify(cases.list_cases())


@app.route("/api/cases/<slug>", methods=["GET"])
def api_case(slug):
    c = cases.load_case(slug)
    if c is None:
        return jsonify(error="not found"), 404
    return jsonify(name=c["name"], params=c["params"], meta=c["meta"],
                   author=c["author"], notes=c["notes"], created=c["created"],
                   overlay="data:image/png;base64," + base64.b64encode(c["overlay_png"]).decode())


@app.route("/api/cases", methods=["POST"])
def api_create_case():
    token = os.environ.get("CASES_TOKEN")
    if not token or request.headers.get("X-Cases-Token") != token:
        return jsonify(error="unauthorized"), 401
    body = request.get_json(force=True, silent=True) or {}
    name = body.get("name", "")
    if not name.startswith(PUBLIC_PREFIXES):
        return jsonify(error="only public/JSRT images may be shared"), 400
    rec = RECS.get(name)
    if not rec:
        return jsonify(error=f"unknown image '{name}'"), 400
    params = {k: float(body["params"][k]) for k in detector.DEFAULTS if k in body.get("params", {})}
    src = body.get("mask", "auto")
    mask = jsrt.scr_mask(name, MASKDIR) if src == "scr" and MASKDIR else None
    out = detector.detect(jsrt.load(rec["path"]), mask=mask, use_mask=src != "none", **params)
    gt = jsrt.gt_of(rec, params.get("mm_per_px", detector.MM_PER_PX))
    cands = detector.mark_hits(out["cands"], gt)
    ov = np.zeros((detector.WORK, detector.WORK, 4), np.uint8)
    ov[out["mask"] > 0] = (0, 255, 0, 60)
    for c in cands:
        color = (0, 255, 0, 255) if c["hit"] else (0, 0, 255, 220)
        cv2.circle(ov, (c["x"], c["y"]), int(c["r"]), color, 2)
    if gt:
        cv2.circle(ov, (int(gt["x"]), int(gt["y"])), int(gt["r"]), (255, 255, 0, 255), 2)
    ok, png = cv2.imencode(".png", ov)
    if not ok:
        return jsonify(error="png encode failed"), 500
    import re as _re
    notes = _re.sub(r"\b\d{6,}\b", "[redacted]", body.get("notes", ""))
    slug = cases.save_case(name=name, params=params,
                           rec_meta={k: rec[k] for k in rec if k != "path"},
                           overlay_png_bytes=png.tobytes(),
                           author=body.get("author", ""), notes=notes)
    return jsonify(slug=slug, permalink=f"/api/cases/{slug}")

# ── DL detection ──────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    return jsonify(dl_model.status(DL_CHECKPOINT))


@app.route("/api/detect_dl")
def api_detect_dl():
    a = request.args
    name = a.get("name", "")
    rec = RECS.get(name)
    if not rec:
        return jsonify(error=f"unknown image '{name}'"), 404
    img = jsrt.load(rec["path"], a.get("invert", "auto"))
    score_thresh = float(a.get("score_thresh", 0.05))
    cands, err = dl_model.detect_dl(img, DL_CHECKPOINT, score_thresh=score_thresh)
    if err:
        return jsonify(error=err, cands=[]), 503
    gt = jsrt.gt_of(rec, detector.MM_PER_PX)
    cands = detector.mark_hits(cands, gt)
    # blank lung mask (DL doesn't use it) — empty contours so the frontend's
    # overlay code doesn't break
    mask = np.zeros((detector.WORK, detector.WORK), np.uint8)
    view = img
    jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 95,
                                     cv2.IMWRITE_JPEG_OPTIMIZE, 1])[1].tobytes()
    return jsonify(image="data:image/jpeg;base64," + base64.b64encode(jpg).decode(),
                   cands=cands, gt=gt, contours=[],
                   meta={k: rec[k] for k in rec if k != "path"},
                   mm_per_px=detector.MM_PER_PX,
                   mode="dl", score_thresh=score_thresh)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--model", default="fasterrcnn_jsrt.pth",
                help="path to trained Faster R-CNN checkpoint")
    a = ap.parse_args()
    DL_CHECKPOINT = a.model
    RECS.update(jsrt.scan_folder(a.data))
    MASKDIR = a.masks
    print(f"{len(RECS)} images, {sum(r['x'] is not None for r in RECS.values())} with annotations")
    st = dl_model.status(DL_CHECKPOINT)
    print(f"DL model: {'loaded' if st.get('loaded') else st.get('error', 'available, will load lazily')}")
    print(f"{len(RECS)} images, {sum(r['x'] is not None for r in RECS.values())} with annotations")
    app.run(port=a.port, debug=False)
