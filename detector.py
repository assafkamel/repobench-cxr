"""Classical nodule candidate detector for chest X-rays (OpenCV only, no training)."""
import cv2
import numpy as np
from functools import lru_cache

GRID = 2048
WORK = 1024
MM_PER_PX = 0.35

DEFAULTS = dict(
    clip=2.0, tile=8, blur=1.0, bg_sigma=30.0,
    min_mm=6.0, max_mm=30.0, n_scales=6,
    thresh=0.5, max_cands=15, mm_per_px=MM_PER_PX,
    min_contrast=0.03, min_circularity=0.25, rank_mode="score",
)
INT_KEYS = ("tile", "n_scales", "max_cands")


@lru_cache(maxsize=32)
def _sigmas(min_mm, max_mm, n_scales, mm_per_px):
    rmin, rmax = min_mm / 2 / mm_per_px, max_mm / 2 / mm_per_px
    return np.geomspace(max(rmin / 1.414, 2.0), max(rmax / 1.414, 2.5), n_scales)


def lung_mask(g):
    n0 = 256
    s = cv2.GaussianBlur(cv2.resize(g, (n0, n0), interpolation=cv2.INTER_AREA), (5, 5), 0)
    t, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx((s < t).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a > 0.01 * n0 * n0:
            touches = (x == 0 or y == 0 or x + w >= n0 or y + h >= n0)
            comps.append((touches, -a, i))
    if not comps:
        return np.ones((WORK, WORK), np.uint8)
    interior = [c for c in comps if not c[0]]
    comps = interior if interior else comps
    comps.sort(key=lambda c: c[1])
    m = np.isin(lab, [c[2] for c in comps[:2]]).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnts, -1, 1, -1)
    return cv2.resize(m, (WORK, WORK), interpolation=cv2.INTER_NEAREST)


def _contrast(f, x, y, r):
    H, W = f.shape
    R = int(2.5 * r) + 1
    y0, y1, x0, x1 = max(0, y - R), min(H, y + R + 1), max(0, x - R), min(W, x + R + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    c = f[y0:y1, x0:x1]
    inn, out = c[d <= r], c[(d > 1.5 * r) & (d <= 2.5 * r)]
    return float(inn.mean() - out.mean()) if inn.size and out.size else 0.0


def _circularity(f, x, y, r):
    H, W = f.shape
    R = int(1.5 * r) + 2
    y0, y1, x0, x1 = max(0, y - R), min(H, y + R + 1), max(0, x - R), min(W, x + R + 1)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return 0.0
    roi = f[y0:y1, x0:x1]
    _, bw = cv2.threshold((roi * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    A, P = cv2.contourArea(c), cv2.arcLength(c, True)
    return float(4 * np.pi * A / (P * P + 1e-9)) if P > 0 else 0.0


def detect(gray, mask=None, use_mask=True, **kw):
    p = {**DEFAULTS, **kw}
    for k in INT_KEYS:
        p[k] = int(p[k])
    g = gray if gray.shape == (WORK, WORK) else cv2.resize(gray, (WORK, WORK), interpolation=cv2.INTER_AREA)
    enh = cv2.createCLAHE(clipLimit=p["clip"], tileGridSize=(p["tile"], p["tile"])).apply(g)
    f = enh.astype(np.float32) / 255
    if p["blur"] > 0:
        f = cv2.GaussianBlur(f, (0, 0), p["blur"])
    if not use_mask:
        m = np.ones((WORK, WORK), np.uint8)
    elif mask is not None:
        m = cv2.resize(mask.astype(np.uint8), (WORK, WORK), interpolation=cv2.INTER_NEAREST)
    else:
        m = lung_mask(g)
    f = f - cv2.GaussianBlur(f, (0, 0), p["bg_sigma"])
    sel = m > 0
    f /= (f[sel].std() + 1e-6) if sel.any() else (f.std() + 1e-6)
    sig = _sigmas(float(p["min_mm"]), float(p["max_mm"]), int(p["n_scales"]), float(p["mm_per_px"]))
    stack = np.stack([-(s * s) * cv2.Laplacian(cv2.GaussianBlur(f, (0, 0), float(s)), cv2.CV_32F) for s in sig])
    R, S = stack.max(0), stack.argmax(0)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    falloff = np.clip(dist / 12.0, 0.0, 1.0)
    R = R * falloff
    rmin = float(p["min_mm"]) / 2 / float(p["mm_per_px"])
    k = int(max(9, 2 * rmin)) | 1
    peaks = (R == cv2.dilate(R, np.ones((k, k), np.uint8))) & (R > p["thresh"])
    ys, xs = np.nonzero(peaks)
    raw = [(float(R[y, x]), int(x), int(y), float(sig[S[y, x]] * 1.414)) for y, x in zip(ys, xs)]
    raw.sort(key=lambda t: -t[0])
    kept = []
    for score, x, y, r in raw:
        if all(np.hypot(x - kx, y - ky) > 0.75 * max(r, kr) for _, kx, ky, kr in kept):
            kept.append((score, x, y, r))
    cands = []
    for score, x, y, r in kept:
        con = _contrast(f, x, y, r)
        cir = _circularity(f, x, y, r)
        if con < p["min_contrast"] or cir < p["min_circularity"]:
            continue
        composite = score * max(con, 0.0) * cir
        cands.append(dict(
            x=x, y=y, r=r, score=score, contrast=con, circularity=cir,
            composite=composite, diam_mm=2 * r * p["mm_per_px"],
        ))
    key = "composite" if p["rank_mode"] == "composite" else "score"
    cands.sort(key=lambda c: -c[key])
    # NEW: penalize candidates sitting on rib/vessel lines and re-rank by score_adj
    rib_suppress(g, cands, weight=0.3)
    return dict(enh=enh, mask=m, cands=cands[:p["max_cands"]], params=p)


def mark_hits(cands, gt):
    for c in cands:
        c["hit"] = False
        c["near_hit"] = False
    if not gt:
        return cands
    rad = max(gt["r"], 8)
    best = None
    for c in cands:
        if np.hypot(c["x"] - gt["x"], c["y"] - gt["y"]) <= rad:
            c["near_hit"] = True
            if best is None or c["score"] > best["score"]:
                best = c
    if best is not None:
        best["hit"] = True
    return cands
    
def rib_suppress(gray, cands, weight=0.3):
    """Detect rib arcs via black-hat, penalize candidates near rib lines.
    Must be called *inside* detect(), after R is computed and candidates are built,
    because it needs `gray` at WORK resolution.
    Operates on c['score'] in place and sets c['score_adj'] = penalized score."""
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bh = cv2.threshold(bh, bh.mean() * 1.5, 255, cv2.THRESH_BINARY)[1]
    dist_to_rib = cv2.distanceTransform(255 - bh, cv2.DIST_L2, 5)
    for c in cands:
        d = dist_to_rib[c["y"], c["x"]]
        c["rib_dist"] = float(d)
        penalty = np.exp(-d / 8.0)           # 0 px -> 1.0, 16 px -> 0.14
        c["score_adj"] = c["score"] * (1 - weight * (1 - penalty))
    cands.sort(key=lambda c: -c["score_adj"])
    return cands
