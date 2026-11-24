"""JSRT loading: .IMG files, annotations, optional SCR lung masks."""
import csv, glob, os, re
from functools import lru_cache
import cv2, numpy as np
from PIL import Image
from detector import GRID, WORK, MM_PER_PX

NUM_ORDER = ["subtlety", "size_mm", "age", "x", "y"]


def _read(path):
    if path.upper().endswith(".IMG"):
        a = np.fromfile(path, dtype=">u2")
        n = int(round(np.sqrt(a.size)))
        return a.reshape(n, n).astype(np.float32), True
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32), False


@lru_cache(maxsize=16)
def load(path, invert="auto"):
    a, raw = _read(path)
    a = cv2.resize(a, (WORK, WORK), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    if invert != "off":
        c = np.r_[a[:80, :80].ravel(), a[:80, -80:].ravel()].mean()
        if invert == "on" or c > np.percentile(a, 10):
            a = 255 - a
    return a


def scan_folder(folder):
    files = [f for ext in ("IMG", "img", "png", "jpg", "jpeg")
             for f in glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True)]
    recs = {}
    for f in sorted(set(files)):
        name = os.path.splitext(os.path.basename(f))[0]
        recs[name] = dict(name=name, path=f, label="non-nodule" if name.startswith("JPCNN") else "unknown",
                          subtlety=None, size_mm=None, age=None, sex=None, x=None, y=None)
    for f in glob.glob(os.path.join(folder, "**", "annotations.csv"), recursive=True):
        for row in csv.DictReader(open(f, encoding="utf-8-sig")):
            r = recs.get(os.path.splitext(row["file"])[0])
            if r:
                r.update(x=float(row["x"]), y=float(row["y"]), size_mm=float(row["diameter_mm"]),
                         subtlety=int(row["subtlety"]) if row.get("subtlety") else None,
                         label=row.get("label") or "nodule")
        return recs
    for f in glob.glob(os.path.join(folder, "**", "CLNDAT_EN.txt"), recursive=True):
        for line in open(f, errors="ignore"):
            toks = line.split()
            if not toks or not toks[0].upper().endswith(".IMG"):
                continue
            r = recs.get(os.path.splitext(toks[0])[0])
            if not r:
                continue
            low = line.lower()
            r["label"] = "malignant" if "malignant" in low else "benign" if "benign" in low else r["label"]
            r["sex"] = "male" if re.search(r"\bmale\b", low) else "female" if "female" in low else None
            nums = [float(t) for t in toks[1:] if re.fullmatch(r"\d+(\.\d+)?", t)]
            for k, v in zip(NUM_ORDER, nums):
                r[k] = v
            if r["subtlety"]:
                r["subtlety"] = int(r["subtlety"])
    return recs


def scr_mask(name, mask_dir):
    m = None
    for p in glob.glob(os.path.join(mask_dir, "**", f"{name}.gif"), recursive=True):
        if "lung" in p.lower():
            a = (np.array(Image.open(p).convert("L")) > 0).astype(np.uint8)
            m = a if m is None else np.maximum(m, a)
    return None if m is None else cv2.resize(m, (WORK, WORK), interpolation=cv2.INTER_NEAREST)


def gt_of(rec, mm_per_px=MM_PER_PX):
    if rec["x"] is None or rec["y"] is None:
        return None
    s = WORK / GRID
    return dict(x=rec["x"] * s, y=rec["y"] * s, r=(rec["size_mm"] or 15) / 2 / mm_per_px)
