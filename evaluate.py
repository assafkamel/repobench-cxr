"""FROC evaluation with bootstrap CIs.  python evaluate.py --data JSRT [--masks SCR]"""
import argparse, csv
import numpy as np
import detector, jsrt


def _froc(tp_scores, fp_scores, n_lesions, n_images, targets):
    ts = np.array(sorted(set(tp_scores) | set(fp_scores), reverse=True))
    if ts.size == 0:
        return {t: (0.0, np.inf) for t in targets}
    tps, fps = np.array(tp_scores), np.array(fp_scores)
    sens = np.array([(tps >= t).sum() for t in ts]) / max(n_lesions, 1)
    fppi = np.array([(fps >= t).sum() for t in ts]) / max(n_images, 1)
    out = {}
    for tgt in targets:
        ok = fppi <= tgt
        out[tgt] = (float(sens[ok].max()) if ok.any() else 0.0, float(tgt))
    return out


def _bootstrap(tp_scores, fp_scores, n_lesions, n_images, targets, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    tp, fp = np.asarray(tp_scores), np.asarray(fp_scores)
    draws = {t: [] for t in targets}
    for _ in range(B):
        ti = rng.integers(0, len(tp), len(tp)) if len(tp) else np.array([], int)
        fi = rng.integers(0, len(fp), len(fp)) if len(fp) else np.array([], int)
        r = _froc(tp[ti].tolist(), fp[fi].tolist(), n_lesions, n_images, targets)
        for t in targets:
            draws[t].append(r[t][0])
    return {t: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for t, v in draws.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--masks")
    ap.add_argument("--mask", default="auto", choices=["auto", "scr", "none"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--boot", type=int, default=1000)
    for k, v in detector.DEFAULTS.items():
        ap.add_argument(f"--{k}", type=float if not isinstance(v, str) else str, default=v)
    # evaluate at the same operating point the UI uses, so numbers are comparable
    ap.set_defaults(thresh=0.5, max_cands=15, min_contrast=0.03, min_circularity=0.25)
    a = ap.parse_args()
    p = {k: getattr(a, k) for k in detector.DEFAULTS}
    for k in detector.INT_KEYS:
        p[k] = int(p[k])
    recs = list(jsrt.scan_folder(a.data).values())
    rows, tp_by_sub, fp_all = [], {}, []
    n_les = 0
    for i, r in enumerate(recs):
        mask = jsrt.scr_mask(r["name"], a.masks) if a.mask == "scr" and a.masks else None
        out = detector.detect(jsrt.load(r["path"]), mask=mask, use_mask=a.mask != "none", **p)
        gt = jsrt.gt_of(r)
        cands = detector.mark_hits(out["cands"], gt)
        n_les += gt is not None
        # in the candidate loop, after detector.mark_hits:
        for k in cands:
            sc = k.get("score_adj", k["score"])   # <-- use suppressed score
            if k["hit"]:
                tp_by_sub.setdefault(r["subtlety"], []).append(sc)
            elif not k["near_hit"]:
                fp_all.append(sc)
            rows.append([r["name"], r["label"], r["subtlety"], k["x"], k["y"],
                         round(k["score"], 3), round(k.get("contrast", 0), 3),
                         round(k.get("circularity", 0), 3), int(k["hit"]), int(k["near_hit"])])
        print(f"\r{i+1}/{len(recs)}  lesions={n_les}  FP={len(fp_all)}", end="")
    n_img = len(recs)
    tp_all = [s for v in tp_by_sub.values() for s in v]
    targets = (0.5, 1, 2, 4, 8)
    froc = _froc(tp_all, fp_all, n_les, n_img, targets)
    print(f"\n\n{n_img} images, {n_les} lesions, {len(tp_all)} TP, {len(fp_all)} FP")
    print(f"{'FP/img':>8} {'sens':>8} {'95% CI':>18}")
    ci = _bootstrap(tp_all, fp_all, n_les, n_img, targets, B=a.boot) if a.boot else {t: (float('nan'),) * 2 for t in targets}
    for t in targets:
        s, _ = froc[t]
        lo, hi = ci[t]
        print(f"{t:>8.1f} {s:>8.3f}   [{lo:.3f}, {hi:.3f}]")
    print("\nby subtlety @ <=2 FP/img:")
    ts = sorted(set(tp_all) | set(fp_all), reverse=True)
    if ts:
        fppi = [sum(1 for s in fp_all if s >= x) / n_img for x in ts]
        sens = [sum(1 for s in tp_all if s >= x) / max(n_les, 1) for x in ts]
        ok = np.array(fppi) <= 2
        thresh = ts[int(np.argmax(ok))] if ok.any() else ts[-1]
        for sub in range(1, 6):
            hits = [s for s in tp_by_sub.get(sub, []) if s >= thresh]
            tot = sum(1 for r in recs if r["subtlety"] == sub and r["x"] is not None)
            if tot:
                print(f"  subtlety {sub}: {len(hits)}/{tot}  ({len(hits)/tot:.3f})")
    with open(a.out + "_candidates.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "label", "subtlety", "x", "y", "score",
                    "contrast", "circularity", "hit", "near_hit"])
        w.writerows(rows)
    with open(a.out + "_froc.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["fp_per_img", "sensitivity", "ci_low", "ci_high"])
        for t in targets:
            s, _ = froc[t]
            lo, hi = ci[t]
            w.writerow([t, f"{s:.4f}", f"{lo:.4f}", f"{hi:.4f}"])


if __name__ == "__main__":
    main()
