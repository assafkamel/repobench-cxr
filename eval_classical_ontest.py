"""Evaluate the classical LoG detector on the SAME 24-image test split used
for the DL model, so the two numbers are directly comparable."""
import torch, jsrt, detector
from evaluate import _froc, _bootstrap

test_names = torch.load("fasterrcnn_jsrt_test_names.pt")
recs = jsrt.scan_folder("jsrt")
test_recs = {n: recs[n] for n in test_names if n in recs and recs[n]["x"] is not None}
print(f"evaluating classical on {len(test_recs)} test images")

p = dict(thresh=0.02, min_circularity=0, min_contrast=0)
tp_by_sub, fp_all = {}, []
for name, r in test_recs.items():
    out = detector.detect(jsrt.load(r["path"]), **p)
    gt = jsrt.gt_of(r)
    cands = detector.mark_hits(out["cands"], gt)
    for c in cands:
        if c["hit"]:
            tp_by_sub.setdefault(r["subtlety"], []).append(c["score"])
        elif not c["near_hit"]:
            fp_all.append(c["score"])

n_les = len(test_recs)
n_img = len(test_recs)
tp_all = [s for v in tp_by_sub.values() for s in v]
targets = (0.5, 1, 2, 4, 8)
froc = _froc(tp_all, fp_all, n_les, n_img, targets)

print(f"\n{n_img} images, {n_les} lesions, {len(tp_all)} TP, {len(fp_all)} FP")
print(f"{'FP/img':>8} {'sens':>8}")
for t in targets:
    print(f"{t:>8.1f} {froc[t][0]:>8.3f}")

ci = _bootstrap(tp_all, fp_all, n_les, n_img, targets, B=1000)
print("\n95% CIs:")
for t in targets:
    lo, hi = ci[t]
    print(f"  {t:.1f} FP/img: [{lo:.3f}, {hi:.3f}]")

import csv
with open("classical_ontest_froc.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fp_per_img", "sensitivity", "ci_low", "ci_high"])
    for t in targets:
        s, _ = froc[t]
        lo, hi = ci[t]
        w.writerow([t, f"{s:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
print("\nwrote classical_ontest_froc.csv")
