"""FROC for the fine-tuned Faster R-CNN. Reuses evaluate.py's bootstrap logic.
   python evaluate_dl.py --data jsrt --model fasterrcnn_jsrt.pth"""
import argparse
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import jsrt
from jsrt_dataset import JSRTDataset, collate_fn
from evaluate import _froc, _bootstrap  # _froc_plot optional, add if absent


def build_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def pred_to_cands(pred, threshold=0.05):
    """pred: dict with boxes/scores/labels for ONE image. Returns list of dicts like detector.detect()."""
    cands = []
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    for box, sc in zip(boxes, scores):
        if sc < threshold:
            continue
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = max(x2 - x1, y2 - y1) / 2
        cands.append(dict(x=float(cx), y=float(cy), r=float(r), score=float(sc)))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="jsrt")
    ap.add_argument("--model", required=True)
    ap.add_argument("--score_thresh", type=float, default=0.05)
    ap.add_argument("--out", default="dl_results")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(2).to(device)
    model.load_state_dict(torch.load(a.model, map_location=device))
    model.eval()

    recs = jsrt.scan_folder(a.data)
    test_names = torch.load(a.model.replace(".pth", "_test_names.pt"))
    test_recs = {n: recs[n] for n in test_names if n in recs and recs[n]["x"] is not None}
    print(f"evaluating on {len(test_recs)} annotated test images")

    tp_by_sub, fp_all = {}, []
    with torch.no_grad():
        for name, rec in test_recs.items():
            img = jsrt.load(rec["path"])
            img_t = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device) / 255.0
            pred = model(img_t)[0]
            cands = pred_to_cands(pred, a.score_thresh)
            gt = jsrt.gt_of(rec)
            hit_flag = False
            for c in cands:
                near = np.hypot(c["x"] - gt["x"], c["y"] - gt["y"]) <= max(gt["r"], 8)
                if near and not hit_flag:
                    tp_by_sub.setdefault(rec["subtlety"], []).append(c["score"])
                    hit_flag = True
                elif not near:
                    fp_all.append(c["score"])
            print(f"\r{name}: {len(cands)} cands", end="")

    n_les = len(test_recs)
    n_img = len(test_recs)
    tp_all = [s for v in tp_by_sub.values() for s in v]
    targets = (0.5, 1, 2, 4, 8)
    froc = _froc(tp_all, fp_all, n_les, n_img, targets)
    print(f"\n\n{n_img} images, {n_les} lesions, {len(tp_all)} TP, {len(fp_all)} FP")
    print(f"{'FP/img':>8} {'sens':>8}")
    for t in targets:
        print(f"{t:>8.1f} {froc[t][0]:>8.3f}")

    if a.boot:
        ci = _bootstrap(tp_all, fp_all, n_les, n_img, targets, B=a.boot)
        print("\n95% CIs:")
        for t in targets:
            lo, hi = ci[t]
            print(f"  {t:.1f} FP/img: [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
