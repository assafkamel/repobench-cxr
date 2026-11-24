"""Lazy-loading singleton for the trained Faster R-CNN. Safe to import
even if torch isn't installed or the checkpoint doesn't exist yet."""
import os
import numpy as np

_MODEL = None
_DEVICE = None
_LOAD_ERROR = None


def _try_load(checkpoint):
    global _MODEL, _DEVICE, _LOAD_ERROR
    if _LOAD_ERROR is not None:
        return None
    if _MODEL is not None:
        return _MODEL
    try:
        import torch
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as e:
        _LOAD_ERROR = f"torch not installed: {e}"
        return None
    if not os.path.exists(checkpoint):
        _LOAD_ERROR = f"checkpoint not found: {checkpoint}"
        return None
    try:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
        m.load_state_dict(torch.load(checkpoint, map_location=_DEVICE))
        m.to(_DEVICE).eval()
        _MODEL = m
        return _MODEL
    except Exception as e:
        _LOAD_ERROR = f"failed to load: {type(e).__name__}: {e}"
        return None


def detect_dl(gray_uint8, checkpoint, score_thresh=0.05):
    """Run Faster R-CNN on a uint8 (1024,1024) image. Returns (cands, error).
    cands: list of dicts {x, y, r, score}. error: str or None."""
    import torch
    model = _try_load(checkpoint)
    if model is None:
        return [], _LOAD_ERROR
    img = torch.from_numpy(gray_uint8).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
    img = img.unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        pred = model(img)[0]
    cands = []
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    for box, sc in zip(boxes, scores):
        if sc < score_thresh:
            continue
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = max(x2 - x1, y2 - y1) / 2
        cands.append(dict(x=float(cx), y=float(cy), r=float(r), score=float(sc)))
    cands.sort(key=lambda c: -c["score"])
    return cands, None


def status(checkpoint):
    """For /api/status — cheap, doesn't load the model."""
    if _LOAD_ERROR is not None:
        return {"ok": False, "error": _LOAD_ERROR}
    if _MODEL is not None:
        return {"ok": True, "loaded": True, "device": str(_DEVICE)}
    if not os.path.exists(checkpoint):
        return {"ok": False, "error": f"checkpoint not found: {checkpoint}"}
    try:
        import torch  # noqa: F401
    except ImportError:
        return {"ok": False, "error": "torch not installed"}
    return {"ok": True, "loaded": False, "checkpoint": checkpoint}
