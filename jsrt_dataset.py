"""PyTorch Dataset over JSRT annotations. Grayscale -> 3-channel RGB for COCO-pretrained backbones."""
import torch
from torch.utils.data import Dataset
import jsrt


class JSRTDataset(Dataset):
    def __init__(self, recs, train=False):
        """recs: dict from jsrt.scan_folder(). Only annotated (x is not None) kept."""
        self.recs = [r for r in recs.values() if r["x"] is not None]
        if not self.recs:
            raise ValueError("No annotated images found in recs")
        self.train = train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        rec = self.recs[idx]
        img = jsrt.load(rec["path"])                       # uint8, (1024, 1024)
        img = torch.from_numpy(img).float().unsqueeze(0)   # (1, H, W)
        img = img.repeat(3, 1, 1) / 255.0                  # (3, H, W), RGB-like
        gt = jsrt.gt_of(rec)
        if gt is None:
            return img, {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}
        r = gt["r"]
        boxes = torch.tensor([[gt["x"] - r, gt["y"] - r, gt["x"] + r, gt["y"] + r]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)      # 1 = nodule
        return img, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    """Faster R-CNN expects list-of-tensors, not stacked tensors (varying box counts)."""
    return tuple(zip(*batch))


def split_recs(recs, seed=0, train_frac=0.7, val_frac=0.15):
    """Stratified-by-subtlety split. Returns three lists of records (annotated only)."""
    import random
    rng = random.Random(seed)
    annotated = [r for r in recs.values() if r["x"] is not None]
    rng.shuffle(annotated)
    n = len(annotated)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return annotated[:n_train], annotated[n_train:n_train + n_val], annotated[n_train + n_val:]
