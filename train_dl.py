"""Fine-tune COCO-pretrained Faster R-CNN on JSRT.  python train_dl.py --data jsrt"""
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import jsrt
from jsrt_dataset import JSRTDataset, collate_fn, split_recs


def build_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, loader, device, epoch, log_every=10):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % log_every == 0:
            print(f"  epoch {epoch} [{i+1}/{len(loader)}] loss={running_loss/(i+1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="jsrt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--out", default="fasterrcnn_jsrt.pth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    recs = jsrt.scan_folder(a.data)
    train_recs, val_recs, test_recs = split_recs(recs, seed=a.seed)
    print(f"train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}")

    train_ds = JSRTDataset({r["name"]: r for r in train_recs}, train=True)
    val_ds = JSRTDataset({r["name"]: r for r in val_recs}, train=False)
    test_ds = JSRTDataset({r["name"]: r for r in test_recs}, train=False)

    train_loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    # save test_ds for evaluate_dl.py to consume later
    torch.save([r["name"] for r in test_recs], a.out.replace(".pth", "_test_names.pt"))

    model = build_model(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=0.0005)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(a.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_sched.step()
        torch.save(model.state_dict(), a.out)
        print(f"  saved checkpoint -> {a.out}")

    print(f"\ndone. Model at {a.out}; test image names at {a.out.replace('.pth', '_test_names.pt')}")


if __name__ == "__main__":
    main()
