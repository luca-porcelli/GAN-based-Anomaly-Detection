import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.ganomaly import GANomaly
from src.dataset import ImageFolderFlat
from src.utils import load_json
from sklearn.metrics import roc_auc_score

def parse_args():
    p = argparse.ArgumentParser(description="Inference for GANomaly-style model")
    p.add_argument("--data_root", type=str, default="data/mvtec")
    p.add_argument("--category", type=str, default="bottle")
    p.add_argument("--weights", type=str, default="runs/bottle_1756463597/weights/best.pt")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--lambda_img", type=float, default=None)
    p.add_argument("--lambda_latent", type=float, default=None)
    p.add_argument("--z_dim", type=int, default=128)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_json("configs.json", {})
    img_size = args.img_size or cfg.get("img_size", 256)
    lambda_img = args.lambda_img or cfg.get("lambda_img", 50.0)
    lambda_latent = args.lambda_latent or cfg.get("lambda_latent", 1.0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GANomaly(in_c=3, z_dim=args.z_dim, lambda_img=lambda_img, lambda_latent=lambda_latent).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    test_dir = Path(args.data_root)/args.category/"test"
    ds = ImageFolderFlat(str(test_dir), img_size=img_size, return_label=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    paths, scores, labels = [], [], []
    with torch.no_grad():
        for x, y, p in dl:
            x = x.to(device)
            s = model.anomaly_score(x, reduction=None).cpu().numpy().tolist()
            paths.append(p[0])
            scores += s
            labels.append(int(y.item()))

    # Optional ROC-AUC if labels available
    auc = None
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        pass

    # Save TSV
    out_tsv = Path("runs")/"inference_scores.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("path	score	label\n")
        for p, s, y in zip(paths, scores, labels):
            f.write(f"{p}\t{s}\t{y}\n")
    print(f"Saved scores: {out_tsv}")
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
