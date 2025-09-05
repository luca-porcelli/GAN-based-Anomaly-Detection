import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from models.ganomaly import GANomaly
from src.utils import load_json

def residual_map(model, x):
    with torch.no_grad():
        x_rec, _, _ = model(x)
    # residual in [0,2] because inputs in [-1,1]
    res = torch.abs(x - x_rec) * 0.5  # scale to [0,1]
    res = res.mean(dim=1, keepdim=True)  # grayscale residual
    return res

def parse_args():
    p = argparse.ArgumentParser(description="Create residual heatmap for an image")
    p.add_argument("--image", type=str, default="data/mvtec/bottle/test/broken_large/000.png")
    p.add_argument("--weights", type=str, default="runs/bottle_1756463597/weights/best.pt")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--lambda_img", type=float, default=None)
    p.add_argument("--lambda_latent", type=float, default=None)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--out", type=str, default="residual.png")
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

    im = Image.open(args.image).convert("RGB").resize((img_size, img_size))
    x = torch.from_numpy(np.array(im).astype("float32")/255.0).permute(2,0,1).unsqueeze(0)
    x = x*2 - 1
    x = x.to(device)

    res = residual_map(model, x)[0,0].cpu().numpy()
    res = (res / (res.max() + 1e-8) * 255).astype("uint8")
    Image.fromarray(res).save(args.out)
    print(f"Saved residual heatmap to {args.out}")

if __name__ == "__main__":
    main()
