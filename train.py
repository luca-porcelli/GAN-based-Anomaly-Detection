import argparse, os, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from src.dataset import ImageFolderFlat
from src.utils import seed_everything, save_json, load_json
from models.ganomaly import GANomaly


def parse_args():
    p = argparse.ArgumentParser(description="Train a GANomaly-style anomaly detector")
    p.add_argument("--data_root", type=str, default="data/mvtec")
    p.add_argument("--category", type=str, default="bottle")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--lambda_img", type=float, default=None)
    p.add_argument("--lambda_latent", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Load defaults from config if exists
    cfg = load_json("configs.json", {})
    img_size = args.img_size or cfg.get("img_size", 256)
    batch_size = args.batch_size or cfg.get("batch_size", 16)
    epochs = args.epochs or cfg.get("epochs", 50)
    lr = args.lr or cfg.get("lr", 2e-4)
    lambda_img = args.lambda_img or cfg.get("lambda_img", 50.0)
    lambda_latent = args.lambda_latent or cfg.get("lambda_latent", 1.0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Data
    train_dir = Path(args.data_root) / args.category / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")
    ds = ImageFolderFlat(str(train_dir), img_size=img_size, return_label=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Model
    model = GANomaly(in_c=3, z_dim=args.z_dim,
                     lambda_img=lambda_img, lambda_latent=lambda_latent).to(device)
    optG = Adam(list(model.E.parameters()) +
                list(model.DEC.parameters()) +
                list(model.E2.parameters()), lr=lr, betas=(0.5, 0.999))
    optD = Adam(model.D.parameters(), lr=lr, betas=(0.5, 0.999))
    adv_criterion = BCEWithLogitsLoss()

    # Run name
    run_name = args.run_name or f"{args.category}_{int(time.time())}"
    out_dir = Path("runs") / run_name
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}")
        loss_epoch = 0.0

        for batch in pbar:
            x, _ = batch if isinstance(batch, (list, tuple)) and len(batch) == 2 else (batch[0], None)
            x = x.to(device)

            # === Train Discriminator ===
            optD.zero_grad()
            with torch.no_grad():
                x_rec, _, _ = model.generator(x)
            d_real = model.D(x)
            d_fake = model.D(x_rec)
            l_adv_real = adv_criterion(d_real, torch.ones_like(d_real))
            l_adv_fake = adv_criterion(d_fake, torch.zeros_like(d_fake))
            d_loss = 0.5 * (l_adv_real + l_adv_fake)
            d_loss.backward()
            optD.step()

            # === Train Generator ===
            optG.zero_grad()
            x_rec, z, z_rec = model.generator(x)
            d_fake_for_g = model.D(x_rec)
            l_g_adv = adv_criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))
            l_img = torch.mean(torch.abs(x - x_rec))
            l_latent = torch.mean(torch.abs(z - z_rec))
            g_loss = l_g_adv + model.lambda_img * l_img + model.lambda_latent * l_latent
            g_loss.backward()
            optG.step()

            loss_epoch += g_loss.item()
            pbar.set_postfix(
                g_loss=g_loss.item(),
                d_loss=d_loss.item(),
                l_img=l_img.item(),
                l_latent=l_latent.item()
            )

        # Epoch summary
        loss_epoch /= len(dl)
        history.append({"epoch": epoch, "g_loss": loss_epoch})

        # Save best (based on generator loss)
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            torch.save(model.state_dict(), out_dir / "weights" / "best.pt")

        # Always save last
        torch.save(model.state_dict(), out_dir / "weights" / "last.pt")

    # Save run config
    run_cfg = {
        "category": args.category,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "lambda_img": lambda_img,
        "lambda_latent": lambda_latent,
        "z_dim": args.z_dim,
        "device": device,
        "best_g_loss": best_loss
    }
    save_json(out_dir / "run_config.json", run_cfg)
    print(f"Training finished. Best weights saved to: {out_dir/'weights'/'best.pt'}")


if __name__ == "__main__":
    main()
