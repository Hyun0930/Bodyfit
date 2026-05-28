import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data import BodyFitDataset
from src.models.cvae import CVAE

EXERCISES = ["squat", "bench", "deadlift", "ohp"]


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, device):
    model.train()
    total, recon_sum, kl_sum = 0.0, 0.0, 0.0
    for pose, body in loader:
        pose, body = pose.to(device), body.to(device)
        pose_recon, mu, log_var = model(pose, body)
        loss, recon, kl = CVAE.compute_loss(pose, pose_recon, mu, log_var, model.beta)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return total / n, recon_sum / n, kl_sum / n


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total, recon_sum, kl_sum = 0.0, 0.0, 0.0
    for pose, body in loader:
        pose, body = pose.to(device), body.to(device)
        pose_recon, mu, log_var = model(pose, body)
        loss, recon, kl = CVAE.compute_loss(pose, pose_recon, mu, log_var, model.beta)
        total += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return total / n, recon_sum / n, kl_sum / n


def save_curve(train_losses, val_losses, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", nargs="+", default=EXERCISES)
    parser.add_argument("--data_root", default="data/processed")
    parser.add_argument("--ckpt_dir", default="checkpoints/cvae")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    dataset = BodyFitDataset(exercises=args.exercise, root=Path(args.data_root))
    train_set, val_set = dataset.split(train_ratio=0.9, seed=42)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    model = CVAE(latent_dim=args.latent_dim, beta=args.beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_recon, tr_kl = train_epoch(model, train_loader, optimizer, device)
        vl_loss, vl_recon, vl_kl = val_epoch(model, val_loader, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {tr_loss:.4f} (recon {tr_recon:.4f} kl {tr_kl:.4f}) | "
            f"val {vl_loss:.4f} (recon {vl_recon:.4f} kl {vl_kl:.4f})"
        )

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": vl_loss,
                    "config": vars(args),
                },
                ckpt_dir / "best.pt",
            )
            print(f"  → saved best checkpoint (val_loss={best_val:.4f})")

    save_curve(train_losses, val_losses, Path("results/cvae_train_curve.png"))
    print("Done. Curve saved to results/cvae_train_curve.png")


if __name__ == "__main__":
    main()
