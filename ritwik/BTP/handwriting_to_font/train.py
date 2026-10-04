"""
Stage 1 training loop: paired handwriting -> font line-crop translation.

Usage:
    python train.py --data_root /path/to/paired_lines --epochs 100
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import UNetGenerator, PatchDiscriminator
from losses import Stage1Loss
from dataset import PairedLineDataset, collate_pad


def masked(tensor, mask):
    """Zero out padded columns so they never contribute to loss."""
    return tensor * mask


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PairedLineDataset(args.data_root, height=args.height)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pad, num_workers=args.num_workers, drop_last=True,
    )

    G = UNetGenerator(in_ch=1, out_ch=1).to(device)
    D = PatchDiscriminator(in_ch=2).to(device)
    criterion = Stage1Loss(ocr_model=None).to(device)  # plug in an OCR model once you have one

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    out_dir = Path(args.output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for handwriting, font, mask in loader:
            handwriting, font, mask = handwriting.to(device), font.to(device), mask.to(device)

            # ---- discriminator step ----
            with torch.no_grad():
                fake_font = G(handwriting)
            fake_font_m = masked(fake_font, mask)
            font_m = masked(font, mask)

            d_real = D(handwriting, font_m)
            d_fake = D(handwriting, fake_font_m.detach())
            d_loss = criterion.discriminator_loss(d_real, d_fake)

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ---- generator step ----
            fake_font = G(handwriting)
            fake_font_m = masked(fake_font, mask)
            d_fake_for_g = D(handwriting, fake_font_m)

            g_loss, parts = criterion.generator_loss(fake_font_m, font_m, d_fake_for_g)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} | D {d_loss.item():.4f} | G {g_loss.item():.4f} | {parts}")

            if step % args.sample_every == 0:
                sample = torch.cat([handwriting[:4], fake_font_m[:4], font_m[:4]], dim=0)
                save_image(sample, out_dir / "samples" / f"step_{step:07d}.png",
                           nrow=4, normalize=True, value_range=(-1, 1))

            step += 1

        torch.save(G.state_dict(), out_dir / "checkpoints" / f"generator_epoch{epoch}.pt")
        torch.save(D.state_dict(), out_dir / "checkpoints" / f"discriminator_epoch{epoch}.pt")

    print("Training complete. Checkpoints saved to", out_dir / "checkpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                         help="Directory containing handwriting/ and font/ paired subfolders")
    parser.add_argument("--output_dir", type=str, default="./stage1_runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sample_every", type=int, default=200)
    args = parser.parse_args()
    train(args)
