"""
Stage 2 training loop: font-page image -> LaTeX.

Usage:
    python train.py --data_root /path/to/img2latex_data --vocab vocab.json --epochs 50
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import Img2LatexModel
from dataset import Img2LatexDataset, collate_fn
from tokenizer import LatexTokenizer


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = LatexTokenizer.load(args.vocab)
    pad_idx = tokenizer.vocab["<pad>"]

    dataset = Img2LatexDataset(args.data_root, tokenizer, patch_size=args.patch_size,
                                use_boxes=args.use_boxes)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=pad_idx),
        num_workers=args.num_workers, drop_last=True,
    )

    model_config = dict(
        vocab_size=len(tokenizer), embed_dim=args.embed_dim, patch_size=args.patch_size,
        enc_depth=args.enc_depth, dec_depth=args.dec_depth, pad_idx=pad_idx,
    )
    model = Img2LatexModel(**model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for images, targets, boxes, box_mask in loader:
            images, targets = images.to(device), targets.to(device)
            boxes = boxes.to(device) if boxes is not None else None
            box_mask = box_mask.to(device) if box_mask is not None else None

            # Teacher forcing: predict token t+1 from tokens up to t
            decoder_input = targets[:, :-1]
            decoder_target = targets[:, 1:]

            logits = model(images, decoder_input, boxes, box_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), decoder_target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} | loss {loss.item():.4f}")
            step += 1

        torch.save(
            {"model_state_dict": model.state_dict(), "config": model_config},
            out_dir / f"model_epoch{epoch}.pt",
        )

    print("Training complete. Checkpoints saved to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./stage2_runs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--enc_depth", type=int, default=6)
    parser.add_argument("--dec_depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--use_boxes", action="store_true",
                         help="Condition the encoder on line-box geometry from Stage 1 segmentation")
    args = parser.parse_args()
    train(args)
