"""
Run a trained Stage 2 model on a font-style page image to produce LaTeX.

Usage:
    python inference.py --checkpoint model_epoch40.pt --vocab vocab.json --image font_page.png
"""
import argparse
import re
from collections import Counter

import torch
from PIL import Image
from torchvision import transforms

from models import Img2LatexModel
from tokenizer import LatexTokenizer


def load_image(path, patch_size, device):
    img = Image.open(path).convert("L")
    w, h = img.size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    padded = Image.new("L", (new_w, new_h), color=255)
    padded.paste(img, (0, 0))
    tensor = transforms.ToTensor()(padded) * 2 - 1
    return tensor.unsqueeze(0).to(device)


@torch.no_grad()
def beam_search(model, image, tokenizer, device, beam_width=5, max_len=512, length_penalty=0.7):
    bos, eos = tokenizer.vocab["<bos>"], tokenizer.vocab["<eos>"]
    memory, memory_pad_mask = model.encoder(image)

    beams = [(torch.tensor([[bos]], device=device), 0.0, False)]  # (sequence, log_prob, finished)

    for _ in range(max_len):
        candidates = []
        for seq, log_prob, finished in beams:
            if finished:
                candidates.append((seq, log_prob, True))
                continue
            logits = model.decoder(seq, memory, memory_pad_mask)
            next_log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            top_log_probs, top_ids = next_log_probs.topk(beam_width)

            for lp, tok_id in zip(top_log_probs, top_ids):
                new_seq = torch.cat([seq, tok_id.view(1, 1)], dim=1)
                candidates.append((new_seq, log_prob + lp.item(), tok_id.item() == eos))

        candidates.sort(key=lambda c: c[1] / (c[0].shape[1] ** length_penalty), reverse=True)
        beams = candidates[:beam_width]

        if all(finished for _, _, finished in beams):
            break

    best_seq = beams[0][0][0].tolist()
    return tokenizer.decode(best_seq)


def repair_syntax(latex_str):
    """
    Minimal mechanical repair pass: balances braces and \\begin{}/\\end{}
    pairs. Catches the common decoder failure mode of an unmatched
    environment -- cheaper to fix here than to solve purely through
    training.
    """
    open_braces = latex_str.count("{") - latex_str.count("}")
    if open_braces > 0:
        latex_str += "}" * open_braces

    begins = re.findall(r"\\begin\{([a-zA-Z*]+)\}", latex_str)
    ends = re.findall(r"\\end\{([a-zA-Z*]+)\}", latex_str)
    missing = Counter(begins) - Counter(ends)
    for env, count in missing.items():
        latex_str += f"\\end{{{env}}}" * count

    return latex_str


def run(checkpoint_path, vocab_path, image_path, beam_width=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LatexTokenizer.load(vocab_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Img2LatexModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    patch_size = checkpoint["config"]["patch_size"]
    image = load_image(image_path, patch_size, device)
    raw_latex = beam_search(model, image, tokenizer, device, beam_width=beam_width)
    return repair_syntax(raw_latex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--beam_width", type=int, default=5)
    args = parser.parse_args()

    latex = run(args.checkpoint, args.vocab, args.image, args.beam_width)
    print(latex)
