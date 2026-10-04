"""
Run a trained Stage 1 generator on a full page of real handwritten notes.

Pipeline:
    page image -> line segmentation (horizontal projection profiling)
               -> per-line generator inference
               -> paste each result back at its ORIGINAL coordinates
               -> saved "font-style" page image, same layout as the input

Note: this requires a trained checkpoint (see train.py). Running it on a
randomly-initialized generator will produce noise, not useful output --
it's fine for a shape/pipeline sanity check, but not for real results.

This is a simple projection-profile line segmenter, good enough for
handwriting on a plain or lined background with reasonably separated
lines. Swap in a learned line-detector later if your notes have tight or
overlapping lines.

Usage:
    python inference.py --checkpoint path/to/generator_epochN.pt \
                         --image notes.jpg --out out.png
"""
import argparse

import numpy as np
import torch
from PIL import Image

from models import UNetGenerator


def segment_lines(page_gray, min_gap=6, min_line_height=10):
    """
    Returns a list of (y_start, y_end) row ranges, one per detected text
    line, found via horizontal projection profiling: rows containing dark
    (ink) pixels are "text rows"; a run of enough consecutive blank rows
    ends a line.
    """
    arr = np.array(page_gray)
    row_ink = (arr < 200).sum(axis=1)  # count dark pixels per row
    is_text_row = row_ink > 0

    lines, in_line, start, gap = [], False, 0, 0
    for y, has_text in enumerate(is_text_row):
        if has_text:
            if not in_line:
                start, in_line = y, True
            gap = 0
        else:
            if in_line:
                gap += 1
                if gap > min_gap:
                    end = y - gap
                    if end - start >= min_line_height:
                        lines.append((start, end))
                    in_line = False
    if in_line:
        lines.append((start, len(is_text_row)))
    return lines


def run_generator_on_crop(generator, crop_img, device, target_height=64):
    """Resize crop to the model's training height, run inference, resize back."""
    orig_w, orig_h = crop_img.size
    new_w = max(1, round(orig_w * target_height / orig_h))
    new_w = ((new_w + 15) // 16) * 16  # generator needs width divisible by 16

    resized = crop_img.resize((new_w, target_height), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(resized)).float() / 255.0
    tensor = (tensor * 2 - 1).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        output = generator(tensor)

    output = (output.clamp(-1, 1) + 1) / 2  # -> [0, 1]
    output_img = Image.fromarray((output.squeeze().cpu().numpy() * 255).astype(np.uint8))

    # Resize back to this line's ORIGINAL dimensions -- the hard constraint
    # that keeps spacing identical to the input page, not the network's choice.
    return output_img.resize((orig_w, orig_h), Image.BILINEAR)


def process_page(checkpoint_path, image_path, out_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    generator = UNetGenerator(in_ch=1, out_ch=1).to(device)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    page = Image.open(image_path).convert("L")
    lines = segment_lines(page)
    print(f"Detected {len(lines)} lines")

    canvas = Image.new("L", page.size, color=255)
    page_w, _ = page.size

    for (y0, y1) in lines:
        crop = page.crop((0, y0, page_w, y1))
        translated = run_generator_on_crop(generator, crop, device)
        canvas.paste(translated, (0, y0))

    canvas.save(out_path)
    print(f"Saved font-style page to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default="output.png")
    args = parser.parse_args()
    process_page(args.checkpoint, args.image, args.out)
