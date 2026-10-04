"""
Visualize detected line boxes on a real handwriting photo -- no trained
model needed. Run this first on your own notes to confirm segmentation
is finding lines correctly before you invest in training Stage 1 itself.

Usage:
    python check_segmentation.py --image my_notes.jpg --out boxes.png
"""
import argparse

from PIL import Image, ImageDraw

from inference import segment_lines


def visualize(image_path, out_path, min_gap, min_line_height):
    page = Image.open(image_path).convert("L")
    lines = segment_lines(page, min_gap=min_gap, min_line_height=min_line_height)
    print(f"Detected {len(lines)} lines")

    preview = page.convert("RGB")
    draw = ImageDraw.Draw(preview)
    for (y0, y1) in lines:
        draw.rectangle([0, y0, page.width - 1, y1], outline=(220, 40, 40), width=2)

    preview.save(out_path)
    print(f"Saved preview with {len(lines)} line boxes to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default="boxes.png")
    parser.add_argument("--min_gap", type=int, default=6,
                         help="Blank rows needed to end a line. Increase if lines merge together.")
    parser.add_argument("--min_line_height", type=int, default=10,
                         help="Discard detected regions shorter than this (noise filtering).")
    args = parser.parse_args()
    visualize(args.image, args.out, args.min_gap, args.min_line_height)
