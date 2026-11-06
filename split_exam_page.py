"""
Exam Page Multi-Band Slicer Utility.
Splits full-page exam scans into overlapping horizontal strips (e.g. 3 parts)
to allow 100% full-resolution, zero-downsampling VLM inference without GPU VRAM Out-of-Memory (OOM).

Usage:
    python split_exam_page.py --image data/ma102/page-210103096-01.jpg --parts 3
    python split_exam_page.py --batch_dir data/ma102 --pattern "page-210103096-*.jpg" --parts 3
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image


def split_image_into_bands(
    image_path: str,
    output_dir: str = None,
    num_parts: int = 3,
    overlap_ratio: float = 0.08,
    crop_header: bool = False
) -> list:
    """
    Splits an image vertically into overlapping horizontal bands.
    Saves each band in output_dir (or same directory as image) at 100% native quality.
    """
    img_path = Path(image_path)
    # Check candidate paths if direct path does not exist
    if not img_path.exists():
        for candidate in [
            f"data/ma102/{image_path}",
            f"data/{image_path}",
            f"/scratch/p.dinkesh/hmer_project/data/ma102/{image_path}",
            f"/scratch/p.dinkesh/hmer_project/{image_path}"
        ]:
            if Path(candidate).exists():
                img_path = Path(candidate)
                break
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    print(f"\n[Slicer] Processing: {img_path.name}")
    print(f"  Original Resolution: {w} x {h} px ({(w * h) / 1e6:.2f} Megapixels)")

    # Auto-detect and crop printed booklet instruction header if present (zooms directly on handwriting)
    try:
        from src.ocr.qwen_vl_ocr import QwenVLOCREngine
        im = QwenVLOCREngine.crop_booklet_instructions(im)
        w, h = im.size
        print(f"  Active Handwriting Canvas : {w} x {h} px")
    except Exception:
        pass

    out_folder = Path(output_dir) if output_dir else img_path.parent
    out_folder.mkdir(parents=True, exist_ok=True)

    band_h = h / float(num_parts)
    overlap_px = int(band_h * overlap_ratio)

    created_files = []
    base_stem = img_path.stem

    for i in range(num_parts):
        y_start = max(0, int(i * band_h) - (overlap_px if i > 0 else 0))
        y_end = min(h, int((i + 1) * band_h) + (overlap_px if i < num_parts - 1 else 0))

        band = im.crop((0, y_start, w, y_end))
        out_name = f"{base_stem}_part{i + 1}.jpg"
        out_path = out_folder / out_name

        # Save at high JPEG quality without compression degradation
        band.save(out_path, quality=95, subsampling=0)
        bw, bh = band.size
        patches = (bw * bh) / (28 * 28)
        print(f"  -> Saved Part {i + 1}/{num_parts}: {out_name} | {bw}x{bh} px (~{int(patches)} visual tokens)")
        created_files.append(str(out_path))

    return created_files


def main():
    parser = argparse.ArgumentParser(description="Split full-page exam scripts into high-resolution overlapping horizontal bands.")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--batch_dir", type=str, default=None, help="Directory containing multiple images to split")
    parser.add_argument("--pattern", type=str, default="page-*.jpg", help="Glob pattern for batch directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for saved parts (defaults to image directory)")
    parser.add_argument("--parts", type=int, default=3, help="Number of horizontal bands to create (default: 3)")
    parser.add_argument("--overlap", type=float, default=0.08, help="Fractional vertical overlap between adjacent bands (default: 0.08)")
    parser.add_argument("--crop_header", action="store_true", help="Crop printed booklet header before splitting")
    args = parser.parse_args()

    if args.image:
        split_image_into_bands(
            args.image,
            output_dir=args.output_dir,
            num_parts=args.parts,
            overlap_ratio=args.overlap,
            crop_header=args.crop_header
        )
    elif args.batch_dir:
        bdir = Path(args.batch_dir)
        images = sorted(list(bdir.glob(args.pattern)))
        print(f"[Slicer Batch] Found {len(images)} images matching '{args.pattern}' in {bdir}")
        for img in images:
            # Skip files that are already parts
            if "_part" in img.stem:
                continue
            split_image_into_bands(
                str(img),
                output_dir=args.output_dir,
                num_parts=args.parts,
                overlap_ratio=args.overlap,
                crop_header=args.crop_header
            )
    else:
        print("[Error] Please specify either --image <path> or --batch_dir <dir>.")
        sys.exit(1)


if __name__ == "__main__":
    main()
