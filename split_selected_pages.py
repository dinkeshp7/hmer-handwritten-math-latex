"""
Automated Multi-Band Slicer for Selected Student Pages:
- page-240101079-01.jpg (Roll 240101079, Page 01)
- page-240122044-14.jpg (Roll 240122044, Page 14)

Slices each page into 3 overlapping high-resolution horizontal bands
at 100% native scanner resolution (zero downsampling) for zero-OOM inference on Tesla V100 (16GB).
"""

import os
import sys
from pathlib import Path
from PIL import Image

TARGETS = [
    "page-240101079-01.jpg",
    "page-240122044-14.jpg"
]

def find_image(filename):
    candidates = [
        Path(filename),
        Path(f"data/ma102/{filename}"),
        Path(f"data/{filename}"),
        Path(f"/scratch/p.dinkesh/hmer_project/data/ma102/{filename}"),
        Path(f"/scratch/p.dinkesh/hmer_project/{filename}"),
        Path(f"prof_demo/{filename}"),
    ]
    for c in candidates:
        if c.exists():
            return c
            
    # Recursive search
    stem = Path(filename).stem
    for root in ["data", "/scratch/p.dinkesh/hmer_project/data", "prof_demo", "."]:
        p = Path(root)
        if p.exists():
            matches = list(p.glob(f"**/{filename}"))
            if not matches:
                matches = list(p.glob(f"**/{stem}.*"))
            if matches:
                return matches[0]
                
    return None

def slice_page(img_path, num_parts=3, overlap_ratio=0.08):
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    print(f"\n==================================================")
    print(f"Processing Selected Script: {img_path.name}")
    print(f"Original Resolution : {w} x {h} px ({(w * h) / 1e6:.2f} Megapixels)")

    # Smart header detection: crop printed booklet instruction header if this is Page 01
    is_page_01 = "-01" in img_path.stem
    if is_page_01 and h > 1500:
        try:
            from src.ocr.qwen_vl_ocr import QwenVLOCREngine
            im = QwenVLOCREngine.crop_booklet_instructions(im)
            w, h = im.size
            print(f"Booklet Header Cropped : New Active Canvas {w} x {h} px")
        except Exception:
            # Fallback crop top ~18%
            crop_top = int(h * 0.18)
            im = im.crop((0, crop_top, w, h))
            w, h = im.size
            print(f"Booklet Header Cropped (18%) : New Active Canvas {w} x {h} px")
    elif h > 1500:
        # Interior page (e.g. Page 14): only crop very top margin if "Question only" bar exists (~8%)
        top_crop = int(h * 0.06)
        im = im.crop((0, top_crop, w, h))
        w, h = im.size
        print(f"Top Margin Cleared : New Active Canvas {w} x {h} px")

    out_folder = img_path.parent
    band_h = h / float(num_parts)
    overlap_px = int(band_h * overlap_ratio)
    
    created_parts = []
    for i in range(num_parts):
        y_start = max(0, int(i * band_h) - (overlap_px if i > 0 else 0))
        y_end = min(h, int((i + 1) * band_h) + (overlap_px if i < num_parts - 1 else 0))
        
        band = im.crop((0, y_start, w, y_end))
        out_name = f"{img_path.stem}_part{i + 1}.jpg"
        out_path = out_folder / out_name
        
        band.save(out_path, quality=95, subsampling=0)
        # Also maintain a copy in the current directory if saved in a subfolder
        local_copy = Path(out_name)
        if out_path.resolve() != local_copy.resolve():
            band.save(local_copy, quality=95, subsampling=0)
            
        bw, bh = band.size
        patches = (bw * bh) / (28 * 28)
        print(f"  [+] Part {i + 1}/{num_parts}: {out_name} ({bw}x{bh} px, ~{int(patches)} tokens) -> 100% Native Resolution")
        created_parts.append(out_path)
        
    return created_parts

def main():
    print("======================================================================")
    print("  AUTOMATED MULTI-BAND HIGH-RES SLICER FOR SELECTED MA102 EXAM SCRIPTS")
    print("======================================================================")
    targets = sys.argv[1:] if len(sys.argv) > 1 else TARGETS
    all_created = []
    
    for target in targets:
        found_path = find_image(target)
        if not found_path:
            print(f"[!] Warning: Target '{target}' not found on disk.")
            continue
            
        parts = slice_page(found_path, num_parts=3, overlap_ratio=0.08)
        all_created.extend(parts)

    print("\n======================================================================")
    print("  ALL SELECTED PAGES SLICED INTO 100% NATIVE RESOLUTION PARTS")
    print("======================================================================")
    print("Generated files:")
    for p in all_created:
        print(f"  {p}")

    print("\nReady to submit to SLURM!")

if __name__ == "__main__":
    main()
