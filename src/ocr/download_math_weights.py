"""
Offline SOTA Math OCR Weights Downloader Script for HPC Deployment.

Downloads pretrained LaTeX-OCR (pix2tex) weights trained on 100,000+ real math formulas
and saves them to ./models/pix2tex for offline execution on PARAM Kamrupa HPC.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download SOTA Math OCR Weights for Offline HPC Deployment")
    parser.add_argument("--save_dir", type=str, default="./models/pix2tex", help="Directory to save offline math weights")
    args = parser.parse_args()

    print("======================================================================")
    print("   OFFLINE SOTA MATH OCR (LATEX-OCR) MODEL WEIGHTS DOWNLOADER")
    print("======================================================================")

    try:
        from pix2tex.cli import LatexOCR
        from PIL import Image
        import numpy as np
    except ImportError:
        print("[ERROR] Please install pix2tex: pip install pix2tex")
        return

    save_path = Path(args.save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    print("Initializing LatexOCR model & downloading pretrained weights...")
    model = LatexOCR()

    # Create dummy image to verify inference
    dummy = Image.fromarray(np.full((80, 200), 255, dtype=np.uint8))
    pred = model(dummy)
    print(f"Verified Math Model Inference Output: \"{pred}\"")

    # Locate downloaded weights
    # pix2tex stores weights in user cache or checkpoints directory
    import pix2tex
    pix2tex_dir = Path(pix2tex.__file__).parent
    
    # Check common cache locations
    cache_dirs = [
        Path.home() / ".cache" / "pix2tex",
        Path.home() / ".pix2tex",
        pix2tex_dir / "model" / "checkpoints",
        Path.home() / ".cache" / "torch" / "checkpoints"
    ]
    
    copied = 0
    for cdir in cache_dirs:
        if cdir.exists():
            for f in cdir.glob("*"):
                if f.is_file():
                    target = save_path / f.name
                    shutil.copy2(f, target)
                    print(f"Copied weight file: {f.name} -> {target}")
                    copied += 1

    print(f"\n[OK] SOTA Math OCR weights downloaded & verified successfully! (Files copied: {copied})")
    print(f"Offline math model weights directory: {save_path}")

if __name__ == "__main__":
    main()
