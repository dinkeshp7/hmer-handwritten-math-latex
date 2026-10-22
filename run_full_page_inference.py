"""
Full-Page Answer Script to LaTeX Inference Runner.

Executes end-to-end full page conversion:
1. Binarization & Deskewing
2. Page Layout & Line Segmentation (1D Vertical IoU NMS)
3. Offline TrOCR for English text lines
4. MatrixGridParser for 2D matrices
5. Trained HMER Neural Model for mathematical equations
6. Clean LaTeX Document Generation with multi-line align* formatting
"""

import os
import sys
import ctypes
import argparse
from pathlib import Path

# Fix for CentOS 7 / RedHat HPC nodes: Preload Anaconda's modern libstdc++ to resolve CXXABI_1.3.9
for candidate in [
    "/opt/ohpc/pub/apps/Anaconda3-2023.03/lib/libstdc++.so.6",
    "/opt/ohpc/pub/apps/Anaconda3-2023.03/lib/libstdc++.so",
    os.path.expanduser("~/miniconda3/lib/libstdc++.so.6"),
    os.path.expanduser("~/anaconda3/lib/libstdc++.so.6"),
]:
    if os.path.exists(candidate):
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            break
        except Exception:
            pass

import cv2
import numpy as np

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.api.full_page_pipeline import FullPageLaTeXPipeline


def main():
    parser = argparse.ArgumentParser(description="Full-Page Answer Script to LaTeX Neural Converter")
    parser.add_argument("--image_path", type=str, default="page-210103096-01.jpg", help="Path to input answer script image")
    parser.add_argument("--checkpoint", type=str, default="/scratch/p.dinkesh/hmer_project/checkpoints/hmer_checkpoint_epoch_010.pth", help="Trained HMER checkpoint path")
    parser.add_argument("--roll_number", type=str, default="210103096", help="Student roll number")
    parser.add_argument("--output_tex", type=str, default="output_solution.tex", help="Path to save output LaTeX file")
    parser.add_argument("--engine", type=str, default="hybrid", choices=["hybrid", "nougat", "qwen_vl"], help="OCR Engine: 'hybrid' (Dual-Head TrOCR+pix2tex), 'nougat' (Meta Nougat), 'qwen_vl' (Qwen2.5-VL SOTA)")
    parser.add_argument("--model_name", type=str, default=None, help="Specific HuggingFace model repo or path (e.g. Qwen/Qwen2.5-VL-7B-Instruct or facebook/nougat-small)")
    args = parser.parse_args()

    print("======================================================================")
    print("  TRAINED HMER NEURAL MODEL FULL-PAGE LATEX CONVERTER OUTPUT")
    print("======================================================================")
    print(f"Filename            : {os.path.basename(args.image_path)}")
    print(f"Student Roll Number : {args.roll_number}")
    print(f"Engine Selected     : {args.engine}")
    if args.model_name:
        print(f"Model Specified     : {args.model_name}")

    # Check for image file
    img = None
    if os.path.exists(args.image_path):
        img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        # Search common paths
        candidates = [
            f"/scratch/p.dinkesh/hmer_project/data/ma102/{args.image_path}",
            f"/scratch/p.dinkesh/hmer_project/{args.image_path}",
            f"./data/{args.image_path}",
            f"./{args.image_path}"
        ]
        for c in candidates:
            if os.path.exists(c):
                img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    print(f"Loaded image from candidate path: {c}")
                    break

    if img is None:
        raise FileNotFoundError(f"Input image not found on disk: '{args.image_path}'. Checked local and /scratch paths.")

    # Initialize master pipeline
    ckpt = args.checkpoint if os.path.exists(args.checkpoint) else None
    pipeline = FullPageLaTeXPipeline(
        checkpoint_path=ckpt,
        engine=args.engine,
        model_name=args.model_name
    )

    # Process page
    result = pipeline.process_full_page(img, roll_number=args.roll_number)

    print(f"Segmented Regions   : {result['num_regions_segmented']}")
    print(f"Processed Lines     : {result['num_lines_processed']}")
    print()
    print("--- GENERATED COMPLETE .TEX DOCUMENT OUTPUT ---")
    print(result["tex_document"])
    print("======================================================================")

    # Save to disk
    out_path = Path(args.output_tex)
    out_path.write_text(result["tex_document"], encoding="utf-8")
    print(f"Saved complete LaTeX code to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
