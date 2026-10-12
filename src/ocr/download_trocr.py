"""
Offline TrOCR Model Downloader Script for HPC Deployment.

Downloads pretrained TrOCR weights (microsoft/trocr-base-handwritten) from HuggingFace Hub
and saves them to disk so HPC compute nodes without internet access can load them locally.
"""

import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download TrOCR Model Weights for Offline HPC Deployment")
    parser.add_argument("--model_name", type=str, default="microsoft/trocr-base-handwritten", help="HuggingFace model identifier")
    parser.add_argument("--save_dir", type=str, default="./models/trocr", help="Local directory to save offline model weights")
    args = parser.parse_args()

    print("======================================================================")
    print(f"   OFFLINE TrOCR MODEL DOWNLOADER ({args.model_name})")
    print("======================================================================")

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        print("[ERROR] Please install transformers: pip install transformers")
        return

    save_path = Path(args.save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Fetching TrOCR Processor and Model from HuggingFace...")
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    print(f"Saving offline model files to: {save_path}")
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"[OK] Successfully saved offline TrOCR weights to {save_path}")
    print("You can now transfer this folder to PARAM Kamrupa HPC under /scratch/p.dinkesh/hmer_project/models/trocr/")

if __name__ == "__main__":
    main()
