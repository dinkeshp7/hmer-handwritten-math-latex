"""
Download & Offline Model Caching Script for Nougat and Qwen-VL Models.

Downloads weights to local cache or /scratch/p.dinkesh/hmer_project/models/
for zero-internet execution on PARAM Kamrupa GPU nodes.
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


def download_nougat(output_dir: str = "models/nougat", model_name: str = "facebook/nougat-small"):
    print(f"=== Downloading Meta Nougat Model ({model_name}) ===")
    from transformers import NougatProcessor, VisionEncoderDecoderModel
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Target directory: {output_dir}")
    
    processor = NougatProcessor.from_pretrained(model_name)
    processor.save_pretrained(output_dir)
    print("[Nougat] Processor saved.")
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    print("[Nougat] Model weights saved.")
    print(f"[SUCCESS] Nougat offline model ready in: {output_dir}\n")


def download_qwen_vl(output_dir: str = "models/qwen_vl", model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    print(f"=== Downloading Qwen-VL Model ({model_name}) ===")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Target directory: {output_dir}")
    
    # Preferred method: snapshot_download directly from HuggingFace Hub
    # Does NOT require PyTorch or Torchvision to be instantiated in RAM on login node
    try:
        from huggingface_hub import snapshot_download
        print(f"[Qwen-VL] Using direct snapshot download (zero PyTorch RAM overhead)...")
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )
        print(f"[SUCCESS] Qwen-VL offline model ready in: {output_dir}\n")
        return
    except Exception as err:
        print(f"[Notice] Direct snapshot failed ({err}), falling back to AutoProcessor/Model...")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(output_dir)
    print("[Qwen-VL] Processor saved.")
    
    model_cls = None
    try:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    except Exception:
        try:
            from transformers import Qwen2VLForConditionalGeneration
            model_cls = Qwen2VLForConditionalGeneration
        except Exception:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq
            
    print(f"[Qwen-VL] Using model class: {model_cls.__name__}")
    model = model_cls.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    print("[Qwen-VL] Model weights saved.")
    print(f"[SUCCESS] Qwen-VL offline model ready in: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Nougat and Qwen-VL weights for offline HPC execution")
    parser.add_argument("--model", type=str, choices=["all", "nougat", "qwen_vl"], default="all")
    parser.add_argument("--nougat_model", type=str, default="facebook/nougat-small", help="HuggingFace model ID for Nougat")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace model ID for Qwen VL (e.g. Qwen/Qwen2-VL-2B-Instruct or Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--nougat_dir", type=str, default="/scratch/p.dinkesh/hmer_project/models/nougat")
    parser.add_argument("--qwen_dir", type=str, default="/scratch/p.dinkesh/hmer_project/models/qwen_vl")
    args = parser.parse_args()

    # Fallback to local ./models if /scratch doesn't exist
    nougat_path = args.nougat_dir if os.path.exists("/scratch") else "./models/nougat"
    qwen_path = args.qwen_dir if os.path.exists("/scratch") else "./models/qwen_vl"

    if args.model in ["all", "nougat"]:
        try:
            download_nougat(output_dir=nougat_path, model_name=args.nougat_model)
        except Exception as e:
            print(f"[Error downloading Nougat] {e}")

    if args.model in ["all", "qwen_vl"]:
        try:
            download_qwen_vl(output_dir=qwen_path, model_name=args.qwen_model)
        except Exception as e:
            print(f"[Error downloading Qwen-VL] {e}")
