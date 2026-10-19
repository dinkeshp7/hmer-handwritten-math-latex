"""
Qwen2.5-VL / Qwen2-VL State-of-the-Art Vision-Language OCR & Mathematical Transcription Engine.

Provides zero-shot full-page multimodal reasoning and mathematical transcription.
Utilizes Dynamic Resolution Vision Transformer (ViT) with 2D Rotary Position Embeddings (2D-RoPE)
to transcribe cursive handwriting, linear algebra matrices, and multi-step derivations
directly into complete, compilable LaTeX code without fragile line-chopping.
Supports Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-3B-Instruct, and Qwen2-VL-2B-Instruct.
"""

import os
import re
from typing import Optional, Union, Dict, Any
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class QwenVLOCREngine:
    """
    Qwen-VL Vision-Language Mathematical Transcription Engine.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        local_model_dir: Optional[str] = "/scratch/p.dinkesh/hmer_project/models/qwen_vl",
        device: Optional[str] = None,
        load_in_4bit: bool = False
    ):
        # Default to environment variable or flagship Qwen2.5-VL model
        self.model_name = model_name or os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.local_model_dir = local_model_dir
        self.device = device or ("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")
        self.load_in_4bit = load_in_4bit
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Loads Qwen-VL model weights and multimodal processor."""
        if self.model is not None:
            return

        try:
            # Check local directories first for offline cluster execution
            target_path = None
            if self.local_model_dir and os.path.exists(self.local_model_dir):
                target_path = self.local_model_dir
                print(f"[Qwen-VL] Loading offline weights from: {target_path}")
            elif os.path.exists("./models/qwen_vl"):
                target_path = "./models/qwen_vl"
                print(f"[Qwen-VL] Loading offline weights from: {target_path}")
            else:
                target_path = self.model_name
                print(f"[Qwen-VL] Loading model weights for: {target_path}")

            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(target_path)

            # Auto-detect best model class
            model_cls = None
            try:
                from transformers import AutoModelForImageTextToText
                model_cls = AutoModelForImageTextToText
            except Exception:
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    model_cls = Qwen2_5_VLForConditionalGeneration
                except Exception:
                    from transformers import Qwen2VLForConditionalGeneration
                    model_cls = Qwen2VLForConditionalGeneration

            kwargs: Dict[str, Any] = {}
            if self.device == "cuda":
                # Use bfloat16 if supported, else float16
                if torch.cuda.is_bf16_supported():
                    kwargs["torch_dtype"] = torch.bfloat16
                else:
                    kwargs["torch_dtype"] = torch.float16
                if self.load_in_4bit:
                    kwargs["load_in_4bit"] = True
            
            self.model = model_cls.from_pretrained(target_path, **kwargs)
            if not self.load_in_4bit and self.device:
                self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[Qwen-VL] Initialized successfully using {model_cls.__name__} on device: {self.device}")
        except Exception as err:
            print(f"[Qwen-VL Warning] Model loading failed ({err}). Falling back.")
            self.model = None
            self.processor = None

    def recognize_page(self, image_input: Union[np.ndarray, str, Image.Image], roll_number: Optional[str] = None) -> str:
        """
        Transcribes a full-page answer script scan directly into complete LaTeX code.
        """
        self.load_model()
        if self.model is None or self.processor is None:
            return ""

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            return ""

        prompt = (
            "You are an expert mathematical document transcription assistant. "
            "Please transcribe the handwritten mathematics examination solution in this image into complete, compilable LaTeX code. "
            "Follow these strict formatting guidelines:\n"
            "1. Transcribe all mathematical equations into standard LaTeX math environments (e.g. \\[ ... \\] or \\begin{align*} ... \\end{align*}).\n"
            "2. Convert any matrices into \\begin{bmatrix} ... \\end{bmatrix}.\n"
            "3. Accurately preserve all English mathematical reasoning, theorem citations, and explanatory prose.\n"
            "4. Ignore any crossed-out scribbles or scratch-outs.\n"
            "5. Return the solution wrapped in standard article document structure with amsmath, amssymb, and geometry packages."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        output_text = ""
        try:
            from qwen_vl_utils import process_vision_info
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
        except Exception:
            # Fallback using standard processor pipeline if qwen_vl_utils is not available
            try:
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[text_prompt],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
            except Exception as err:
                print(f"[Qwen-VL Error] Transcription failed: {err}")
                return ""

        # Post-processing: extract clean LaTeX code
        cleaned_text = output_text.strip()
        
        # Remove markdown fences if wrapped in ```latex ... ``` or ```tex ... ```
        if "```latex" in cleaned_text:
            match = re.search(r"```latex\s*(.*?)\s*```", cleaned_text, re.DOTALL)
            if match:
                cleaned_text = match.group(1).strip()
        elif "```tex" in cleaned_text:
            match = re.search(r"```tex\s*(.*?)\s*```", cleaned_text, re.DOTALL)
            if match:
                cleaned_text = match.group(1).strip()
        elif "```" in cleaned_text:
            match = re.search(r"```\s*(.*?)\s*```", cleaned_text, re.DOTALL)
            if match:
                cleaned_text = match.group(1).strip()

        # If document structure was not included by the model, wrap in standard document preamble
        if "\\documentclass" not in cleaned_text:
            roll = roll_number or "Unknown"
            cleaned_text = (
                "\\documentclass[12pt]{article}\n"
                "\\usepackage[utf8]{inputenc}\n"
                "\\usepackage{amsmath, amssymb, amsfonts}\n"
                "\\usepackage{geometry}\n"
                "\\geometry{a4paper, margin=1in}\n"
                "\\title{MA102 Answer Script Solution}\n"
                f"\\author{{Roll Number: {roll}}}\n"
                "\\date{\\today}\n"
                "\\begin{document}\n"
                "\\maketitle\n"
                "\\section*{Solution}\n\n"
                f"{cleaned_text}\n\n"
                "\\end{document}\n"
            )

        return cleaned_text


if __name__ == "__main__":
    print("=== Testing Qwen-VL OCR Engine Definition ===")
    engine = QwenVLOCREngine()
    print(f"[OK] QwenVLOCREngine successfully instantiated for model: {engine.model_name}")
