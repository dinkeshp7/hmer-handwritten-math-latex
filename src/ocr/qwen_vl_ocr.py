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
            # Dynamic resolution bounds calibrated for Tesla V100 (16GB VRAM)
            min_pixels = 256 * 28 * 28
            max_pixels = 1024 * 28 * 28
            try:
                self.processor = AutoProcessor.from_pretrained(
                    target_path,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels
                )
            except Exception:
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

            if model_cls is None:
                raise ImportError(
                    "None of AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration, "
                    "or Qwen2VLForConditionalGeneration could be imported from transformers. "
                    "Please install transformers==4.46.3."
                )

            kwargs: Dict[str, Any] = {}
            if self.device == "cuda":
                # Use bfloat16 if supported, else float16
                if torch.cuda.is_bf16_supported():
                    kwargs["torch_dtype"] = torch.bfloat16
                else:
                    kwargs["torch_dtype"] = torch.float16
                if self.load_in_4bit:
                    kwargs["load_in_4bit"] = True
                
                # Enable memory-efficient SDPA (Cutlass kernel) on Tesla V100 (sm_70)
                try:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                except Exception:
                    pass

            try:
                kwargs_sdpa = dict(kwargs, attn_implementation="sdpa")
                self.model = model_cls.from_pretrained(target_path, **kwargs_sdpa)
            except Exception:
                self.model = model_cls.from_pretrained(target_path, **kwargs)
            if not self.load_in_4bit and self.device:
                self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[Qwen-VL] Initialized successfully using {model_cls.__name__} on device: {self.device}")
        except Exception as err:
            import traceback
            print(f"[Qwen-VL Error] Model loading failed ({err}):")
            traceback.print_exc()
            self.model = None
            self.processor = None
            raise RuntimeError(f"Failed to load Qwen-VL model: {err}")

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

        # Resize oversized scan images to max 1280px dimension to avoid quadratic attention OOM on 16GB GPUs
        max_dim = 1280
        if max(image.size) > max_dim:
            ratio = max_dim / float(max(image.size))
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Generic, strict academic transcription prompt
        prompt = (
            "You are an expert scientific document transcription system specializing in university handwritten mathematics examination scripts.\n"
            "Your task is to transcribe the handwritten mathematics examination solution shown in the image into valid, compilable, standalone LaTeX code.\n\n"
            "STRICT TRANSCRIPTION & FIDELITY RULES:\n"
            "1. STRICT FIDELITY: Transcribe exactly what the student wrote. Do NOT solve the problem, do NOT simplify expressions, do NOT correct arithmetic mistakes, and do NOT complete unfinished derivations. Preserve the student's exact mathematical notations and steps.\n"
            "2. IGNORE CANCELLED INK: Strictly ignore any scribbles, strike-throughs, or crossed-out mathematical text and equations.\n"
            "3. MATHEMATICAL ENVIRONMENTS:\n"
            "   - Use $ ... $ for inline variables, numbers, and expressions inside text sentences.\n"
            "   - Use displayed math environments \\[ ... \\] or \\begin{align*} ... \\end{align*} for standalone multi-line equations and derivations.\n"
            "   - Format matrices using standard LaTeX environments: \\begin{bmatrix} ... \\end{bmatrix} or \\begin{pmatrix} ... \\end{pmatrix} with '&' for column separation and '\\\\' for row breaks.\n"
            "   - Format piecewise definitions or equation systems using \\begin{cases} ... \\end{cases} or \\begin{aligned} ... \\end{aligned}.\n"
            "   - Transcribe standard linear algebra and calculus symbols accurately (\\in, \\notin, \\subseteq, \\mathbb{R}, \\lambda, \\vec{v}, \\boxed{...}).\n"
            "4. PRESERVE REASONING: Faithfully transcribe all explanatory prose, theorem statements, basis definitions, and English sentences written by the student.\n"
            "5. NO MARKDOWN: Output pure LaTeX code ONLY. Do NOT wrap output in markdown code fences (no ```, no ```latex, no ```markdown). Do NOT use markdown headers (no ## or ###). Use standard LaTeX sectioning (\\subsection*{...} or \\textbf{...}).\n"
            "6. COMPLETE COMPILABLE DOCUMENT: Produce a complete, standalone LaTeX document starting with \\documentclass and ending with \\end{document}."
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
        if HAS_TORCH and torch.cuda.is_available():
            torch.backends.cudnn.enabled = False

        # Stable, calibrated generation configuration tested on Tesla V100
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": 2048,
            "do_sample": False,
            "repetition_penalty": 1.10,
        }

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

            if hasattr(self.model, "dtype") and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
        except Exception as exc1:
            # Fallback using standard processor pipeline if qwen_vl_utils is not available
            try:
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[text_prompt],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                if hasattr(self.model, "dtype") and "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
            except Exception as err:
                import traceback
                print(f"[Qwen-VL Error] Transcription failed: {err}")
                traceback.print_exc()
                return ""

        # Post-processing: extract clean LaTeX code programmatically
        cleaned_text = output_text.strip()

        # 1. Strip markdown fences if present
        cleaned_text = re.sub(r"^```(?:markdown|latex|tex)?\s*", "", cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"```(?:markdown|latex|tex)?", "", cleaned_text)
        cleaned_text = cleaned_text.replace("```", "").strip()

        # 2. If text contains \end{document}, truncate any leaked trailing tokens after it
        if "\\end{document}" in cleaned_text:
            cleaned_text = cleaned_text.split("\\end{document}")[0] + "\\end{document}"

        # 3. Transpile any residual markdown headings into standard LaTeX sectioning
        cleaned_text = re.sub(r"^###\s+(.*)$", r"\\subsubsection*{\1}", cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"^##\s+(.*)$", r"\\subsection*{\1}", cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"^#\s+(.*)$", r"\\section*{\1}", cleaned_text, flags=re.MULTILINE)

        # 4. Convert markdown bold (**text**) to LaTeX \textbf{text}
        cleaned_text = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", cleaned_text)

        # 5. Generic consecutive identical line deduplication within equation/matrix blocks
        lines = cleaned_text.splitlines()
        deduped_lines = []
        for line in lines:
            stripped = line.strip()
            if (
                deduped_lines
                and stripped
                and stripped == deduped_lines[-1].strip()
                and any(kw in stripped for kw in ["pmatrix", "bmatrix", "array", "align", "begin", "end"])
            ):
                continue
            deduped_lines.append(line)
        cleaned_text = "\n".join(deduped_lines)

        # 6. Ensure full compilable document structure
        if "\\documentclass" not in cleaned_text:
            roll = roll_number or "Student"
            cleaned_text = (
                "\\documentclass[12pt]{article}\n"
                "\\usepackage[utf8]{inputenc}\n"
                "\\usepackage{amsmath, amssymb, amsfonts}\n"
                "\\usepackage{geometry}\n"
                "\\geometry{a4paper, margin=1in}\n"
                "\\title{MA102 Examination Solution}\n"
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
