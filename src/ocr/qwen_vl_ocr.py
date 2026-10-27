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
            min_pixels = 512 * 28 * 28
            max_pixels = 1600 * 28 * 28
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

    @staticmethod
    def crop_booklet_instructions(image: Image.Image) -> Image.Image:
        """
        If the image contains standard printed answer booklet instructions at the top,
        crops the designated answering area to concentrate visual tokens onto student handwriting.
        """
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        if h > w and h > 1500:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
            # Search for the horizontal dividing line between 12% and 22% of page height
            y_start_search, y_end_search = int(0.12 * h), int(0.22 * h)
            sub = gray[y_start_search:y_end_search, :]
            proj = np.sum(sub < 100, axis=1)
            if len(proj) > 0 and np.max(proj) > 0.20 * w:
                peak_sub_y = int(np.argmax(proj))
                line_y = y_start_search + peak_sub_y
                # Buffer down to clear 'Space for answering Question-X only'
                crop_top = min(line_y + 120, int(0.24 * h))
                crop_bottom = int(0.97 * h)
                print(f"[Qwen-VL Preprocessor] Detected exam booklet header at y={line_y}. Cropping answer region [{crop_top}:{crop_bottom}].")
                return image.crop((0, crop_top, w, crop_bottom))
        return image

    def recognize_page(
        self,
        image_input: Union[np.ndarray, str, Image.Image],
        roll_number: Optional[str] = None,
        crop_header: bool = False
    ) -> str:
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

        # Optionally crop the top printed booklet instructions to zoom visual tokens on student ink
        if crop_header:
            image = self.crop_booklet_instructions(image)

        # High-resolution ceiling calibrated for Tesla V100 (16GB VRAM)
        max_dim = 1600
        if max(image.size) > max_dim:
            ratio = max_dim / float(max(image.size))
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Strict academic OCR transcription system prompt
        system_prompt = (
            "You are an expert mathematical OCR transcription engine specializing in university mathematics examinations and handwritten answer scripts.\n"
            "Your sole objective is to transcribe ALL handwritten mathematical expressions, derivations, matrices, and prose shown in the image into valid, compilable, standalone LaTeX code.\n\n"
            "STRICT ZERO-HALLUCINATION & FIDELITY RULES:\n"
            "1. LITERAL INK ONLY: Transcribe strictly and verbatim what is written on the paper. Do NOT solve the problem, do NOT simplify expressions, do NOT correct arithmetic mistakes, and do NOT complete unfinished derivations.\n"
            "2. NEVER SUBSTITUTE TOY PROBLEMS: You must NEVER invent or substitute standard textbook problems (e.g. NEVER substitute with 2x2 inverse matrices or diag(1,2,3)). If an augmented matrix or a 4x4 matrix with decimal numbers is present (-1, 1.5, 2.5, 3, etc.), transcribe all 4 rows and all 4 columns exactly.\n"
            "3. ZERO CONVERSATIONAL CHATTER: Do NOT output any conversational text, introductory remarks (e.g. 'This image appears to be a scanned page...', 'The provided image contains...', 'Here is the transcription...'), bulleted summaries of the image, or concluding sign-offs. Output LaTeX code ONLY.\n"
            "4. NO PROGRAMMING CODE: Absolutely NEVER generate Python code, NumPy, SymPy, or any programming scripts (NEVER output 'import numpy', 'np.array', 'def f(x)', or print statements). You are a LaTeX document OCR transcriber, NOT a software programmer. Transcribe all math into LaTeX equations ($ ... $, \\[ ... \\], \\begin{bmatrix} ... \\end{bmatrix}) ONLY.\n"
            "5. NO FAKE CHAPTERS OR EXTERNAL INPUTS: Transcribe ONLY the text and math physically present on this specific page. NEVER invent textbook structures, fake '\\input{sections/...}' directives, or chapters on unrelated topics (like complex numbers or geometry) unless literally written on the page.\n"
            "6. IGNORE SCRATCHES: Ignore scribbles, strike-throughs, or crossed-out mathematical text.\n"
            "7. MATH ENVIRONMENTS:\n"
            "   - Use $ ... $ for inline math.\n"
            "   - Use \\[ ... \\] or \\begin{align*} ... \\end{align*} for displayed equations.\n"
            "   - Format matrices using standard \\begin{bmatrix} ... \\end{bmatrix} or augmented matrix environments.\n"
            "8. COMPILABLE DOCUMENT: Produce a complete, standalone LaTeX document starting with \\documentclass[12pt]{article} and ending with \\end{document}."
        )

        user_prompt = (
            "Transcribe all mathematical expressions, matrices, row operations, derivations, "
            "and text from this image (whether handwritten or printed) into complete, compilable LaTeX code verbatim. "
            "Output LaTeX code only. Do NOT write Python code and do NOT describe the image."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        output_text = ""
        if HAS_TORCH and torch.cuda.is_available():
            torch.backends.cudnn.enabled = False

        # Stable, calibrated generation configuration tested on Tesla V100
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": 2560,
            "do_sample": False,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 15,
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

        # Post-processing: extract clean, fully-compilable LaTeX code programmatically
        cleaned_text = output_text.strip()

        # 1. Strip markdown fences (any language tag, e.g. ```latex, ```markdown, ```python)
        cleaned_text = re.sub(r"^```[a-zA-Z0-9_\-]*\s*", "", cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"```[a-zA-Z0-9_\-]*", "", cleaned_text)
        cleaned_text = cleaned_text.replace("```", "").strip()

        # 2. Extract strictly from \documentclass if present
        if "\\documentclass" in cleaned_text:
            cleaned_text = "\\documentclass" + cleaned_text.split("\\documentclass", 1)[1]

        # 3. Truncate any leaked trailing tokens after \end{document}
        if "\\end{document}" in cleaned_text:
            cleaned_text = cleaned_text.split("\\end{document}")[0] + "\\end{document}"

        # 4. Strip conversational assistant descriptions & sign-offs
        conversational_patterns = [
            r"(?i)^.*?The provided image contains.*?\n+",
            r"(?i)^.*?This image appears to be.*?\n+",
            r"(?i)^.*?Here is a (?:detailed )?description.*?\n+",
            r"(?i)^.*?The notes (?:cover|start with|are written).*?\n+",
            r"(?i)^.*?To provide more detailed information.*?\n+",
            r"(?i)\n+.*?If you (?:need|have) (?:further|more) (?:assistance|questions).*?$",
            r"(?i)\n+.*?feel free to ask.*?$",
            r"(?i)\n+.*?Given the content and structure of the notes.*?$",
            r"(?i)\n+.*?Given the complexity of the content.*?$",
            r"(?i)\n+.*?This Python (?:code|program) defines.*?$",
        ]
        for pattern in conversational_patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.MULTILINE).strip()

        # 5. Eliminate leaked Python imports / functions if model attempted coding
        if any(k in cleaned_text for k in ["import numpy", "np.array", "def f(", "np.linalg"]):
            cleaned_text = re.sub(r"(?m)^(?:python\s*)?import\s+[a-zA-Z0-9_]+.*$", "", cleaned_text)
            cleaned_text = re.sub(r"(?m)^from\s+[a-zA-Z0-9_]+\s+import.*$", "", cleaned_text)
            cleaned_text = re.sub(r"(?m)^def\s+[a-zA-Z0-9_]+\(.*?\):.*$", "", cleaned_text)
            cleaned_text = re.sub(r"(?m)^print\(.*?\).*$", "", cleaned_text)
            cleaned_text = re.sub(r"(?m)^[A-Za-z0-9_]+\s*=\s*np\.[a-zA-Z0-9_]+.*$", "", cleaned_text)
            cleaned_text = re.sub(r"(?m)^This Python (?:code|program).*$", "", cleaned_text)

        # 6. Transpile any residual markdown headings into standard LaTeX sectioning
        cleaned_text = re.sub(r"^###\s+(.*)$", r"\\subsubsection*{\1}", cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"^##\s+(.*)$", r"\\subsection*{\1}", cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"^#\s+(.*)$", r"\\section*{\1}", cleaned_text, flags=re.MULTILINE)

        # 7. Convert markdown bold (**text**) to LaTeX \textbf{text}
        cleaned_text = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", cleaned_text)

        # 8. Collapse excessive repeating token loops (e.g. 0, 0, 0, 0, 0, 0...)
        cleaned_text = re.sub(r'(?:\b0,\s*){6,}', r'0, \\dots, 0, ', cleaned_text)
        cleaned_text = re.sub(r'(?:\b0\s+&?\s*){6,}', r'0 & \\dots & 0 ', cleaned_text)

        # 9. Generic consecutive identical line deduplication within equation/matrix blocks
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
        cleaned_text = "\n".join(deduped_lines).strip()

        # 10. Ensure full compilable document structure with all mathematical packages
        req_pkgs = ["amsmath", "amssymb", "amsfonts", "mathtools", "array"]
        if "\\documentclass" not in cleaned_text:
            roll = roll_number or "Student"
            pkg_lines = "\n".join([f"\\usepackage{{{p}}}" for p in req_pkgs])
            cleaned_text = (
                "\\documentclass[12pt]{article}\n"
                "\\usepackage[utf8]{inputenc}\n"
                f"{pkg_lines}\n"
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
        else:
            if "\\begin{document}" in cleaned_text:
                preamble, body = cleaned_text.split("\\begin{document}", 1)
                for pkg in req_pkgs:
                    if pkg not in preamble:
                        preamble += f"\n\\usepackage{{{pkg}}}"
                cleaned_text = preamble + "\n\\begin{document}" + body

        # 11. Balance any unclosed LaTeX environments (ensures compilability if generation stopped)
        env_matches = re.findall(r'\\(begin|end)\{([a-zA-Z0-9*]+)\}', cleaned_text)
        stack = []
        for tag, name in env_matches:
            if name == "document":
                continue
            if tag == "begin":
                stack.append(name)
            elif tag == "end":
                if stack and stack[-1] == name:
                    stack.pop()
        if stack:
            closing_str = "\n".join([f"\\end{{{env}}}" for env in reversed(stack)])
            if "\\end{document}" in cleaned_text:
                cleaned_text = cleaned_text.replace("\\end{document}", closing_str + "\n\\end{document}")
            else:
                cleaned_text += "\n" + closing_str

        if "\\end{document}" not in cleaned_text:
            cleaned_text += "\n\\end{document}"

        return cleaned_text


if __name__ == "__main__":
    print("=== Testing Qwen-VL OCR Engine Definition ===")
    engine = QwenVLOCREngine()
    print(f"[OK] QwenVLOCREngine successfully instantiated for model: {engine.model_name}")
