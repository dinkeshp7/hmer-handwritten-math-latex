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
    # Compatibility shim for PyTorch < 2.1 (e.g. PyTorch 2.0.1+cu118 on HPC)
    # Transformers 4.49+ calls torch.compiler.is_compiling() during generation
    if not hasattr(torch, "compiler"):
        import types
        _compiler = types.ModuleType("compiler")
        _compiler.is_compiling = lambda: False
        _compiler.disable = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda x: x)
        _compiler.wrap_numpy = lambda fn=None: fn if fn is not None else (lambda x: x)
        torch.compiler = _compiler
    elif not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False
except ImportError:
    HAS_TORCH = False


class QwenVLOCREngine:
    """
    Qwen-VL Vision-Language Mathematical Transcription Engine.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        local_model_dir: Optional[str] = None,
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
            if self.model_name and os.path.exists(self.model_name):
                target_path = self.model_name
                print(f"[Qwen-VL] Loading specified offline weights from: {target_path}")
            elif self.local_model_dir and os.path.exists(self.local_model_dir):
                target_path = self.local_model_dir
                print(f"[Qwen-VL] Loading offline weights from local_model_dir: {target_path}")
            elif os.path.exists("/scratch/p.dinkesh/hmer_project/models/qwen_vl_7b") or os.path.exists("./models/qwen_vl_7b"):
                target_path = "/scratch/p.dinkesh/hmer_project/models/qwen_vl_7b" if os.path.exists("/scratch/p.dinkesh/hmer_project/models/qwen_vl_7b") else "./models/qwen_vl_7b"
                print(f"[Qwen-VL] Loading flagship 7B offline weights from: {target_path}")
            elif os.path.exists("./models/qwen_vl"):
                target_path = "./models/qwen_vl"
                print(f"[Qwen-VL] Loading default offline weights from: {target_path}")
            else:
                target_path = self.model_name
                print(f"[Qwen-VL] Loading model weights for: {target_path}")

            from transformers import AutoProcessor
            # Dynamic resolution bounds calibrated for Tesla V100 (16GB VRAM)
            # 1024 * 28 * 28 (~800k pixels) fits perfectly inside V100 16GB memory budget
            # For 3-band slices (1200x470 px = 564k px), this provides 1200px width (3.5x token density vs full page)
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
            for cls_name in [
                "AutoModelForImageTextToText",
                "AutoModelForMultimodalLM",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen2VLForConditionalGeneration",
                "AutoModelForVision2Seq"
            ]:
                try:
                    import transformers
                    if hasattr(transformers, cls_name):
                        model_cls = getattr(transformers, cls_name)
                        break
                except Exception:
                    pass

            if model_cls is None:
                raise ImportError(
                    "None of AutoModelForImageTextToText, AutoModelForMultimodalLM, "
                    "Qwen2_5_VLForConditionalGeneration, or Qwen2VLForConditionalGeneration "
                    "could be imported from transformers. Please install transformers>=4.46.3."
                )

            kwargs: Dict[str, Any] = {}
            use_device_map = False
            if self.device == "cuda":
                # Use bfloat16 if supported, else float16
                if torch.cuda.is_bf16_supported():
                    kwargs["torch_dtype"] = torch.bfloat16
                else:
                    kwargs["torch_dtype"] = torch.float16
                if self.load_in_4bit:
                    kwargs["load_in_4bit"] = True
                else:
                    try:
                        import accelerate
                        kwargs["device_map"] = "auto"
                        use_device_map = True
                    except ImportError:
                        pass
                
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
            if not use_device_map and not self.load_in_4bit and self.device:
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

    @staticmethod
    def slice_into_bands(image: Image.Image, num_bands: int = 3, overlap_ratio: float = 0.08) -> list:
        """
        Slices a full-page image vertically into overlapping horizontal bands.
        Each band preserves 100% native scanner width and resolution while fitting within GPU VRAM.
        """
        w, h = image.size
        if num_bands <= 1:
            return [image]
        
        band_h = h / float(num_bands)
        overlap_px = int(band_h * overlap_ratio)
        
        bands = []
        for i in range(num_bands):
            y_start = max(0, int(i * band_h) - (overlap_px if i > 0 else 0))
            y_end = min(h, int((i + 1) * band_h) + (overlap_px if i < num_bands - 1 else 0))
            bands.append(image.crop((0, y_start, w, y_end)))
        return bands

    def recognize_page(
        self,
        image_input: Union[np.ndarray, str, Image.Image],
        roll_number: Optional[str] = None,
        crop_header: bool = False,
        max_dim: Optional[int] = 1200,
        full_res: bool = False
    ) -> str:
        """
        Transcribes an answer script page or image band directly into complete LaTeX code.
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
        # If full_res is False and max_dim is set, only resize if image exceeds max_dim (default 1800)
        # Horizontal strips (e.g. 1653x888) have max dimension 1653 <= 1800, preserving 100% native quality
        if not full_res and max_dim is not None and max(image.size) > max_dim:
            ratio = max_dim / float(max(image.size))
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Strict academic OCR transcription system prompt
        system_prompt = (
            "You are an exact, literal mathematical OCR transcription engine. Your sole task is to transcribe "
            "the student's handwritten exam answer script verbatim into LaTeX code.\n\n"
            "CRITICAL COPYIST DIRECTIVES (ZERO-HALLUCINATION):\n"
            "1. LITERAL HANDWRITTEN INK ONLY: Transcribe strictly and verbatim what the student physically wrote on the paper. "
            "You are a photocopier, NOT a mathematician. DO NOT solve the problem, DO NOT calculate row reductions, "
            "DO NOT correct arithmetic errors, and DO NOT invent your own steps.\n"
            "2. PRESERVE STUDENT'S EXACT INTERMEDIATE MATRICES & ROW OPERATIONS: If the student wrote an elementary row operation "
            "like 'R_1 -> -R_1' or 'R_2 -> R_2 - 3.5 R_1', transcribe it verbatim. If the student wrote a matrix following that operation, "
            "transcribe every single number in all 4 rows and all 4 columns exactly as written by the student, even if the student made an arithmetic mistake.\n"
            "3. TRANSCRIBE ALL EQUATIONS IN COLUMNS: If the student wrote multiple equations vertically (e.g. four row-sum equations), "
            "transcribe every single equation without omitting any line.\n"
            "4. NEVER WRITE A LECTURE OR TUTORIAL: Absolutely NEVER write commentary, explanations of Gaussian elimination, "
            "introductory textbook remarks (e.g., 'To determine if there exists a solution...', 'Step 1: Swap R1 and R2...', 'After performing these steps...'), "
            "or disclaimers about missing constants. Output the student's handwritten mathematical ink ONLY.\n"
            "5. NO PROGRAMMING CODE: Transcribe math into standard LaTeX environments ($...$, \\[ ... \\], \\begin{bmatrix} ... \\end{bmatrix}). Never output Python, NumPy, or SymPy.\n"
            "6. COMPILABLE DOCUMENT: Produce a complete LaTeX document starting with \\documentclass[12pt]{article} and ending with \\end{document}."
        )

        user_prompt = (
            "Transcribe this handwritten student exam answer script page verbatim into LaTeX. "
            "Do NOT solve the math. Do NOT invent your own row operations or matrices. "
            "Transcribe all handwritten equations, all matrix entries, and all student row reduction steps "
            "line-by-line exactly as physically written by the student's pen."
        )

        roll = roll_number or "Student"
        prefill_text = (
            "\\documentclass[12pt]{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{amsmath, amssymb, amsfonts, mathtools, array}\n"
            "\\usepackage{geometry}\n"
            "\\geometry{a4paper, margin=1in}\n"
            "\\title{Student Answer Script Transcription}\n"
            f"\\author{{Roll Number: {roll}}}\n"
            "\\date{\\today}\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": prefill_text
            }
        ]

        output_text = ""
        if HAS_TORCH and torch.cuda.is_available():
            torch.backends.cudnn.enabled = False

        # Pure mathematical OCR greedy decoding (no n-gram bans or repetition penalty)
        # Banning n-grams or penalizing repetition destroys repeated matrix rows and vertical equation sets
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": 2560,
            "do_sample": False,
        }

        try:
            from qwen_vl_utils import process_vision_info
            try:
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, continue_final_message=True)
                is_prefilled = True
            except Exception:
                # Fallback if continue_final_message is unsupported
                messages_no_asst = messages[:2]
                text_prompt = self.processor.apply_chat_template(messages_no_asst, tokenize=False, add_generation_prompt=True)
                is_prefilled = False

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

            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                decoded_body = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                output_text = (prefill_text + decoded_body) if is_prefilled else decoded_body
        except Exception as exc1:
            # Fallback using standard processor pipeline if qwen_vl_utils is not available
            try:
                try:
                    text_prompt = self.processor.apply_chat_template(messages, tokenize=False, continue_final_message=True)
                    is_prefilled = True
                except Exception:
                    messages_no_asst = messages[:2]
                    text_prompt = self.processor.apply_chat_template(messages_no_asst, tokenize=False, add_generation_prompt=True)
                    is_prefilled = False

                inputs = self.processor(
                    text=[text_prompt],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                if hasattr(self.model, "dtype") and "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)

                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    decoded_body = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    output_text = (prefill_text + decoded_body) if is_prefilled else decoded_body
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

        # 4. Strip conversational assistant descriptions & sign-offs (both paragraph-based and regex-based)
        chatter_keywords = [
            "to summarize", "the image shows", "the provided image", "this image appears",
            "here is a detailed description", "presence of multiple lines of handwriting",
            "educational purposes", "problem-solving exercise", "classroom setting",
            "this python code", "this python program", "given the complexity of the content",
            "given the content and structure of the notes", "feel free to ask",
            "if you need further assistance", "indicating that it might have been used",
            "the given question involves", "to determine if there exists a solution",
            "since no further details were provided", "without additional constraints",
            "it remains impossible to provide explicit numerical solutions",
            "henceforth, based upon available data alone", "no definitive conclusion",
            "after performing these steps correctly, you will eventually reach",
            "in case of inconsistency, the system would look like",
            "indicates infinite number of solutions due to lack of independent constraints",
            "thus based on above analysis, one needs to carefully perform",
            "step-by-step row operations", "initial matrix:", "swap r1 and r2 so that",
            "subtract appropriate multiples of new r2 from other rows"
        ]
        paragraphs = cleaned_text.split("\n\n")
        retained_paragraphs = []
        for p in paragraphs:
            p_strip = p.strip()
            p_lower = p_strip.lower()
            if any(kw in p_lower for kw in chatter_keywords) and not any(env in p_strip for env in ["\\begin{", "\\[", "\\("]):
                continue
            retained_paragraphs.append(p)
        cleaned_text = "\n\n".join(retained_paragraphs).strip()

        conversational_patterns = [
            r"(?i)^.*?To summarize, the image shows.*?\n*",
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
            r"(?i)^.*?The presence of multiple lines of handwriting suggests.*?\n*",
            r"(?i)^.*?indicating that it might have been used.*?\n*",
        ]
        for pattern in conversational_patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.MULTILINE).strip()

        # 4b. Eliminate hallucinated textbook/monograph structures and index spam
        cleaned_text = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r"\\chapter\*?\{.*?\}", "", cleaned_text)
        cleaned_text = re.sub(r"\\tableofcontents", "", cleaned_text)
        cleaned_text = re.sub(r"\\mainmatter", "", cleaned_text)
        cleaned_text = re.sub(r"\\index\{.*?\}", "", cleaned_text)
        cleaned_text = re.sub(r"\\Index\{.*?\}", "", cleaned_text)

        # 4c. Deduplicate \maketitle if model generated an extra \maketitle in document body
        if cleaned_text.count("\\maketitle") > 1:
            first_idx = cleaned_text.find("\\maketitle")
            cleaned_text = (
                cleaned_text[:first_idx + len("\\maketitle")]
                + cleaned_text[first_idx + len("\\maketitle"):].replace("\\maketitle", "")
            )

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

    def recognize_page_chunked(
        self,
        image_input: Union[np.ndarray, str, Image.Image],
        roll_number: Optional[str] = None,
        num_chunks: int = 3,
        crop_header: bool = False
    ) -> str:
        """
        Slices a full-page image into overlapping horizontal bands (default: 3 parts),
        transcribes each band at 100% native scanner resolution (zero downsampling),
        and stitches the recognized LaTeX bodies into a single unified compilable document.
        """
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

        if crop_header:
            image = self.crop_booklet_instructions(image)

        bands = self.slice_into_bands(image, num_bands=num_chunks, overlap_ratio=0.08)
        print(f"[Qwen-VL Tiling] Page sliced into {len(bands)} full-resolution horizontal bands:")
        for idx, b in enumerate(bands):
            print(f"  Band {idx+1}/{len(bands)}: Dimensions {b.size[0]}x{b.size[1]} px (100% Native Quality)")

        extracted_bodies = []
        for idx, band in enumerate(bands):
            print(f"[Qwen-VL Tiling] Transcribing Band {idx+1}/{len(bands)} at full native resolution...")
            band_tex = self.recognize_page(
                band,
                roll_number=roll_number,
                crop_header=False,
                max_dim=1800,
                full_res=True
            )
            body = band_tex
            if "\\begin{document}" in body:
                body = body.split("\\begin{document}", 1)[1]
            if "\\maketitle" in body:
                body = body.split("\\maketitle", 1)[1]
            if "\\end{document}" in body:
                body = body.split("\\end{document}", 1)[0]
            body = body.strip()
            if body:
                extracted_bodies.append(f"% --- Band {idx+1} Content ---\n" + body)

            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

        combined_body = "\n\n".join(extracted_bodies)
        roll = roll_number or "Student"
        full_document = (
            "\\documentclass[12pt]{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{amsmath, amssymb, amsfonts, mathtools, array}\n"
            "\\usepackage{geometry}\n"
            "\\geometry{a4paper, margin=1in}\n"
            "\\title{Student Answer Script Transcription (High-Resolution Multi-Band)}\n"
            f"\\author{{Roll Number: {roll}}}\n"
            "\\date{\\today}\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
            f"{combined_body}\n\n"
            "\\end{document}\n"
        )
        return full_document


if __name__ == "__main__":
    print("=== Testing Qwen-VL OCR Engine Definition ===")
    engine = QwenVLOCREngine()
    print(f"[OK] QwenVLOCREngine successfully instantiated for model: {engine.model_name}")
