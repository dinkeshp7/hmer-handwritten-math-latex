"""
Meta Nougat (Neural Optical Understanding for Academic Documents) OCR Engine.

End-to-end full-page academic and mathematical document-to-LaTeX converter.
Bypasses manual line segmentation by taking full-page scans directly and
decoding structured mathematical markdown/LaTeX via Swin Transformer + mBART.
"""

import os
from typing import Optional, Union
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class NougatOCREngine:
    """
    Meta Nougat Full-Page Mathematical OCR Engine.
    """
    def __init__(
        self,
        model_name: str = "facebook/nougat-small",
        local_model_dir: Optional[str] = "/scratch/p.dinkesh/hmer_project/models/nougat",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.device = device or ("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Loads Nougat processor and vision-encoder-decoder weights."""
        if self.model is not None:
            return

        try:
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            target_path = None
            if self.local_model_dir and os.path.exists(self.local_model_dir):
                target_path = self.local_model_dir
                print(f"[Nougat] Loading offline weights from: {target_path}")
            elif os.path.exists("./models/nougat"):
                target_path = "./models/nougat"
                print(f"[Nougat] Loading offline weights from: {target_path}")
            else:
                target_path = self.model_name
                print(f"[Nougat] Loading model weights for: {target_path}")

            self.processor = NougatProcessor.from_pretrained(target_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(target_path).to(self.device)
            self.model.eval()
            print(f"[Nougat] Model initialized successfully on device: {self.device}")
        except Exception as err:
            import traceback
            print(f"[Nougat Error] Failed to load Nougat ({err}):")
            traceback.print_exc()
            self.model = None
            self.processor = None
            raise RuntimeError(f"Failed to load Meta Nougat model: {err}")

    def recognize_page(self, image_input: Union[np.ndarray, str, Image.Image]) -> str:
        """
        Transcribes a full-page answer script or document image into formatted LaTeX.
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

        try:
            if HAS_TORCH and torch.cuda.is_available():
                torch.backends.cudnn.enabled = False

            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            gen_kwargs = {
                "min_length": 1,
                "max_new_tokens": 1024,
            }
            if hasattr(self.processor, "tokenizer") and getattr(self.processor.tokenizer, "unk_token_id", None) is not None:
                gen_kwargs["bad_words_ids"] = [[self.processor.tokenizer.unk_token_id]]

            with torch.no_grad():
                outputs = self.model.generate(pixel_values, **gen_kwargs)
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            cleaned_latex = self.processor.post_process_generation(decoded, fix_markdown=True)
            return cleaned_latex
        except Exception as err:
            import traceback
            print(f"[Nougat Error] Inference failed: {err}")
            traceback.print_exc()
            return ""


if __name__ == "__main__":
    print("=== Testing Nougat OCR Engine Definition ===")
    engine = NougatOCREngine()
    print("[OK] NougatOCREngine class successfully defined and ready.")
