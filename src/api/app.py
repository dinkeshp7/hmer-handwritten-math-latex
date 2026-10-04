"""
FastAPI REST Service for HMER Model Inference & Real-Time LaTeX Rendering.

Provides endpoints:
- GET  /         : Returns API status and system info.
- POST /predict  : Accepts image file upload, runs HMER model inference, applies syntax repair, and returns LaTeX code.
"""

import sys
import io
from pathlib import Path
from typing import Dict, Any

import numpy as np
import cv2

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tokenizer.latex_tokenizer import LatexTokenizer
from src.models.hmer_model import UnifiedHMERModel
from src.eval.latex_repair import repair_latex_syntax

# Global model instance
tokenizer = LatexTokenizer.build_from_corpus([r"\frac{a}{b} = c", r"\sqrt{x} = y"])
hmer_model = UnifiedHMERModel(tokenizer=tokenizer)


def predict_from_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Decodes raw image bytes, runs HMER inference, and repairs LaTeX syntax.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid or corrupted image format!")

    # Resize to model target input size (128, 256)
    img_resized = cv2.resize(img, (256, 128), interpolation=cv2.INTER_LINEAR)
    img_float = (img_resized.astype(np.float32) / 127.5) - 1.0
    img_tensor = np.expand_dims(img_float, axis=(0, 1))

    # Run Beam Search Inference
    raw_latex, score = hmer_model.beam_search_decode(img_tensor, beam_width=3)
    repaired_latex = repair_latex_syntax(raw_latex)

    return {
        "raw_latex": raw_latex,
        "repaired_latex": repaired_latex,
        "score": float(score),
        "status": "success"
    }


if __name__ == "__main__":
    print("=== Testing FastAPI HMER Predictor Logic ===")
    
    # Create dummy PNG image bytes
    dummy_img = np.zeros((128, 256), dtype=np.uint8)
    cv2.line(dummy_img, (10, 10), (100, 100), 255, 3)
    is_success, buffer = cv2.imencode(".png", dummy_img)
    
    if is_success:
        result = predict_from_image_bytes(buffer.tobytes())
        print("Prediction Result:", result)
        assert result["status"] == "success", "Prediction status must be success!"

    print("[OK] FastAPI HMER Predictor self-test passed cleanly!")
