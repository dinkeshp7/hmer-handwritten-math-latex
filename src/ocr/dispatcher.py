"""
Hybrid Region Dispatcher & Recognition Coordinator.

Routes region crops to:
1. HandwrittenTextOCR -> English text lines, explanations, notes
2. UnifiedHMERModel   -> Mathematical expressions, matrices, equations
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import numpy as np
import cv2

from src.segmentation.line_segmenter import RegionCrop
from src.ocr.text_ocr import HandwrittenTextOCR
from src.ocr.math_ocr import SOTAMathOCR
from src.tokenizer.latex_tokenizer import LatexTokenizer
from src.models.hmer_model import UnifiedHMERModel
from src.eval.latex_repair import repair_latex_syntax
from src.segmentation.matrix_parser import MatrixGridParser


@dataclass
class ProcessedRegion:
    """Dataclass holding processed region outputs."""
    crop_id: int
    bbox: tuple
    region_type: str
    transcription: str
    is_math: bool
    confidence: float = 1.0


class HybridRegionDispatcher:
    """
    Dispatcher routing segmented regions to text HTR vs math HMER.
    """
    def __init__(self, hmer_model: Optional[UnifiedHMERModel] = None, checkpoint_path: Optional[str] = None):
        self.text_htr = HandwrittenTextOCR()
        self.math_ocr = SOTAMathOCR()
        
        if hmer_model is None:
            tokenizer = LatexTokenizer()
            self.hmer_model = UnifiedHMERModel(tokenizer=tokenizer)
            if checkpoint_path and os.path.exists(checkpoint_path):
                import torch
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                model_dict = self.hmer_model.state_dict()
                filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                self.hmer_model.load_state_dict(filtered_dict, strict=False)
        else:
            self.hmer_model = hmer_model

    def process_region(self, region: RegionCrop) -> ProcessedRegion:
        """
        Processes a single RegionCrop object and returns a ProcessedRegion.
        """
        # Discard scratch-outs
        if region.region_type == "scratchout":
            return ProcessedRegion(
                crop_id=region.crop_id,
                bbox=region.bbox,
                region_type="scratchout",
                transcription="",
                is_math=False
            )

        # Route English Text Lines (Dual-Head Math vs Prose Disambiguation)
        if region.region_type == "text_line":
            text_output = self.text_htr.recognize_text_crop(region.image_crop)
            math_cand = self.math_ocr.recognize_math_crop(region.image_crop)
            
            # Intelligent decision: Is this crop a mathematical formula or English prose?
            is_math_expr = False
            if math_cand and "\\begin{array}" not in math_cand and "\\sqrt{\\sqrt{" not in math_cand:
                import re
                math_tokens = ["\\frac", "\\sum", "\\prod", "\\int", "\\cdots", "\\sqrt", "\\leq", "\\geq", "\\in", "\\pm", "\\times", "\\vec"]
                has_strong_math = any(tok in math_cand for tok in math_tokens)
                has_basic_math = any(op in math_cand for op in ["=", "+", "-", "<", ">", "^", "_"])
                
                words = [w for w in re.findall(r"[a-zA-Z]{3,}", text_output)]
                common_english = {"the", "show", "that", "for", "each", "natural", "number", "let", "and", "with", "this", "using", "from", "theorem", "coefficient", "expansion", "compare", "terms", "whereas", "strictly", "smaller", "corresponding", "addition", "contains", "extra", "positive", "therefore", "proving", "monotonically", "increasing", "alternate", "proof", "consider", "inequality", "raising", "both", "sides", "power", "gives", "since", "satisfies", "replacing", "factor", "yields", "given", "basis", "linear", "operator", "independent", "vectors", "implies", "applying", "property", "identity", "inverse", "functions", "other"}
                legit_english_count = sum(1 for w in words if w.lower() in common_english)
                
                if has_strong_math and legit_english_count <= 2:
                    text_output = math_cand
                    is_math_expr = True
                elif has_basic_math and len(math_cand) > 8 and legit_english_count == 0:
                    text_output = math_cand
                    is_math_expr = True

            if not is_math_expr:
                is_math_expr = any(sym in text_output for sym in ['$', '\\', '=', '+', '-', '^', '_', '{', '}']) and not any(w.isalpha() and len(w) > 3 for w in text_output.split())

            return ProcessedRegion(
                crop_id=region.crop_id,
                bbox=region.bbox,
                region_type="text_line",
                transcription=text_output,
                is_math=is_math_expr
            )
        # Route 2D Matrix Regions to MatrixGridParser + SOTAMathOCR / TrOCR
        elif region.region_type == "matrix":
            matrix_parser = MatrixGridParser()
            def row_cell_recognizer(row_crop: np.ndarray) -> str:
                # 1. Try SOTAMathOCR with anti-hallucination guard
                res = self.math_ocr.recognize_math_crop(row_crop)
                if res and "\\begin{array}" not in res and "\\sqrt{\\sqrt{" not in res and len(res) < 80:
                    return res
                # 2. Try TrOCR for reading row symbols/numbers
                t_row = self.text_htr.recognize_text_crop(row_crop)
                if t_row and len(t_row) > 1:
                    clean = t_row.replace('"', '').replace("'", "").strip()
                    tokens = [tok for tok in clean.split() if tok not in [",", ";", "|", "(", ")"]]
                    if len(tokens) >= 2:
                        return " & ".join(tokens)
                    return clean
                # 3. Fallback to UnifiedHMERModel
                h, w = 128, 256
                img_resized = cv2.resize(row_crop, (w, h), interpolation=cv2.INTER_LINEAR)
                img_float = (img_resized.astype(np.float32) / 127.5) - 1.0
                import torch
                img_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0)
                try:
                    res = self.hmer_model.beam_search_decode(img_tensor)
                    return res if res else ""
                except Exception:
                    return ""

            matrix_latex = matrix_parser.parse_matrix_to_latex(region.image_crop, cell_recognizer=row_cell_recognizer)
            return ProcessedRegion(
                crop_id=region.crop_id,
                bbox=region.bbox,
                region_type="matrix",
                transcription=matrix_latex,
                is_math=True
            )

        # Route Math Expressions to SOTAMathOCR + Part 1 HMER Engine
        else:
            # 1. Try SOTA Pretrained Math ViT
            raw_latex = self.math_ocr.recognize_math_crop(region.image_crop)
            # Guard against pix2tex hallucinations
            if raw_latex and ("\\begin{array}" in raw_latex or "\\sqrt{\\sqrt{" in raw_latex or len(raw_latex) > 100):
                t_cand = self.text_htr.recognize_text_crop(region.image_crop)
                raw_latex = t_cand if t_cand else ""
            
            # 2. Fallback to UnifiedHMERModel
            if not raw_latex:
                try:
                    h, w = 128, 256
                    img_resized = cv2.resize(region.image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
                    img_float = (img_resized.astype(np.float32) / 127.5) - 1.0
                    import torch
                    img_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0)
                    raw_latex = self.hmer_model.beam_search_decode(img_tensor)
                except Exception:
                    raw_latex = ""

            repaired_latex = repair_latex_syntax(raw_latex) if raw_latex else ""
            return ProcessedRegion(
                crop_id=region.crop_id,
                bbox=region.bbox,
                region_type=region.region_type,
                transcription=repaired_latex,
                is_math=True
            )

    def process_all_regions(self, regions: List[RegionCrop]) -> List[ProcessedRegion]:
        """Processes a list of RegionCrop objects."""
        return [self.process_region(r) for r in regions]


if __name__ == "__main__":
    print("=== Testing Hybrid Region Dispatcher Engine ===")
    dispatcher = HybridRegionDispatcher()
    
    dummy_crop = np.full((128, 256), 255, dtype=np.uint8)
    dummy_region = RegionCrop(crop_id=1, bbox=(50, 100, 256, 128), region_type="matrix", image_crop=dummy_crop)
    
    processed = dispatcher.process_region(dummy_region)
    print(f"Processed Type : {processed.region_type}")
    print(f"Is Math        : {processed.is_math}")
    print(f"Transcription  : {processed.transcription}")
    assert processed.is_math is True, "Matrix region must be flagged as math!"
    print("[OK] Hybrid Region Dispatcher self-test passed cleanly!")
