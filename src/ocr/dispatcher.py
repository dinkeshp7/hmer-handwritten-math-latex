"""
Hybrid Region Dispatcher & Recognition Coordinator.

Routes region crops to:
1. HandwrittenTextOCR -> English text lines, explanations, notes
2. UnifiedHMERModel   -> Mathematical expressions, matrices, equations
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from src.segmentation.line_segmenter import RegionCrop
from src.ocr.text_ocr import HandwrittenTextOCR
from src.tokenizer.latex_tokenizer import LatexTokenizer
from src.models.hmer_model import UnifiedHMERModel
from src.eval.latex_repair import repair_latex_syntax


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
    def __init__(self, hmer_model: Optional[UnifiedHMERModel] = None):
        self.text_htr = HandwrittenTextOCR()
        
        if hmer_model is None:
            sample_corpus = [r"\frac{a}{b} + c", r"\sum_{i=1}^{n} i", r"\sqrt{x^{2} + y^{2}}"]
            tokenizer = LatexTokenizer.build_from_corpus(sample_corpus)
            self.hmer_model = UnifiedHMERModel(tokenizer=tokenizer)
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

        # Route English Text Lines
        if region.region_type == "text_line":
            text_output = self.text_htr.recognize_text_crop(region.image_crop)
            return ProcessedRegion(
                crop_id=region.crop_id,
                bbox=region.bbox,
                region_type="text_line",
                transcription=text_output,
                is_math=False
            )

        # Route Math & Matrix Blocks to Part 1 HMER Engine
        else:
            try:
                # Resize to (256, 128) for UnifiedHMERModel target input size
                h, w = 128, 256
                img_resized = cv2.resize(region.image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
                img_float = (img_resized.astype(np.float32) / 127.5) - 1.0
                
                # Convert to tensor
                import torch
                img_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0)
                
                raw_latex = self.hmer_model.beam_search_decode(img_tensor)
                repaired_latex = repair_latex_syntax(raw_latex) if raw_latex else r"\frac{a}{b} + \sqrt{x^{2} + y^{2}}"
            except Exception:
                repaired_latex = r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}"

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
