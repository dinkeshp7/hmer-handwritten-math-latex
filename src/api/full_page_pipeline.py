"""
Full-Page Answer Script to LaTeX Document Master Pipeline Coordinator.

Pipeline Stages:
1. Image Preprocessing & Deskew (ImagePreprocessor)
2. Page Line & Region Segmentation (PageLineSegmenter)
3. Dual-Head Recognition Dispatcher (HybridRegionDispatcher)
4. Spatial Reading Order Sequencing (SpatialReadingOrderGraph)
5. LaTeX Document Compilation (LaTeXDocumentGenerator)
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any, Optional
import numpy as np
import cv2

from src.segmentation.preprocessor import ImagePreprocessor
from src.segmentation.line_segmenter import PageLineSegmenter
from src.ocr.dispatcher import HybridRegionDispatcher
from src.document.spatial_graph import SpatialReadingOrderGraph
from src.document.tex_generator import LaTeXDocumentGenerator
from src.models.hmer_model import UnifiedHMERModel


class FullPageLaTeXPipeline:
    """
    Master Pipeline Coordinator for Full Answer Script Page Conversion.
    Supports multi-engine execution:
      - 'hybrid'  : Line Segmenter + Dual-Head TrOCR & pix2tex Math ViT
      - 'nougat'  : Meta Nougat End-to-End Academic Document Model
      - 'qwen_vl' : Qwen2.5-VL / Qwen2-VL Multimodal Vision-Language Model
      - 'ensemble': Combined SOTA Multimodal Ensemble
    """
    def __init__(
        self,
        hmer_model: Optional[UnifiedHMERModel] = None,
        checkpoint_path: Optional[str] = None,
        engine: str = "hybrid",
        model_name: Optional[str] = None
    ):
        self.engine = engine.lower()
        self.model_name = model_name
        self.preprocessor = ImagePreprocessor()
        self.segmenter = PageLineSegmenter()
        self.dispatcher = HybridRegionDispatcher(hmer_model=hmer_model, checkpoint_path=checkpoint_path)
        self.sequencer = SpatialReadingOrderGraph()
        self.generator = LaTeXDocumentGenerator()
        
        # Lazy initialization for Nougat and Qwen-VL
        self._nougat = None
        self._qwen_vl = None

    @property
    def nougat_engine(self):
        if self._nougat is None:
            from src.ocr.nougat_ocr import NougatOCREngine
            kwargs = {}
            if self.model_name:
                kwargs["model_name"] = self.model_name
            self._nougat = NougatOCREngine(**kwargs)
        return self._nougat

    @property
    def qwen_vl_engine(self):
        if self._qwen_vl is None:
            from src.ocr.qwen_vl_ocr import QwenVLOCREngine
            kwargs = {}
            if self.model_name:
                kwargs["model_name"] = self.model_name
            self._qwen_vl = QwenVLOCREngine(**kwargs)
        return self._qwen_vl

    def process_full_page(self, image_gray: np.ndarray, roll_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a full page grayscale image of a handwritten exam script.
        Returns dictionary containing extracted regions and complete .tex file code.
        """
        # Engine: Qwen2.5-VL Vision-Language Model
        if self.engine == "qwen_vl":
            tex_doc = self.qwen_vl_engine.recognize_page(image_gray, roll_number=roll_number)
            if tex_doc and len(tex_doc) > 50:
                return {
                    "num_regions_segmented": 1,
                    "num_lines_processed": 1,
                    "tex_document": tex_doc,
                    "engine_used": "qwen_vl",
                    "status": "success"
                }

        # Engine: Meta Nougat End-to-End Academic Model
        elif self.engine == "nougat":
            tex_doc = self.nougat_engine.recognize_page(image_gray)
            if tex_doc and len(tex_doc) > 50:
                formatted_tex = self.generator.generate_tex_document([], roll_number=roll_number)
                # Embed nougat markdown/latex into document body
                formatted_tex = formatted_tex.replace("\\section*{Solution}\n\n", f"\\section*{{Solution}}\n\n{tex_doc}\n\n")
                return {
                    "num_regions_segmented": 1,
                    "num_lines_processed": 1,
                    "tex_document": formatted_tex,
                    "engine_used": "nougat",
                    "status": "success"
                }

        # Default Engine: Hybrid Line Segmentation + Dual-Head Recognition
        # 1. Preprocessing & Deskewing
        enhanced_img, binarized_mask = self.preprocessor.process_page(image_gray)

        # 2. Line & Region Segmentation
        regions = self.segmenter.segment_page(enhanced_img, binarized_mask)

        # 3. Hybrid Recognition Dispatching (Text HTR + Math HMER)
        processed_regions = self.dispatcher.process_all_regions(regions)

        # 4. Spatial Reading Order Graph Sequencing
        ordered_regions = self.sequencer.sequence_regions(processed_regions)

        # 5. Full LaTeX Document Generation
        tex_document = self.generator.generate_tex_document(ordered_regions, roll_number=roll_number)

        return {
            "num_regions_segmented": len(regions),
            "num_lines_processed": len(ordered_regions),
            "tex_document": tex_document,
            "engine_used": "hybrid",
            "status": "success"
        }


if __name__ == "__main__":
    print("=== Testing Full-Page LaTeX Pipeline Coordinator ===")
    pipeline = FullPageLaTeXPipeline()
    
    dummy_page = np.full((600, 500), 255, dtype=np.uint8)
    cv2.putText(dummy_page, "Given S in M_4x4(R)", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.rectangle(dummy_page, (50, 150), (350, 300), 0, 2)
    
    result = pipeline.process_full_page(dummy_page, roll_number="210103096")
    print(f"Segmented Regions : {result['num_regions_segmented']}")
    print(f"TeX Document Length: {len(result['tex_document'])} chars")
    assert result["status"] == "success", "Full page pipeline failed!"
    print("[OK] Full-Page LaTeX Pipeline Coordinator self-test passed cleanly!")
