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
    """
    def __init__(self, hmer_model: Optional[UnifiedHMERModel] = None, checkpoint_path: Optional[str] = None):
        self.preprocessor = ImagePreprocessor()
        self.segmenter = PageLineSegmenter()
        self.dispatcher = HybridRegionDispatcher(hmer_model=hmer_model, checkpoint_path=checkpoint_path)
        self.sequencer = SpatialReadingOrderGraph()
        self.generator = LaTeXDocumentGenerator()

    def process_full_page(self, image_gray: np.ndarray, roll_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a full page grayscale image of a handwritten exam script.
        Returns dictionary containing extracted regions and complete .tex file code.
        """
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
