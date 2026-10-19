"""
OCR and Region Dispatcher Package.
"""

from src.ocr.text_ocr import HandwrittenTextOCR
from src.ocr.math_ocr import SOTAMathOCR
from src.ocr.dispatcher import HybridRegionDispatcher, ProcessedRegion
from src.ocr.nougat_ocr import NougatOCREngine
from src.ocr.qwen_vl_ocr import QwenVLOCREngine

__all__ = [
    "HandwrittenTextOCR",
    "SOTAMathOCR",
    "HybridRegionDispatcher",
    "ProcessedRegion",
    "NougatOCREngine",
    "QwenVLOCREngine",
]
