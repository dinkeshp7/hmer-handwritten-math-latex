"""
OCR and Region Dispatcher Package.
"""

from src.ocr.text_ocr import HandwrittenTextOCR
from src.ocr.dispatcher import HybridRegionDispatcher, ProcessedRegion

__all__ = ["HandwrittenTextOCR", "HybridRegionDispatcher", "ProcessedRegion"]
