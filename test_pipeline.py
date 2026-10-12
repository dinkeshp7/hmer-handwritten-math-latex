"""
Pipeline Verification Script for HMER Layout Parser, Dispatcher, and TeX Generator.
"""

import os
import sys
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tokenizer.latex_tokenizer import LatexTokenizer
from src.models.hmer_model import UnifiedHMERModel
from src.ocr.text_ocr import HandwrittenTextOCR
from src.ocr.dispatcher import HybridRegionDispatcher, ProcessedRegion
from src.segmentation.line_segmenter import PageLineSegmenter, RegionCrop
from src.segmentation.matrix_parser import MatrixGridParser
from src.document.tex_generator import LaTeXDocumentGenerator

def test_full_pipeline():
    print("=== Testing Updated HMER Full Pipeline Components ===")
    
    # 1. Tokenizer
    tok = LatexTokenizer()
    print(f"1. LatexTokenizer default vocab size: {len(tok)}")
    assert len(tok) > 50, "Tokenizer vocabulary size should be expanded!"

    # 2. Text OCR (Offline capable)
    ocr = HandwrittenTextOCR()
    print("2. HandwrittenTextOCR initialized successfully.")

    # 3. Hybrid Region Dispatcher
    dispatcher = HybridRegionDispatcher()
    print("3. HybridRegionDispatcher initialized with full tokenizer.")

    # 4. Matrix Grid Parser
    m_parser = MatrixGridParser()
    dummy_matrix = np.full((200, 300), 255, dtype=np.uint8)
    latex_matrix = m_parser.parse_matrix_to_latex(dummy_matrix)
    print("4. MatrixGridParser output:")
    print(latex_matrix)

    # 5. Document Generator Test
    regions = [
        ProcessedRegion(crop_id=1, bbox=(10, 10, 200, 30), region_type="text_line", transcription="Given, S \\subset M_{4 \\times 4}(\\mathbb{R})", is_math=False),
        ProcessedRegion(crop_id=2, bbox=(10, 50, 200, 30), region_type="text_line", transcription="Here, collection of matrices whose 4^{th} column = sum(1^{st}, 2^{nd}, 3^{rd})", is_math=False),
        ProcessedRegion(crop_id=3, bbox=(10, 90, 200, 30), region_type="math_block", transcription=r"(3) \implies -1 + 1.5 + 2.5 = 4 - 1 = 3", is_math=True),
        ProcessedRegion(crop_id=4, bbox=(10, 130, 200, 30), region_type="math_block", transcription=r"\implies 3.5 - 2 + 0.5 = 4 - 2 = 2", is_math=True),
        ProcessedRegion(crop_id=5, bbox=(10, 170, 200, 30), region_type="math_block", transcription=r"\implies 4.5 + 2.5 - 3 = 7 - 3 = 4", is_math=True),
        ProcessedRegion(crop_id=6, bbox=(10, 210, 200, 30), region_type="math_block", transcription=r"\implies -0.5 - 1.5 + 3 = 3 - 2 = 1", is_math=True),
        ProcessedRegion(crop_id=7, bbox=(10, 250, 200, 30), region_type="text_line", transcription=r"Arithmetically A or S is subset of M_{4 \times 4}(\mathbb{R}) since all the conditions are true.", is_math=False),
        ProcessedRegion(crop_id=8, bbox=(10, 290, 200, 150), region_type="matrix", transcription=r"\begin{bmatrix} -1 & 1.5 & 2.5 & 3 \\ 3.5 & -2 & 0.5 & 2 \\ 4.5 & 2.5 & -3 & 4 \\ -0.5 & -1.5 & 3 & 1 \end{bmatrix}", is_math=True)
    ]

    doc_gen = LaTeXDocumentGenerator()
    tex_doc = doc_gen.generate_tex_document(regions, roll_number="210103096")
    print("\n5. Generated LaTeX Document Output:")
    print("======================================================================")
    print(tex_doc)
    print("======================================================================")

    assert r"\begin{align*}" in tex_doc, "Multi-line align environment missing!"
    assert r"\begin{bmatrix}" in tex_doc, "Matrix environment missing!"
    print("\n[OK] ALL HMER PIPELINE TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    test_full_pipeline()
