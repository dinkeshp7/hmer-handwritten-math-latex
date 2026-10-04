"""
LaTeX Document Compiler & Jinja2 Template Generator.

Compiles sequenced text lines, explanations, and math/matrix blocks
into a complete, ready-to-compile LaTeX (.tex) document file.
"""

from typing import List, Dict, Any, Optional
from src.ocr.dispatcher import ProcessedRegion


class LaTeXDocumentGenerator:
    """
    Assembles ProcessedRegion objects into a full compilable .tex file string.
    """
    def __init__(self, title: str = "MA102 Answer Script Solution", author: str = "IIT Guwahati Student"):
        self.title = title
        self.author = author

    def generate_tex_document(self, ordered_regions: List[ProcessedRegion], roll_number: Optional[str] = None) -> str:
        """
        Generates full LaTeX document string with headers, preamble, and environment blocks.
        """
        author_str = f"Roll Number: {roll_number}" if roll_number else self.author

        tex_lines = [
            r"\documentclass[12pt]{article}",
            r"\usepackage[utf8]{utf8}",
            r"\usepackage{amsmath, amssymb, amsfonts}",
            r"\usepackage{geometry}",
            r"\geometry{a4paper, margin=1in}",
            r"\title{" + self.title + r"}",
            r"\author{" + author_str + r"}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
            r"\section*{Solution}",
            ""
        ]

        for region in ordered_regions:
            text = region.transcription.strip()
            if not text:
                continue

            if region.region_type == "matrix":
                tex_lines.append(r"\[")
                tex_lines.append(text if r"\begin" in text else r"\begin{bmatrix} " + text + r" \end{bmatrix}")
                tex_lines.append(r"\]")
                tex_lines.append("")
            elif region.is_math:
                tex_lines.append(r"\begin{equation}")
                tex_lines.append("    " + text)
                tex_lines.append(r"\end{equation}")
                tex_lines.append("")
            else:
                tex_lines.append(text)
                tex_lines.append("")

        tex_lines.extend([
            r"\end{document}"
        ])

        return "\n".join(tex_lines)


if __name__ == "__main__":
    print("=== Testing LaTeX Document Generator Engine ===")
    generator = LaTeXDocumentGenerator()
    
    r1 = ProcessedRegion(crop_id=1, bbox=(50, 50, 300, 40), region_type="text_line", transcription="Given S is subspace of M_4x4(R)", is_math=False)
    r2 = ProcessedRegion(crop_id=2, bbox=(50, 150, 300, 100), region_type="matrix", transcription=r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}", is_math=True)
    
    tex_doc = generator.generate_tex_document([r1, r2], roll_number="210103096")
    print(f"Generated Document Length: {len(tex_doc)} chars")
    print(tex_doc[:300])
    assert r"\documentclass" in tex_doc, "Document header missing!"
    assert r"\begin{document}" in tex_doc, "Document environment missing!"
    print("[OK] LaTeX Document Generator self-test passed cleanly!")
