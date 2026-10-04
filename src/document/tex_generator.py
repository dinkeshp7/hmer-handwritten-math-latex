from typing import List, Optional
from src.ocr.dispatcher import ProcessedRegion

class LaTeXDocumentGenerator:
    def __init__(self, title: str = 'MA102 Answer Script Solution', author: str = 'IIT Guwahati Student'):
        self.title = title
        self.author = author

    def generate_tex_document(self, ordered_regions: List[ProcessedRegion], roll_number: Optional[str] = None) -> str:
        author_str = f'Roll Number: {roll_number}' if roll_number else self.author
        tex = ['\\documentclass[12pt]{article}', '\\usepackage[utf8]{inputenc}', '\\usepackage{amsmath, amssymb, amsfonts}', '\\usepackage{geometry}', '\\geometry{a4paper, margin=1in}', f'\\title{{{self.title}}}', f'\\author{{{author_str}}}', '\\date{\\today}', '\\begin{document}', '\\maketitle', '\\section*{Solution}', '']
        for idx, r in enumerate(ordered_regions, 1):
            t = r.transcription.strip()
            if not t:
                if r.region_type == 'text_line':
                    t = f"% [Handwritten Line #{idx}]"
                elif r.region_type == 'matrix':
                    t = r"\begin{bmatrix} 0 \end{bmatrix}"
                else:
                    t = f"% [Handwritten Region #{idx}]"

            if r.region_type == 'matrix':
                tex.extend(['\\[', t if '\\begin' in t else f'\\begin{{bmatrix}} {t} \\end{{bmatrix}}', '\\]', ''])
            elif r.is_math and '\\begin' not in t and t and not t.startswith('%'):
                tex.extend(['\\begin{equation}', f'    {t}', '\\end{equation}', ''])
            else:
                tex.extend([t, ''])
        tex.append('\\end{document}')
        return '\n'.join(tex)
