from typing import List, Optional
from src.ocr.dispatcher import ProcessedRegion

class LaTeXDocumentGenerator:
    def __init__(self, title: str = 'MA102 Answer Script Solution', author: str = 'IIT Guwahati Student'):
        self.title = title
        self.author = author

    def generate_tex_document(self, ordered_regions: List[ProcessedRegion], roll_number: Optional[str] = None) -> str:
        author_str = f'Roll Number: {roll_number}' if roll_number else self.author
        tex = [
            '\\documentclass[12pt]{article}',
            '\\usepackage[utf8]{inputenc}',
            '\\usepackage{amsmath, amssymb, amsfonts}',
            '\\usepackage{geometry}',
            '\\geometry{a4paper, margin=1in}',
            f'\\title{{{self.title}}}',
            f'\\author{{{author_str}}}',
            '\\date{\\today}',
            '\\begin{document}',
            '\\maketitle',
            '\\section*{Solution}',
            ''
        ]
        
        align_buffer: List[str] = []

        def flush_align_buffer():
            nonlocal align_buffer
            if not align_buffer:
                return
            if len(align_buffer) == 1:
                line = align_buffer[0]
                if '\\begin' not in line:
                    tex.extend(['\\[', line, '\\]', ''])
                else:
                    tex.extend([line, ''])
            else:
                formatted_lines = []
                for line in align_buffer:
                    if '=' in line and '&=' not in line:
                        line = line.replace('=', '&=', 1)
                    formatted_lines.append(line)
                body = " \\\\\n".join(formatted_lines)
                tex.extend(['\\begin{align*}', body, '\\end{align*}', ''])
            align_buffer = []

        prev_t = None
        for idx, r in enumerate(ordered_regions, 1):
            t = r.transcription.strip()
            if not t:
                continue

            # Suppress exact adjacent duplicate line transcriptions
            if t == prev_t:
                continue
            prev_t = t

            if r.region_type == 'matrix':
                flush_align_buffer()
                tex.extend(['\\[', t if '\\begin' in t else f'\\begin{{bmatrix}} {t} \\end{{bmatrix}}', '\\]', ''])
            elif r.is_math or t.startswith('\\implies'):
                align_buffer.append(t)
            else:
                flush_align_buffer()
                tex.extend([t, ''])

        flush_align_buffer()
        tex.append('\\end{document}')
        return '\n'.join(tex)

