"""
LaTeX Multi-Band Document Stitcher.
Combines multiple LaTeX outputs generated from high-resolution image slices (Part 1, Part 2, Part 3)
into a single unified, compilable LaTeX document.

Usage:
    python stitch_latex_parts.py output_qwen_vl_page-210103096-01_part*.tex --output output_qwen_vl_page-210103096-01_stitched.tex
"""

import sys
import re
import argparse
from pathlib import Path


def extract_body(tex_content):
    """Extracts the mathematical content between preamble and \\end{document}."""
    body = tex_content.strip()
    
    # Strip markdown fences if present
    body = re.sub(r"^```[a-zA-Z0-9_\-]*\s*", "", body, flags=re.MULTILINE)
    body = re.sub(r"\s*```$", "", body, flags=re.MULTILINE)
    body = body.replace("```", "").strip()

    if "\\begin{document}" in body:
        body = body.split("\\begin{document}", 1)[1]
    if "\\maketitle" in body:
        body = body.split("\\maketitle", 1)[1]
    if "\\end{document}" in body:
        body = body.split("\\end{document}")[0]
        
    return body.strip()


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def stitch_latex_files(input_files, output_file, roll_number=None):
    """Stitches multiple .tex part files into a single master document."""
    sorted_files = sorted(input_files, key=natural_sort_key)
    if not roll_number or roll_number == "auto":
        roll_match = re.search(r"(\d{9})", " ".join([str(f) for f in sorted_files]))
        roll_number = roll_match.group(1) if roll_match else "210103096"

    print(f"[Stitcher] Target Roll Number: {roll_number}")
    print(f"[Stitcher] Combining {len(sorted_files)} LaTeX part files in natural order:")
    for f in sorted_files:
        print(f"  - {f}")

    bodies = []
    for idx, fpath in enumerate(sorted_files):
        p = Path(fpath)
        if not p.exists():
            print(f"  [Warning] File not found: {fpath}")
            continue
        raw_text = p.read_text(encoding="utf-8")
        body = extract_body(raw_text)
        if body:
            header = f"% ==================================================\n% Content from Slice {idx + 1}: {p.name}\n% =================================================="
            bodies.append(f"{header}\n{body}")

    combined_body = "\n\n".join(bodies)
    
    # Clean unescaped underscores in text mode outside math ($...$, \[...\], \begin{...})
    # Protect math blocks first using an alphanumeric placeholder with NO underscores
    math_placeholders = []
    def save_math(m):
        math_placeholders.append(m.group(0))
        return f"ZZMATHBLOCK{len(math_placeholders)-1}ZZ"

    protected = re.sub(r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\begin\{.*?\}.*?\\end\{.*?\})", save_math, combined_body, flags=re.DOTALL)
    # Escape any bare underscores in prose text
    protected = re.sub(r"(?<!\\)_", r"\\_", protected)
    # Restore math blocks cleanly
    for idx, mb in enumerate(math_placeholders):
        protected = protected.replace(f"ZZMATHBLOCK{idx}ZZ", mb)
    combined_body = protected

    # Balance unclosed LaTeX environments
    env_matches = re.findall(r'\\(begin|end)\{([a-zA-Z0-9*]+)\}', combined_body)
    stack = []
    for tag, name in env_matches:
        if name == "document":
            continue
        if tag == "begin":
            stack.append(name)
        elif tag == "end":
            if stack and stack[-1] == name:
                stack.pop()
    if stack:
        closing_str = "\n".join([f"\\end{{{env}}}" for env in reversed(stack)])
        combined_body += "\n" + closing_str

    master_doc = (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{amsmath, amssymb, amsfonts, mathtools, array}\n"
        "\\usepackage{geometry}\n"
        "\\geometry{a4paper, margin=1in}\n"
        "\\title{Student Answer Script Transcription (High-Resolution Stitched Document)}\n"
        f"\\author{{Roll Number: {roll_number}}}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        f"{combined_body}\n\n"
        "\\end{document}\n"
    )

    out_p = Path(output_file)
    out_p.write_text(master_doc, encoding="utf-8")
    print(f"\n[Stitcher] Successfully saved master document to: {out_p.resolve()}")
    return master_doc


def main():
    parser = argparse.ArgumentParser(description="Stitch multiple LaTeX part files into a single unified compilable document.")
    parser.add_argument("files", nargs="+", help="Input LaTeX part files (e.g. part1.tex part2.tex part3.tex)")
    parser.add_argument("--output", type=str, default="stitched_solution.tex", help="Output stitched LaTeX file path")
    parser.add_argument("--roll_number", type=str, default=None, help="Student roll number (auto-detected from filename if omitted)")
    args = parser.parse_args()

    stitch_latex_files(args.files, args.output, roll_number=args.roll_number)


if __name__ == "__main__":
    main()
