"""
Build a token vocabulary from a corpus of .tex files.

Usage:
    python build_vocab.py --corpus_dir /path/to/tex/files --out vocab.json --min_freq 2
"""
import argparse
from pathlib import Path

from tokenizer import LatexTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="vocab.json")
    parser.add_argument("--min_freq", type=int, default=2)
    args = parser.parse_args()

    tex_files = list(Path(args.corpus_dir).rglob("*.tex"))
    print(f"Found {len(tex_files)} .tex files")

    tokenizer = LatexTokenizer.build_from_corpus(tex_files, min_freq=args.min_freq)
    tokenizer.save(args.out)
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Saved to {args.out}")
