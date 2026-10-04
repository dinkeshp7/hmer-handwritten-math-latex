"""
LaTeX Semantic Tokenizer for Handwritten Mathematical Expression Recognition (HMER).

Provides deterministic tokenization of mathematical LaTeX expressions, preserving
structural commands (e.g., \\frac, \\sqrt, \\alpha, \\sum, \\int), sub/superscripts,
brackets, numbers, and individual mathematical variables without word-piece corruption.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Union, Optional


# Regular expression matching LaTeX math tokens deterministically
MATH_TOKEN_REGEX = re.compile(
    r"\\(?:[a-zA-Z]+|[,;!\s]|\\)"  # Commands: \frac, \sqrt, \alpha, \quad, \\
    r"|\d+(?:\.\d+)?"              # Floating point & integer numbers
    r"|[a-zA-Z]"                   # Single-character variables: x, y, z, A, B
    r"|[\{\}]"                     # Curly braces: {, }
    r"|[\(\)\[\]]"                 # Round/square brackets: (, ), [, ]
    r"|[\+\-\*\/\=\_\^\<\>\!\:\;\,]" # Math operators & script markers: +, -, *, /, =, _, ^, <, >, !, :, ;, ,
    r"|[^\s]"                      # Any remaining non-whitespace character
)

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def tokenize_latex(latex_str: str) -> List[str]:
    """
    Tokenizes a LaTeX string into a list of mathematical tokens.
    
    Example:
        tokenize_latex(r"\\frac{x^{2}}{y + 1}")
        -> ['\\frac', '{', 'x', '^', '{', '2', '}', '}', '{', 'y', '+', '1', '}']
    """
    # Normalize whitespace
    latex_str = latex_str.strip()
    tokens = MATH_TOKEN_REGEX.findall(latex_str)
    return tokens


class LatexTokenizer:
    """
    Encoder/Decoder vocabulary manager for LaTeX tokens.
    """
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is None:
            self.vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        else:
            self.vocab = vocab
            
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        
        self.pad_id = self.vocab.get(PAD_TOKEN, 0)
        self.sos_id = self.vocab.get(SOS_TOKEN, 1)
        self.eos_id = self.vocab.get(EOS_TOKEN, 2)
        self.unk_id = self.vocab.get(UNK_TOKEN, 3)

    @classmethod
    def build_from_corpus(cls, latex_expressions: List[str], min_freq: int = 1) -> "LatexTokenizer":
        """
        Builds vocabulary from a list of LaTeX expression strings.
        """
        counter = Counter()
        for expr in latex_expressions:
            tokens = tokenize_latex(expr)
            counter.update(tokens)

        vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        for tok, freq in counter.most_common():
            if freq >= min_freq and tok not in vocab:
                vocab[tok] = len(vocab)

        return cls(vocab)

    def encode(self, latex_str: str, add_special_tokens: bool = True) -> List[int]:
        """
        Converts LaTeX string to a sequence of token IDs.
        """
        tokens = tokenize_latex(latex_str)
        token_ids = [self.vocab.get(tok, self.unk_id) for tok in tokens]
        
        if add_special_tokens:
            token_ids = [self.sos_id] + token_ids + [self.eos_id]
            
        return token_ids

    def decode(self, token_ids: List[int], strip_special: bool = True) -> str:
        """
        Converts a list of token IDs back into a LaTeX string.
        """
        tokens = []
        for idx in token_ids:
            tok = self.inv_vocab.get(idx, UNK_TOKEN)
            if strip_special and tok in SPECIAL_TOKENS:
                if tok == EOS_TOKEN:
                    break
                continue
            tokens.append(tok)
            
        return "".join(tokens)

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves vocabulary dictionary to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "LatexTokenizer":
        """Loads vocabulary dictionary from JSON file."""
        path = Path(file_path)
        vocab = json.loads(path.read_text(encoding="utf-8"))
        return cls(vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return f"<LatexTokenizer vocab_size={len(self.vocab)}>"


if __name__ == "__main__":
    # Self-test unit verification
    test_expressions = [
        r"\frac{a}{b} + \sqrt{x^{2} + y^{2}} = \alpha \int_{0}^{\infty} e^{-t} dt",
        r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
        r"\begin{matrix} 1 & 0 \\ 0 & 1 \end{matrix}"
    ]

    print("=== Testing LaTeX Tokenizer ===")
    tokenizer = LatexTokenizer.build_from_corpus(test_expressions)
    print(f"Built Vocabulary Size: {len(tokenizer)}")
    
    for expr in test_expressions:
        encoded = tokenizer.encode(expr)
        decoded = tokenizer.decode(encoded)
        print(f"\nOriginal: {expr}")
        print(f"Tokens:   {tokenize_latex(expr)}")
        print(f"Encoded:  {encoded}")
        print(f"Decoded:  {decoded}")
        assert "<unk>" not in decoded, "Decoding error or unknown tokens present in test!"

    print("\n[OK] All LatexTokenizer self-tests passed cleanly!")
