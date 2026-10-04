"""
LaTeX tokenizer for Stage 2.

Splits LaTeX source into tokens that keep structural/spacing commands
intact as single tokens (e.g. \\begin{itemize}, \\\\, \\quad) rather than
letting a generic subword tokenizer fragment them -- this is what lets
the decoder learn "gap size -> spacing command" reliably instead of
treating whitespace as noise.
"""
import json
import re
from collections import Counter
from pathlib import Path

TOKEN_PATTERN = re.compile(
    r"\\begin\{[a-zA-Z*]+\}"   # \begin{itemize}
    r"|\\end\{[a-zA-Z*]+\}"    # \end{itemize}
    r"|\\[a-zA-Z]+\*?"         # \section, \textbf, \quad, \par ...
    r"|\\\\"                   # \\  (explicit line break)
    r"|[{}]"                   # braces
    r"|[&%$#_^~]"              # other reserved chars
    r"|\s+"                    # whitespace run, kept as its own token
    r"|[^\\{}\s]+"             # ordinary word/number runs
)

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize(latex_str):
    return TOKEN_PATTERN.findall(latex_str)


class LatexTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    @classmethod
    def build_from_corpus(cls, tex_files, min_freq=2):
        counter = Counter()
        for path in tex_files:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            counter.update(tokenize(text))

        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok, freq in counter.most_common():
            if freq >= min_freq and tok not in vocab:
                vocab[tok] = len(vocab)
        return cls(vocab)

    def save(self, path):
        Path(path).write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path):
        vocab = json.loads(Path(path).read_text())
        return cls(vocab)

    def encode(self, latex_str, add_special=True):
        unk = self.vocab["<unk>"]
        ids = [self.vocab.get(tok, unk) for tok in tokenize(latex_str)]
        if add_special:
            ids = [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]
        return ids

    def decode(self, ids, strip_special=True):
        tokens = [self.inv_vocab.get(i, "<unk>") for i in ids]
        if strip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return "".join(tokens)

    def __len__(self):
        return len(self.vocab)
