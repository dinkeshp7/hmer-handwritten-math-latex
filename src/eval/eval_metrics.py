"""
CROHME & MA102 Benchmark Evaluation Suite for HMER.

Computes standard academic evaluation metrics:
1. ExpRate0 (Expression Recognition Rate with 0 errors / exact match)
2. ExpRate1 (Expression Recognition Rate with <= 1 error tolerance)
3. ExpRate2 (Expression Recognition Rate with <= 2 errors tolerance)
4. Token Error Rate (TER) (Levenshtein edit distance at token level)
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any

import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tokenizer.latex_tokenizer import tokenize_latex


def compute_levenshtein_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    """
    Computes Levenshtein edit distance between reference and hypothesis token sequences.
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j],       # Deletion
                                   dp[i, j - 1],       # Insertion
                                   dp[i - 1, j - 1])   # Substitution

    return int(dp[n, m])


def evaluate_hmer_predictions(
    references: List[str],
    hypotheses: List[str]
) -> Dict[str, float]:
    """
    Computes ExpRate0, ExpRate1, ExpRate2, and TER over a list of reference and hypothesis LaTeX strings.
    """
    assert len(references) == len(hypotheses), "References and Hypotheses count mismatch!"
    total_samples = len(references)
    if total_samples == 0:
        return {"exprate_0": 0.0, "exprate_1": 0.0, "exprate_2": 0.0, "ter": 0.0}

    correct_0 = 0
    correct_1 = 0
    correct_2 = 0

    total_ref_tokens = 0
    total_edit_distance = 0

    for ref_str, hyp_str in zip(references, hypotheses):
        ref_toks = tokenize_latex(ref_str)
        hyp_toks = tokenize_latex(hyp_str)

        dist = compute_levenshtein_distance(ref_toks, hyp_toks)
        total_edit_distance += dist
        total_ref_tokens += max(len(ref_toks), 1)

        if dist == 0:
            correct_0 += 1
        if dist <= 1:
            correct_1 += 1
        if dist <= 2:
            correct_2 += 1

    exprate_0 = (correct_0 / total_samples) * 100.0
    exprate_1 = (correct_1 / total_samples) * 100.0
    exprate_2 = (correct_2 / total_samples) * 100.0
    ter = (total_edit_distance / max(total_ref_tokens, 1)) * 100.0

    return {
        "exprate_0": float(exprate_0),
        "exprate_1": float(exprate_1),
        "exprate_2": float(exprate_2),
        "ter": float(ter),
        "total_samples": total_samples
    }


if __name__ == "__main__":
    print("=== Testing HMER Evaluation Metrics Suite ===")
    
    sample_refs = [
        r"\frac{a}{b} + c",
        r"\sqrt{x^{2} + y^{2}}",
        r"\int_{0}^{1} x dx",
        r"\sum_{i=1}^{n} i"
    ]
    
    # Simulate predictions with minor token differences
    sample_hyps = [
        r"\frac{a}{b} + c",        # Exact match (0 errors)
        r"\sqrt{x^{2} + y^{2}}",   # Exact match (0 errors)
        r"\int_{0}^{1} x dx",      # Exact match (0 errors)
        r"\sum_{i=1}^{N} i"        # 1 error (N vs n)
    ]
    
    metrics = evaluate_hmer_predictions(sample_refs, sample_hyps)
    print("Evaluation Results:", metrics)
    
    assert metrics["exprate_0"] == 75.0, "ExpRate0 should be 75% for 3/4 exact matches!"
    assert metrics["exprate_1"] == 100.0, "ExpRate1 should be 100% for 4/4 <=1 error!"
    
    print("[OK] HMER Evaluation Metrics Suite self-test passed cleanly!")
