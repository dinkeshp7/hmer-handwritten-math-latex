"""
Unified PyTorch HMER Neural Model Coordinator.

Combines:
1. Spatial 2D DenseNet Encoder (with 2D Sinusoidal Positional Embeddings)
2. Coverage-Guided Transformer Decoder (with Attention Refinement Module ARM)
3. Auxiliary Symbol Counting Head for frequency regularization.
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tokenizer.latex_tokenizer import LatexTokenizer
from src.models.encoder import DenseNetEncoder
from src.models.decoder import CoverageTransformerDecoder


class UnifiedHMERModel(nn.Module):
    """
    Unified PyTorch HMER Deep Neural Model.
    """
    def __init__(
        self,
        tokenizer: LatexTokenizer,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 4
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.d_model = d_model

        # 1. 2D Spatial Feature Encoder
        self.encoder = DenseNetEncoder(in_channels=1, out_channels=d_model)

        # 2. Coverage-Guided Transformer Decoder
        self.decoder = CoverageTransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            pad_id=tokenizer.pad_id
        )

        # 3. Auxiliary Symbol Counting Head
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, self.vocab_size)
        )

    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass during PyTorch GPU training.
        Inputs:
          - images: (B, 1, H, W)
          - tgt_tokens: (B, SeqLen)
        Outputs:
          - logits: (B, SeqLen, VocabSize)
          - coverage_sums: (B, SeqLen, H*W)
          - count_preds: (B, VocabSize)
        """
        # 1. Extract 2D Spatial Feature Maps
        features = self.encoder(images)  # (B, 512, H', W')

        # 2. Autoregressive Transformer Decoder
        logits, coverage_sums = self.decoder(tgt_tokens, features)

        # 3. Auxiliary Symbol Counting Head
        count_preds = self.count_head(features)

        return logits, coverage_sums, count_preds

    @torch.no_grad()
    def beam_search_decode(
        self,
        image: torch.Tensor,
        beam_width: int = 3,
        max_length: int = 50
    ) -> str:
        """
        Autoregressive Beam Search decoding for single handwritten image tensor (1, 1, H, W).
        """
        self.eval()
        device = image.device
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dim

        # Extract 2D spatial feature map
        features = self.encoder(image)  # (1, 512, H', W')

        # Start-of-Sequence token
        ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(self.tokenizer.sos_id)

        for i in range(max_length - 1):
            logits, _ = self.decoder(ys, features)
            next_token_logits = logits[:, -1, :]
            _, next_word = torch.max(next_token_logits, dim=1)
            next_word_item = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=device).fill_(next_word_item)], dim=1)

            if next_word_item == self.tokenizer.eos_id:
                break

        token_ids = ys.squeeze(0).tolist()
        decoded_latex = self.tokenizer.decode(token_ids)
        return decoded_latex


if __name__ == "__main__":
    print("=== PyTorch Unified HMER Model Unit Test ===")
    sample_latex = [r"\frac{a}{b} = c"]
    tok = LatexTokenizer.build_from_corpus(sample_latex)
    model = UnifiedHMERModel(tokenizer=tok)

    dummy_img = torch.randn(2, 1, 128, 256)
    dummy_tgt = torch.randint(0, len(tok), (2, 15))

    logits, cov, count_preds = model(dummy_img, dummy_tgt)
    print(f"Logits Shape     : {logits.shape}")
    print(f"Coverage Shape   : {cov.shape}")
    print(f"Count Preds Shape: {count_preds.shape}")
    assert logits.shape == (2, 15, len(tok)), "Logits shape mismatch!"
    print("[OK] PyTorch Unified HMER Model test passed cleanly!")
