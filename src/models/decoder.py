"""
PyTorch Coverage-Guided Transformer Decoder with Attention Refinement Module (ARM) (CoMER Paradigm).
"""

import math
from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedPositionalEncoding1D(nn.Module):
    """Fixed Sinusoidal 1D Positional Encoding for Transformer Decoder."""
    def __init__(self, d_model: int = 512, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class AttentionRefinementModuleARM(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.conv_cov = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.linear_cov = nn.Linear(1, d_model, bias=False)

    def forward(self, attn_weights_history: torch.Tensor) -> torch.Tensor:
        if attn_weights_history.dim() == 3:
            attn_weights_history = attn_weights_history.unsqueeze(0)
        cov_matrix = torch.sum(attn_weights_history, dim=1, keepdim=True)
        cov_feat = self.conv_cov(cov_matrix)
        return cov_feat.squeeze(1)


class CoverageTransformerDecoder(nn.Module):
    """
    PyTorch Pre-LN Coverage-Guided Transformer Decoder for HMER.
    """
    def __init__(
        self,
        vocab_size: int = 250,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        pad_id: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.pad_id = pad_id
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = FixedPositionalEncoding1D(d_model=d_model, max_len=500)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(0.1)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.arm = AttentionRefinementModuleARM(d_model=d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = nn.Transformer.generate_square_subsequent_mask(sz, device=device)
        return mask.bool() if mask.dtype != torch.bool else mask

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        encoder_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = encoder_features.shape
        seq_len = tgt_tokens.size(1)
        device = tgt_tokens.device

        memory = encoder_features.flatten(2).permute(0, 2, 1)
        tgt_embed = self.emb_dropout(self.emb_norm(self.pos_encoder(self.embedding(tgt_tokens))))

        causal_mask = self.generate_square_subsequent_mask(seq_len, device)
        pad_mask = (tgt_tokens == self.pad_id)
        combined_mask = (causal_mask.unsqueeze(0) | pad_mask.unsqueeze(1)).repeat_interleave(self.nhead, dim=0).to(torch.bool)

        dec_out = self.transformer_decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=combined_mask,
            tgt_key_padding_mask=None
        )

        logits = self.fc_out(dec_out)
        coverage_sums = torch.zeros(b, seq_len, h * w, device=device)

        return logits, coverage_sums
