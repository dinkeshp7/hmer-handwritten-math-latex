"""
Stage 2 model: font-page image -> LaTeX.

ViT-style patch encoder + autoregressive transformer decoder, with
optional layout (bounding box) conditioning -- feeding line geometry
from the segmentation module as extra tokens gives the decoder an
explicit structural signal for line breaks instead of relying purely
on the vision transformer noticing a pixel gap is wider than usual.
"""
import torch
import torch.nn as nn


class PositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding for a (rows x cols) patch grid."""
    def __init__(self, embed_dim, max_rows=128, max_cols=128):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for 2D positional encoding"
        self.row_embed = nn.Embedding(max_rows, embed_dim // 2)
        self.col_embed = nn.Embedding(max_cols, embed_dim // 2)

    def forward(self, rows, cols, device):
        r = torch.arange(rows, device=device)
        c = torch.arange(cols, device=device)
        row_emb = self.row_embed(r).unsqueeze(1).expand(rows, cols, -1)
        col_emb = self.col_embed(c).unsqueeze(0).expand(rows, cols, -1)
        pos = torch.cat([row_emb, col_emb], dim=-1)   # (rows, cols, embed_dim)
        return pos.reshape(rows * cols, -1)


class ImageEncoder(nn.Module):
    """
    Patch-embedding + transformer encoder over a font-rendered page image.
    Optionally accepts layout boxes (from the Stage 1 segmentation module)
    as extra tokens appended to the patch sequence.
    """
    def __init__(self, in_ch=1, embed_dim=512, patch_size=16,
                 depth=6, n_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = PositionalEncoding2D(embed_dim)
        self.layout_proj = nn.Linear(4, embed_dim)  # (x0, y0, x1, y1) normalized boxes
        layer = nn.TransformerEncoderLayer(
            embed_dim, n_heads, ff_dim, dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images, boxes=None, box_mask=None):
        """
        images: (B, C, H, W), H and W divisible by patch_size
        boxes:  (B, L, 4) optional normalized [0, 1] line boxes
        box_mask: (B, L) optional, 1 for real boxes / 0 for padding
        """
        B = images.shape[0]
        patches = self.patch_embed(images)                   # (B, embed_dim, H/p, W/p)
        rows, cols = patches.shape[2], patches.shape[3]
        patches = patches.flatten(2).transpose(1, 2)          # (B, N, embed_dim)
        patches = patches + self.pos_embed(rows, cols, images.device).unsqueeze(0)

        tokens = patches
        pad_mask = torch.zeros(B, patches.shape[1], dtype=torch.bool, device=images.device)

        if boxes is not None:
            layout_tokens = self.layout_proj(boxes)            # (B, L, embed_dim)
            tokens = torch.cat([tokens, layout_tokens], dim=1)
            if box_mask is not None:
                layout_pad = ~box_mask.bool()
            else:
                layout_pad = torch.zeros(B, boxes.shape[1], dtype=torch.bool, device=images.device)
            pad_mask = torch.cat([pad_mask, layout_pad], dim=1)

        memory = self.encoder(tokens, src_key_padding_mask=pad_mask)
        return self.norm(memory), pad_mask


class LatexDecoder(nn.Module):
    """Autoregressive transformer decoder emitting LaTeX tokens."""
    def __init__(self, vocab_size, embed_dim=512, depth=6, n_heads=8,
                 ff_dim=2048, dropout=0.1, max_len=1024, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerDecoderLayer(
            embed_dim, n_heads, ff_dim, dropout, batch_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, target_ids, memory, memory_pad_mask=None):
        B, T = target_ids.shape
        positions = torch.arange(T, device=target_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embed(target_ids) + self.pos_embed(positions)

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=target_ids.device), diagonal=1
        )
        tgt_pad_mask = target_ids == self.pad_idx

        out = self.decoder(
            x, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.output_proj(self.norm(out))


class Img2LatexModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, patch_size=16,
                 enc_depth=6, dec_depth=6, n_heads=8, ff_dim=2048,
                 dropout=0.1, max_len=1024, pad_idx=0):
        super().__init__()
        self.encoder = ImageEncoder(1, embed_dim, patch_size, enc_depth, n_heads, ff_dim, dropout)
        self.decoder = LatexDecoder(vocab_size, embed_dim, dec_depth, n_heads, ff_dim, dropout, max_len, pad_idx)

    def forward(self, images, target_ids, boxes=None, box_mask=None):
        memory, memory_pad_mask = self.encoder(images, boxes, box_mask)
        return self.decoder(target_ids, memory, memory_pad_mask)
