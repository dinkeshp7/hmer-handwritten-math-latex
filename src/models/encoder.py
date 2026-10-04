"""
PyTorch 2D Spatial Feature Encoder for HMER (ResNet/DenseNet Backbone + 2D Positional Embeddings).

Extracts 2D feature maps from input handwritten math images (B, 1, H, W)
and injects 2D Sinusoidal Positional Encoding to preserve 2D spatial topology.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """
    2D Sinusoidal Positional Encoding for 2D Feature Maps (B, C, H, W).
    """
    def __init__(self, d_model: int = 512, height: int = 32, width: int = 64):
        super().__init__()
        self.d_model = d_model
        self.height = height
        self.width = width

        d_half = d_model // 2
        d_quarter = d_half // 2

        pe = torch.zeros(d_model, height, width)

        # Height (Y-axis) positional encoding
        pos_y = torch.arange(0, height, dtype=torch.float).unsqueeze(1)  # (H, 1)
        div_y = torch.exp(torch.arange(0, d_quarter, dtype=torch.float) * -(math.log(10000.0) / d_quarter)).unsqueeze(0) # (1, d_quarter)
        sin_y = torch.sin(pos_y * div_y).unsqueeze(1).repeat(1, width, 1)  # (H, W, d_quarter)
        cos_y = torch.cos(pos_y * div_y).unsqueeze(1).repeat(1, width, 1)  # (H, W, d_quarter)

        pe[0:d_half:2, :, :] = sin_y.permute(2, 0, 1)
        pe[1:d_half:2, :, :] = cos_y.permute(2, 0, 1)

        # Width (X-axis) positional encoding
        pos_x = torch.arange(0, width, dtype=torch.float).unsqueeze(1)  # (W, 1)
        div_x = torch.exp(torch.arange(0, d_quarter, dtype=torch.float) * -(math.log(10000.0) / d_quarter)).unsqueeze(0) # (1, d_quarter)
        sin_x = torch.sin(pos_x * div_x).unsqueeze(0).repeat(height, 1, 1)  # (H, W, d_quarter)
        cos_x = torch.cos(pos_x * div_x).unsqueeze(0).repeat(height, 1, 1)  # (H, W, d_quarter)

        pe[d_half::2, :, :] = sin_x.permute(2, 0, 1)
        pe[d_half+1::2, :, :] = cos_x.permute(2, 0, 1)

        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        b, c, h, w = x.shape
        pe_slice = self.pe[:, :, :h, :w]
        return x + pe_slice


class ConvBlock(nn.Module):
    """Residual Convolutional Block with BatchNorm and ReLU."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x) + self.shortcut(x), inplace=True)


class DenseNetEncoder(nn.Module):
    """
    Robust 2D Spatial Feature Encoder Backbone for handwritten math images.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 512
    ):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.layer3 = ConvBlock(128, 256, stride=2)
        self.layer4 = ConvBlock(256, out_channels, stride=1)
        
        self.pos_encoder_2d = PositionalEncoding2D(d_model=out_channels, height=32, width=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pos_encoder_2d(out)
        return out  # Shape: (B, 512, H', W')


if __name__ == "__main__":
    print("=== PyTorch 2D Encoder Unit Test ===")
    encoder = DenseNetEncoder(in_channels=1, out_channels=512)
    dummy_img = torch.randn(2, 1, 128, 256)
    out_feat = encoder(dummy_img)
    print(f"Input Image Shape : {dummy_img.shape}")
    print(f"Output Feature Map: {out_feat.shape}")
    assert out_feat.shape[1] == 512, "Encoder output channels must be 512!"
    print("[OK] PyTorch 2D Encoder test passed cleanly!")
