"""
Stage 1 models: U-Net generator + PatchGAN discriminator
for handwriting-line -> font-line image translation.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Single downsampling block: Conv -> InstanceNorm -> LeakyReLU"""
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    """Single upsampling block: ConvTranspose -> InstanceNorm -> ReLU (+ optional Dropout)"""
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.insert(2, nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net generator for handwriting-line -> font-line translation.

    Input:  (B, in_ch, H, W)  handwriting crop, grayscale by default
    Output: (B, out_ch, H, W) font-style crop, same spatial size as input

    H and W must each be divisible by 16 (4 downsampling stages) -- pad
    or resize your crops to a multiple of 16 before feeding them in.
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()

        # Encoder (4 downsampling stages)
        self.enc1 = ConvBlock(in_ch, base_ch, norm=False)        # H/2
        self.enc2 = ConvBlock(base_ch, base_ch * 2)               # H/4
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)           # H/8
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)           # H/16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder (4 upsampling stages), each takes a concatenated skip connection
        self.dec4 = DeconvBlock(base_ch * 8, base_ch * 8, dropout=True)
        self.dec3 = DeconvBlock(base_ch * 8 * 2, base_ch * 4, dropout=True)
        self.dec2 = DeconvBlock(base_ch * 4 * 2, base_ch * 2)
        self.dec1 = DeconvBlock(base_ch * 2 * 2, base_ch)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # output in [-1, 1], matches normalized input range
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        return self.final(d1)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator. Takes a concatenated (input, output) pair and
    outputs a grid of real/fake scores, one per local patch, rather than
    a single global verdict for the whole image.
    """
    def __init__(self, in_ch=2, base_ch=64):  # in_ch=2: handwriting + font channels concatenated
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 8, 1, kernel_size=4, stride=1, padding=1),
            # no sigmoid here -- use BCEWithLogitsLoss for numerical stability
        )

    def forward(self, handwriting, font):
        x = torch.cat([handwriting, font], dim=1)
        return self.model(x)
