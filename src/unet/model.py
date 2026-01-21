from __future__ import annotations
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumes.
    Input:  (N, 2, D, H, W)  (value channel + mask channel)
    Output: (N, 1, D, H, W)
    """
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base: int = 16):
        super().__init__()
        b = base

        self.enc1 = DoubleConv(in_channels, b)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(b, 2*b)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(2*b, 4*b)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(4*b, 8*b)

        self.up3 = nn.ConvTranspose3d(8*b, 4*b, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(8*b, 4*b)

        self.up2 = nn.ConvTranspose3d(4*b, 2*b, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(4*b, 2*b)

        self.up1 = nn.ConvTranspose3d(2*b, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(2*b, b)

        self.out = nn.Conv3d(b, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

