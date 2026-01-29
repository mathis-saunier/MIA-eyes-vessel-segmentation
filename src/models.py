import torch
import torch.nn as nn
import torch.nn.functional as F

class Contracting_Block(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()
        # Optional Max pooling (applied in forward)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pooling else None

        # Two conv blocks (no pooling inside Sequential)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.15),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Apply pooling first (when used) to downsample, then convs
        if self.pool is not None:
            x = self.pool(x)
        return self.layers(x)

class Expansive_Block(nn.Module):
    def __init__(self, in_channels, copy_channels, out_channels):
        super().__init__()
        self.up_sampling = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels + copy_channels,
                out_channels,
                kernel_size=3,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_input, x_copy):

        x_input = self.up_sampling(x_input)
        x_copy_h, x_copy_w = x_copy.size(2), x_copy.size(3)
        x_input_h, x_input_w = x_input.size(2), x_input.size(3)
        dx, dy = x_copy_w - x_input_w, x_copy_h - x_input_h
        x_copy = x_copy[:,:,dx//2:x_copy_w-(dx-dx//2), dy//2:x_copy_h-(dy-dy//2)]
        x = torch.cat([x_copy, x_input], dim=1)
        return self.layers(x)
    
class Output_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.cont1 = Contracting_Block(n_channels, 64, pooling=False)
        self.cont2 = Contracting_Block(64, 128)
        self.cont3 = Contracting_Block(128, 256)
        self.cont4 = Contracting_Block(256, 512)
        self.cont5 = Contracting_Block(512, 1024)
        self.exp1 = Expansive_Block(1024, 512, 512)
        self.exp2 = Expansive_Block(512, 256, 256)
        self.exp3 = Expansive_Block(256, 128, 128)
        self.exp4 = Expansive_Block(128, 64, 64)
        self.out = Output_Conv(64, n_classes)

    def forward(self, x):
        x1 = self.cont1(x)
        x2 = self.cont2(x1)
        x3 = self.cont3(x2)
        x4 = self.cont4(x3)
        x5 = self.cont5(x4)
        x = self.exp1(x5, x4)
        x = self.exp2(x, x3)
        x = self.exp3(x, x2)
        x = self.exp4(x, x1)
        return self.out(x)