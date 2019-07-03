import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, norm, residual, dropout=0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, channel, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1)
        self.bn1 = nn.BatchNorm3d(channel)
        self.do = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1)
        self.bn2 = nn.BatchNorm3d(channel)
        self.residual = residual
        self.norm = norm

    def forward(self, input: torch.Tensor):
        out = F.relu(self.conv1(input))
        out = self.bn1(out) if self.norm else out
        out = self.do(out)
        out = F.relu(self.conv2(out))
        out = self.bn2(out) if self.norm else out
        return torch.cat((input, out), dim=1) if self.residual else out


class UNetBlock(nn.Module):
    def __init__(self, in_channel, channel, depth, kernel_size, norm, max_pool, res, dropout):
        super().__init__()
        self.depth = depth
        if depth > 0:
            self.conv1 = ConvBlock(in_channel, channel, kernel_size, norm, res)
            self.max_pool = max_pool
            if max_pool:
                self.pool = nn.MaxPool3d(2)
            else:
                self.down_conv = nn.Conv3d(channel, channel, kernel_size=3, stride=2, padding=1)
            self.unet = UNetBlock(channel, channel*2, depth-1, kernel_size, norm, max_pool, res, dropout)
            self.upsamp = nn.Upsample(scale_factor=2)
#             self.up_conv = nn.Conv3d(channel*2, channel, kernel_size=3, stride=1, padding=1)
            self.conv2 = ConvBlock(channel*2+channel, channel, kernel_size, norm, res)
        else:
            self.inner_conv = ConvBlock(in_channel, channel, kernel_size, norm, res, dropout)

    def forward(self, input):
        if self.depth > 0:
            x = self.conv1(input)
            if self.max_pool:
                y = self.pool(x)
            else:
                y = F.relu(self.down_conv(x))

            y = self.unet(y)
            y = self.upsamp(y)
#             y = F.relu(self.up_conv(y))
            y = torch.cat((x, y), dim=1)
            y = self.conv2(y)
            return y
        else:
            return self.inner_conv(input)


class UNet(nn.Module):
    def __init__(self, start_channels: int = 16, depth: int = 2, kernel_size: int = 3,
                 in_channels: int = 2, out_channels: int = 1, residual: bool = False, batch_norm: bool = False,
                 dropout: float = 0.5, max_pool=True):
        super().__init__()
        self.unet = UNetBlock(in_channel=in_channels, channel=start_channels, depth=depth, kernel_size=kernel_size, norm=batch_norm, res=residual, max_pool=max_pool, dropout=dropout)
        self.final_conv = nn.Conv3d(start_channels, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, input):
        x = self.unet(input)
        x = self.final_conv(x)
        return x

