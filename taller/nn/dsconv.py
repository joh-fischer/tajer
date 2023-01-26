# Adapted from https://github.com/HansBambel/SmaAt-UNet
import torch
import torch.nn as nn


class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels: int, output_channels: int, kernel_size: int,
                 padding: int = 0, kernels_per_layer: int = 1):
        super(DepthwiseSeparableConv2D, self).__init__()
        self.depth_wise = nn.Conv2d(in_channels, in_channels * kernels_per_layer,
                                    kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.point_wise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.depth_wise(x)
        x = self.point_wise(x)

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 3, 64, 64))
    dsc = DepthwiseSeparableConv2D(3, 16, 3, padding=1)

    print("input:", ipt.shape)          # torch.Size([8, 3, 64, 64])
    print("output:", dsc(ipt).shape)    # torch.Size([8, 16, 64, 64])
