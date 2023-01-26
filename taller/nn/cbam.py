import torch
import torch.nn as nn


# Adapted from: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        """
        Compute channel attention (average- and max-pooled features) to model
        inter-channel relationship ('what' to focus). This squeezes the spatial
        dimension of the input feature map, s.t. the input CxHxW will be mapped
        to Cx1x1. The argument `hidden_ch` can be used to create a bottleneck
        in the mlp (https://arxiv.org/abs/1807.06521).
        Args:
            channels: Number of channels of the feature maps.
            reduction_ratio: Reduction ratio for bottleneck.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False))

    def forward(self, x: torch.Tensor):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out

        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        """
        Compute spatial attention (average- and max-pooled features) to model
        attention along the spatial dimensions ('where' to focus). This squeezes
        the channel dimension of the input feature map, s.t. the input CxHxW will
        be mapped to 1xHxW (paper: https://arxiv.org/abs/1807.06521).
        Args:
            kernel_size: Kernel size for the convolution.
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x)

        return torch.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor):
        x *= self.channel_attn(x)
        x *= self.spatial_attn(x)

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 32, 64, 64))
    print("Input:", ipt.shape)              # torch.Size([8, 32, 64, 64])

    c_attn = ChannelAttention(32)
    print("Channel:", c_attn(ipt).shape)    # torch.Size([8, 32, 1, 1])

    s_attn = SpatialAttention()
    print("Spatial:", s_attn(ipt).shape)    # torch.Size([8, 1, 64, 64])

    cbam = CBAM(32)
    print("CBAM:", cbam(ipt).shape)         # torch.Size([8, 32, 64, 64])
