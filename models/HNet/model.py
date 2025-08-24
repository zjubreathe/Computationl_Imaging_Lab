import torch
import torch.nn as nn
import torch.nn.functional as F
from pyexpat import features

from fontTools.t1Lib import writePFB

# 反卷积核
from utils import Converse2D, LayerNorm, Converse2D_kernel, KernelNet

class block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=1, padding=2, padding_mode='replicate',
                 eps=1e-5):
        super(block, self).__init__()
        self.conv1 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(in_channels, 2 * out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   Converse2D(2 * out_channels, 2 * out_channels, kernel_size, scale=scale,
                                              padding=padding, padding_mode=padding_mode, eps=eps),
                                   nn.GELU(),
                                   nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0))

        self.conv2 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(out_channels, 2 * out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        return x

class block_psf(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=1, padding=2, padding_mode='replicate',
                 eps=1e-5):
        super(block_psf, self).__init__()
        self.conv1_1 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(in_channels, 2 * out_channels, 1, 1, 0),
                                   nn.GELU())
        self.conv1_2 = Converse2D_kernel(2 * out_channels)
        self.conv1_3 =  nn.Sequential(nn.GELU(),
                                      nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0))
        self.conv2 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(out_channels, 2 * out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0))

    def forward(self, x, p):
        x_ = x
        x = self.conv1_1(x)
        x = self.conv1_2(x,p,sf=1)
        x = self.conv1_3(x) + x_
        x = self.conv2(x) + x
        return x

class HNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: int = 32, num_levels: int = 4):
        super(HNet, self).__init__()

        self.intro  = nn.Conv2d(in_channels, features, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(features, out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Encoder
        for num in range(num_levels):
            self.encoders.append(
                nn.Sequential(block(features,features))
            )
            self.downs.append(
                nn.Conv2d(features, 2*features, 2, 2)
            )
            features = features * 2

        # Bottleneck
        self.bottleneck = block(features, features)

        # Decoder
        for num in range(num_levels):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(features, 2*features, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            features = features // 2
            self.decoders.append(
                nn.Sequential(
                    block(features, features)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.intro(x)
        enc_skips = []

        # Encoder path
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            enc_skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for decoder,up, enc_skip in zip(self.decoders, self.ups, enc_skips[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        return x[:, :, :H, :W]

class HNet_a(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: int = 32, num_levels: int = 4):
        super(HNet_a, self).__init__()

        self.intro  = nn.Conv2d(in_channels, features, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(features, out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.conv_encs = nn.ModuleList()
        self.conv_decs = nn.ModuleList()

        self.kernel = KernelNet(kernel_size=7)

        # Encoder
        for num in range(num_levels):
            self.encoders.append(
                block_psf(features,features)
            )
            self.downs.append(
                nn.Conv2d(features, 2*features, 2, 2)
            )
            self.conv_encs.append(nn.Conv2d(16, features*2, 1, 1, 0))
            features = features * 2

        # Bottleneck
        self.bottleneck = block_psf(features, features)
        self.conv_bot = nn.Conv2d(16, features*2, 1, 1, 0)

        # Decoder
        for num in range(num_levels):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(features, 2*features, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            features = features // 2
            self.decoders.append(
                    block_psf(features, features)
            )
            self.conv_decs.append(nn.Conv2d(16, features*2, 1, 1, 0))

    def forward(self, x: torch.Tensor, p:torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.intro(x)
        enc_skips = []

        b, c, h, w = p.shape
        p_ = self.kernel(p)

        # Encoder path
        for encoder, down in zip(self.encoders, self.downs):
            p = self.conv_encs[len(enc_skips)](p_)
            x = encoder(x,p)
            enc_skips.append(x)
            x = down(x)

        # Bottleneck
        p = self.conv_bot(p_)
        x = self.bottleneck(x,p)

        levels = len(enc_skips)

        # Decoder path
        for decoder,up, enc_skip in zip(self.decoders, self.ups, enc_skips[::-1]):
            p = self.conv_decs[levels-len(enc_skips)](p_)
            enc_skips.pop()
            x = up(x)
            x = x + enc_skip
            x = decoder(x,p)

        x = self.ending(x)

        return x[:, :, :H, :W]



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HNet_a(in_channels=1, out_channels=1, features=32).to(device)

    test_input = torch.randn(1, 1, 256, 256).to(device)
    psf_input = torch.randn(1, 1, 7, 7).to(device)

    with torch.no_grad():
        output = model(test_input,psf_input)

    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
