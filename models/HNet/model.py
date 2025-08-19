import torch
import torch.nn as nn
import torch.nn.functional as F

# 反卷积核
from utils import Converse2D, LayerNorm

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

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HNet(in_channels=1, out_channels=1, features=32).to(device)

    test_input = torch.randn(1, 1, 256, 256).to(device)

    with torch.no_grad():
        output = model(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
