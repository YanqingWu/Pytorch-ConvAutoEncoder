import torch
import torch.nn as nn
from utils.utils import init_weights


class BasicConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, channels=(3, 32, 64, 128, 256)):
        super(AutoEncoder, self).__init__()
        encoder = []
        for c in range(len(channels)-1):
            encoder.append(BasicConvBlock(channels[c], channels[c+1]))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for c in range(len(channels)-1, 0, -1):
            decoder.append(BasicConvBlock(channels[c], channels[c-1]))
        self.decoder = nn.Sequential(*decoder)
        init_weights(self.encoder)
        init_weights(self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    x = torch.rand(1, 3, 500, 500)
    model = AutoEncoder()
    y = model(x)
    assert x.shape == y.shape

