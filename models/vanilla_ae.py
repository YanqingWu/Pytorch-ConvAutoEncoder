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
    def __init__(self, img_channels=3):
        super(AutoEncoder, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = BasicConvBlock(ch_in=img_channels, ch_out=64)
        self.Conv2 = BasicConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = BasicConvBlock(ch_in=128, ch_out=256)
        self.Conv4 = BasicConvBlock(ch_in=256, ch_out=512)
        self.Conv5 = BasicConvBlock(ch_in=512, ch_out=1024)

        self.Up5 = UpConvBlock(ch_in=1024, ch_out=512)
        self.Up4 = UpConvBlock(ch_in=512, ch_out=256)
        self.Up3 = UpConvBlock(ch_in=256, ch_out=128)
        self.Up2 = UpConvBlock(ch_in=128, ch_out=64)
        self.Up1 = nn.Conv2d(64, img_channels, kernel_size=1, stride=1, padding=0)

        self.encoder = nn.Sequential(*[self.Conv1, self.Maxpool, self.Conv2, self.Maxpool,
                                       self.Conv3, self.Maxpool, self.Conv4, self.Maxpool, self.Conv5])
        self.decoder = nn.Sequential(*[self.Up5, self.Up4, self.Up3, self.Up2, self.Up1])
        init_weights(self.encoder)
        init_weights(self.decoder)

    def forward(self, x):
        # encoding path
        x = self.encoder(x)

        # decoding path
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    """ img size must be 2^N """
    x = torch.rand(1, 3, 512, 512)
    model = AutoEncoder()
    y = model(x)
    print(x.shape, y.shape)

