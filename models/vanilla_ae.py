import torch
import torch.nn as nn
from utils.utils import init_weights
from models.blocks import BasicConvBlock, UpConvBlock


class AutoEncoder(nn.Module):
    def __init__(self, img_channels=3, hidden_dims=(32, 64, 128, 256, 512)):
        super(AutoEncoder, self).__init__()

        # encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = BasicConvBlock(ch_in=img_channels, ch_out=hidden_dims[0])
        self.Conv2 = BasicConvBlock(ch_in=hidden_dims[0], ch_out=hidden_dims[1])
        self.Conv3 = BasicConvBlock(ch_in=hidden_dims[1], ch_out=hidden_dims[2])
        self.Conv4 = BasicConvBlock(ch_in=hidden_dims[2], ch_out=hidden_dims[3])
        self.Conv5 = BasicConvBlock(ch_in=hidden_dims[3], ch_out=hidden_dims[4])

        # decoder
        self.Up5 = UpConvBlock(ch_in=hidden_dims[4], ch_out=hidden_dims[3])
        self.Up4 = UpConvBlock(ch_in=hidden_dims[3], ch_out=hidden_dims[2])
        self.Up3 = UpConvBlock(ch_in=hidden_dims[2], ch_out=hidden_dims[1])
        self.Up2 = UpConvBlock(ch_in=hidden_dims[1], ch_out=hidden_dims[0])
        self.Up1 = nn.Conv2d(hidden_dims[0], img_channels, kernel_size=1, stride=1, padding=0)

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
    """ img size must be 16*N (224, 240, ...) """
    x = torch.rand(1, 3, 400, 400)
    model = AutoEncoder()
    y = model(x)
    print(x.shape, y.shape)

