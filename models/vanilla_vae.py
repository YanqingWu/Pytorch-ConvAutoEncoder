import torch
from torch import nn
from utils.utils import init_weights
from models.blocks import BasicConvBlock, UpConvBlock


class VanillaVAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128, hidden_dims=(32, 64, 128, 256, 512)):
        super(VanillaVAE, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.img_channels = img_channels

        # encoder
        self.Conv1 = BasicConvBlock(ch_in=img_channels, ch_out=hidden_dims[0])
        self.Conv2 = BasicConvBlock(ch_in=hidden_dims[0], ch_out=hidden_dims[1])
        self.Conv3 = BasicConvBlock(ch_in=hidden_dims[1], ch_out=hidden_dims[2])
        self.Conv4 = BasicConvBlock(ch_in=hidden_dims[2], ch_out=hidden_dims[3])
        self.Conv5 = BasicConvBlock(ch_in=hidden_dims[3], ch_out=hidden_dims[4])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        self.Up5 = UpConvBlock(ch_in=hidden_dims[4], ch_out=hidden_dims[3])
        self.Up4 = UpConvBlock(ch_in=hidden_dims[3], ch_out=hidden_dims[2])
        self.Up3 = UpConvBlock(ch_in=hidden_dims[2], ch_out=hidden_dims[1])
        self.Up2 = UpConvBlock(ch_in=hidden_dims[1], ch_out=hidden_dims[0])
        self.Up1 = nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=1, stride=1, padding=0)
        self.decoder_output = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0],
                               hidden_dims[0] // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[0]//2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dims[0] // 2,
                               hidden_dims[0] // 4,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[0]//4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dims[0] // 4,
                               hidden_dims[0] // 8,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[0]//8),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[0]//8, out_channels=img_channels, kernel_size=3, padding=1),
            nn.Tanh())
        self.encoder = nn.Sequential(*[self.Conv1, self.Maxpool, self.Conv2, self.Maxpool,
                                       self.Conv3, self.Maxpool, self.Conv4, self.Maxpool,
                                       self.Conv5, self.avg_pool])

        self.decoder = nn.Sequential(*[self.Up5, self.Up4, self.Up3, self.Up2, self.Up1,
                                       self.decoder_output])
        init_weights(self.encoder)
        init_weights(self.decoder)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        # result = self.final_layer(result)
        return result

    def forward(self, x):
        x = self.encode(x)
        z = self.reparameterize(x[0], x[1])
        x = self.decode(z)
        return x


if __name__ == '__main__':
    """ img size must be 16*N (224, 240, ...) """
    x = torch.rand(1, 3, 256, 256)
    model = VanillaVAE()
    z = torch.rand(model.latent_dim)
    y = model.decode(z)
    print(x.shape, y.shape)
