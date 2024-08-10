import torch.nn as nn
import torch.nn.functional as F
import torch

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0),
            nn.Conv3d(512, 1, 4, padding=1, bias=False),
        )
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )


    def forward(self, feature):
        output = self.model(feature)
        output = output.view(output.size(0), -1)
        return self.final(output)


