import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
from torchvision.transforms import Compose, Resize, ToTensor

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filters=64):
        super(UNetGenerator, self).__init__()


        self.enc1 = self.encoder_block(in_channels, num_filters*2, normalize=False)
        self.enc2 = self.encoder_block(num_filters*2, num_filters * 4)
        self.enc3 = self.encoder_block(num_filters * 4, num_filters * 8)
        self.enc4 = self.encoder_block(num_filters * 8, num_filters * 16)


        self.bottleneck = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 16, num_filters * 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_filters * 16),
            nn.ReLU(inplace=True)
        )

        self.dec4 = self.decoder_block(num_filters * 32, num_filters * 8)
        self.dec3 = self.decoder_block(num_filters * 16, num_filters * 4)
        self.dec2 = self.decoder_block(num_filters * 8, num_filters*2)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 4, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def encoder_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def decoder_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
      layers = [
          nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
          nn.InstanceNorm2d(out_channels),
          nn.ReLU(inplace=True)
      ]
      return nn.Sequential(*layers)

    def forward(self, x):

        x = F.interpolate(x, size=(144, 256))


        enc1 = self.enc1(x)

        enc2 = self.enc2(enc1)

        enc3 = self.enc3(enc2)

        enc4 = self.enc4(enc3)



        bottleneck = self.bottleneck(enc4) # dec 5


        dec4 = self.dec4(torch.cat((bottleneck, enc4), dim=1))

        dec3 = self.dec3(torch.cat((dec4, enc3), dim=1))

        dec2 = self.dec2(torch.cat((dec3, enc2), dim=1))

        out = self.dec1(torch.cat((dec2, enc1), dim=1))



        out = F.interpolate(out, size=(140, 250))
        return out

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, num_filters=128):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            self.disc_block(in_channels, num_filters, normalize=False),
            self.disc_block(num_filters, num_filters * 2),
            self.disc_block(num_filters * 2, num_filters * 4),
            self.disc_block(num_filters * 4, num_filters * 8, stride=1),
            nn.Conv2d(num_filters * 8, 1, kernel_size=3, padding=1)
        )

    def disc_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, num_filters=128):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            self.disc_block(in_channels, num_filters, normalize=False),
            self.disc_block(num_filters, num_filters * 2),
            self.disc_block(num_filters * 2, num_filters * 4),
            self.disc_block(num_filters * 4, num_filters * 8, stride=1),
            nn.Conv2d(num_filters * 8, 1, kernel_size=3, padding=1)
        )

    def disc_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
         
          out = self.model(x)
          
          return out
