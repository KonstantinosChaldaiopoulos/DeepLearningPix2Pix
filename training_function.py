import torch
from models import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np

def train_pix2pix(paired_loader, epochs, lambda_L1=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = UNetGenerator().to(device)
    D = PatchGANDiscriminator().to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_L1 = torch.nn.L1Loss().to(device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.00001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.999))

    generator_losses = []
    discriminator_losses = []

    for epoch in range(epochs):
        for i, (data_A, data_B) in enumerate(paired_loader):
            real_A = data_A.to(device)
            real_B = data_B.to(device)

            optimizer_G.zero_grad()

            fake_B = G(real_A)

            pred_fake_B = D(torch.cat([real_A, fake_B], 1))
            loss_GAN = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

            loss_L1 = criterion_L1(fake_B, real_B) * lambda_L1

            loss_G = loss_GAN + loss_L1

            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            pred_real = D(torch.cat([real_A, real_B], 1))
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            pred_fake = D(torch.cat([real_A, fake_B.detach()], 1))
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            loss_D.backward()
            optimizer_D.step()

          

            if i % 500 == 0:
                print(f"Epoch {epoch}, Batch {i}, Generator Loss: {loss_G.item()}, Discriminator Loss: {loss_D.item()}")

        if epoch % 1 == 0:
            torch.save(G.state_dict(), f"generator_{epoch}.pth")


            
