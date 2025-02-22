import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import os
from PIL import Image
import kagglehub


path = kagglehub.dataset_download("sabahesaraki/pix2pix-facades-dataset")


os.makedirs(path, exist_ok=True)

dataset = dset.ImageFolder(root=data_root, transform=transforms.Compose([
    transforms.Resize((256, 512)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

#The Generator architecture based on UNET
class UNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            self.down_block(input_channels, features, batch_norm=False),
            self.down_block(features, features * 2),
            self.down_block(features * 2, features * 4),
            self.down_block(features * 4, features * 8),
            self.down_block(features * 8, features * 8),
        )
        self.decoder = nn.Sequential(
            self.up_block(features * 8, features * 8),
            self.up_block(features * 8, features * 4),
            self.up_block(features * 4, features * 2),
            self.up_block(features * 2, features),
            nn.ConvTranspose2d(features, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def down_block(self, in_channels, out_channels, batch_norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

#The critic architecture
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=6, features=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self.disc_block(input_channels, features, batch_norm=False),
            self.disc_block(features, features * 2),
            self.disc_block(features * 2, features * 4),
            nn.Conv2d(features * 4, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def disc_block(self, in_channels, out_channels, batch_norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Initializing the  models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

# Initializing the losses and optimizerss
criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

print("Using device:", device)
print("Generator on CUDA:", next(generator.parameters()).is_cuda)
print("Discriminator on CUDA:", next(discriminator.parameters()).is_cuda)

print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")




num_epochs = 150
for epoch in range(num_epochs):
    for batch in dataloader:
        real_images = batch[0].to(device)
        real_A, real_B = real_images[:, :, :, 256:], real_images[:, :, :, :256]

        # Train Discriminator
        optimizer_D.zero_grad()
        fake_B = generator(real_A)
        real_pair = torch.cat((real_A, real_B), 1)
        fake_pair = torch.cat((real_A, fake_B.detach()), 1)
        real_output = discriminator(real_pair.to(device))
        fake_output = discriminator(fake_pair.to(device))
        loss_D = (criterion_GAN(real_output, torch.ones_like(real_output, device=device)) +
                  criterion_GAN(fake_output, torch.zeros_like(fake_output, device=device))) / 2

        loss_D.backward()
        optimizer_D.step()

        # Training the  Generator
        optimizer_G.zero_grad()
        fake_pair = torch.cat((real_A, fake_B), 1)
        fake_output = discriminator(fake_pair)
        loss_G_GAN = criterion_GAN(fake_output, torch.ones_like(fake_output))
        loss_G_L1 = criterion_L1(fake_B, real_B) * 100
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch + 1}/{num_epochs} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")
    making sure that its working on the gpu , by viewing the memory usage
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    if (epoch + 1) % 10 == 0:
        vutils.save_image(fake_B, f"generated_{epoch + 1}.png", normalize=True)

torch.save(generator.state_dict(), "generator4.pth")
torch.save(discriminator.state_dict(), "critic4.pth")
