import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load("generator4.pth", map_location=device))
generator.eval()  # Set model to evaluation mode


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image_path = "94.jpg"  
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  


with torch.no_grad():
    output_tensor = generator(input_tensor)

output_tensor = output_tensor.squeeze(0).cpu().detach()  # Remove batch dim
output_image = transforms.ToPILImage()(output_tensor * 0.5 + 0.5)  # Denormalize

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title("Generated Output")
plt.axis("off")

plt.show()

output_image.save("generated_output.jpg")
