# PRODIGY_GA_04
Image to Image translation with C_gan

Pix2Pix with U-Net Generator (Facades Dataset)

This repository contains the implementation of a Pix2Pix model using a U-Net generator and a PatchGAN discriminator, trained on the Facades dataset. The model is capable of image-to-image translation, turning architectural sketches (input) into realistic building facades (output).

📦 Dataset

The Facades dataset is downloaded from Kaggle using kagglehub. It contains paired images of architectural labels and their corresponding building facades.

Dataset source: Pix2Pix Facades Dataset on Kaggle

🚀 Model Architecture

Generator (U-Net)

The generator follows the U-Net architecture, consisting of:

Encoder: Down-sampling layers using Conv2D, BatchNorm, and LeakyReLU

Decoder: Up-sampling layers using ConvTranspose2D, BatchNorm, and ReLU

Final layer uses a Tanh activation function to generate images.

Discriminator (PatchGAN)

The discriminator is a PatchGAN, classifying each patch in the image as real or fake.



🛠️ Requirements

Python 3.8+

PyTorch

torchvision

kagglehub

matplotlib

Pillow

