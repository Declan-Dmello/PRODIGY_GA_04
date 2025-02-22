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

🔧 Training

Steps to train the model:

Install dependencies:

pip install torch torchvision kagglehub matplotlib pillow

Download the dataset:

import kagglehub
path = kagglehub.dataset_download("sabahesaraki/pix2pix-facades-dataset")

Run the training script:

python train.py

Training configuration:

Batch size: 16

Image size: 256x512 (paired images)

Epochs: 150

Optimizer: Adam (lr=0.0002, betas=(0.5, 0.999))

Loss functions:

GAN Loss: Binary Cross Entropy Loss (BCE)

L1 Loss: Mean Absolute Error (MAE) for pixel-wise difference

Model Checkpoints:

Generator: generator4.pth

Discriminator: critic4.pth

📈 Inference

To generate images using the trained model:

Load the generator:

generator.load_state_dict(torch.load("generator4.pth"))
generator.eval()

Test with an image:

python generate.py --input_image "94.jpg"

Result: The generated image will be saved as generated_output.jpg


🛠️ Requirements

Python 3.8+

PyTorch

torchvision

kagglehub

matplotlib

Pillow

