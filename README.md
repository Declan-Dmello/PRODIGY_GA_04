# Pix2Pix with U-Net Generator (Facades Dataset)

This repository contains the implementation of a Pix2Pix model using a U-Net generator and a PatchGAN discriminator, trained on the Facades dataset. The model is capable of image-to-image translation, turning architectural sketches (input) into realistic building facades (output).

## 📦 Dataset

The Facades dataset is downloaded from Kaggle using `kagglehub`. It contains paired images of architectural labels and their corresponding building facades.

**Dataset source:** [Pix2Pix Facades Dataset on Kaggle](https://www.kaggle.com/datasets/sabahesaraki/pix2pix-facades-dataset)

## 🚀 Model Architecture

### Generator (U-Net)
The generator follows the U-Net architecture, consisting of:
- **Encoder**: Down-sampling layers using Conv2D, BatchNorm, and LeakyReLU
- **Decoder**: Up-sampling layers using ConvTranspose2D, BatchNorm, and ReLU
- Final layer uses a Tanh activation function to generate images.

### Discriminator (PatchGAN)
The discriminator is a PatchGAN, classifying each patch in the image as real or fake.


## 🧠 How It Works

The Pix2Pix model operates using a conditional Generative Adversarial Network (cGAN) framework:

1. **Input:** The generator receives an input image (such as a sketch or label map) and tries to generate a realistic output image.
2. **Discriminator:** The discriminator takes both the input and the generated image (or real image) and predicts whether the pair is real or fake.
3. **Loss Functions:**
   - **GAN Loss:** Encourages the generator to produce realistic outputs.
   - **L1 Loss:** Ensures the generated image remains close to the ground-truth image by penalizing pixel-wise differences.
4. **Optimization:** The generator aims to minimize its loss, while the discriminator tries to correctly distinguish between real and fake pairs.

The training process alternates between updating the discriminator and the generator, progressively refining the output quality.

