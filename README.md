# Mini UNet Diffuser

A simple U-Net-based image denoising model using a diffusion process in PyTorch.

## Overview
This project aims to understand how diffusion models interact with images and learn them. By using a single image as training data, we ensure fast training and gain better insight into what happens at each step of the diffusion process.

## How It Works
1. **Forward Diffusion:** Noise is added to the image progressively using a noise schedule.
2. **Training:** The U-Net model learns to reverse this noise by adjusting weights via backpropagation and an optimizer.
3. **Reverse Diffusion:** During testing, the trained model attempts to regenerate the original image from the noisy versions.
4. **Goal:** Understand how diffusion models process images and reconstruct them through learned noise-removal techniques.

## Setup and Running the Model
```sh
# Clone the repository
git clone https://github.com/abhijayhm/miniUNETDiffuser.git
cd miniUNETDiffuser

# Install dependencies
pip install -r requirements.txt

# Run the script
python miniUNETDiffuser.py
```

## Dependencies
Ensure you have Python 3.7+ installed. The required Python libraries are listed in `requirements.txt`.

## Expected Output
- The model will train on a single input image.
- Noisy images will be saved in `outputs/noisy/`.
- The final denoised image will be saved in `outputs/`.
- If pre-trained weights exist, you will have the option to test directly.

## Notes
- The U-Net model is designed to progressively denoise images.
- Using a single image dataset helps in rapid experimentation.
- The approach is a simplified version of how modern diffusion models work in deep learning.

