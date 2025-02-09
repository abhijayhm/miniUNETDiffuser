import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from datetime import datetime

# U-Net Model Definition
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Downsampling by 2
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Downsampling by 2 again
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # Upsampling by 2
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)  # Additional upsampling
        )
        
    def forward(self, x):
        enc = self.encoder(x)
        mid = self.middle(enc)
        dec = self.decoder(mid)
        return dec

# Diffusion process: adding and removing noise
def forward_diffusion(x, noise_schedule, device):
    noisy_images = []
    for t in range(len(noise_schedule)):
        noise = torch.randn_like(x).to(device) * noise_schedule[t]
        noisy_image = x + noise
        noisy_images.append(noisy_image)
    return noisy_images

def reverse_diffusion(model, noisy_images, noise_schedule, device):
    x = noisy_images[-1]
    for t in reversed(range(len(noise_schedule))):
        model_output = model(x)
        x = noisy_images[t] - model_output
    return x

# Load Image
class CustomDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        
    def __len__(self):
        return 1  # Only one image for training
    
    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Training setup
def train(model, dataloader, optimizer, epochs, noise_schedule, device):
    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            image = data[0].to(device)
            
            noisy_images = forward_diffusion(image, noise_schedule, device)
            optimizer.zero_grad()
            
            # Reverse diffusion with U-Net
            denoised_image = reverse_diffusion(model, noisy_images, noise_schedule, device)
            
            # Loss calculation (MSE loss)
            loss = nn.MSELoss()(denoised_image, image)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def test(model, dataset, noise_schedule, device):
    model.eval()
    os.makedirs("outputs/noisy", exist_ok=True)  # Ensure the noisy image folder exists

    with torch.no_grad():
        image = dataset[0].to(device).unsqueeze(0)  # Add batch dimension
        noisy_images = forward_diffusion(image, noise_schedule, device)
        
        # Save noisy images
        for idx, noisy_image in enumerate(noisy_images):
            noisy_image_np = noisy_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Remove batch dim and convert to HWC
            noisy_image_np = np.clip(noisy_image_np * 255, 0, 255).astype(np.uint8)
            noisy_image_path = f"outputs/noisy/noise_{idx+1}.jpg"
            Image.fromarray(noisy_image_np).save(noisy_image_path)

        # Perform reverse diffusion
        denoised_image = reverse_diffusion(model, noisy_images, noise_schedule, device)

        # Save the denoised image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        denoised_image_np = denoised_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        denoised_image_np = np.clip(denoised_image_np * 255, 0, 255).astype(np.uint8)
        output_path = f"outputs/output-{timestamp}.jpg"
        Image.fromarray(denoised_image_np).save(output_path)
        print(f"Denoised image saved to {output_path}")

if __name__ == "__main__":    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))  # Resize image for training
    ])

    # Image path
    image_path = "inputs/input.jpg"

    # Hyperparameters
    epochs = 1000
    steps = 5
    noise_schedule = np.linspace(0, 1, steps).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = CustomDataset(image_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize U-Net, optimizer
    model = UNet(in_channels=3, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if saved weights exist
    weights_path = 'model_weights.pth'

    if os.path.exists(weights_path):
        user_input = input("Pre-trained weights found. Do you want to use them for testing (y/n)? ").strip().lower()
        if user_input == 'y':
            # Load saved weights and test the model
            model.load_state_dict(torch.load(weights_path))
            test(model, dataset, noise_schedule, device)
        else:
            print("Proceeding to train the model from scratch.")
            train(model, dataloader, optimizer, epochs, noise_schedule, device)
            torch.save(model.state_dict(), weights_path)
            print(f"Model trained and weights saved to {weights_path}")
            test(model, dataset, noise_schedule, device)
    else:
        print("No pre-trained weights found. Training the model from scratch.")
        train(model, dataloader, optimizer, epochs, noise_schedule, device)
        torch.save(model.state_dict(), weights_path)
        print(f"Model trained and weights saved to {weights_path}")
        test(model, dataset, noise_schedule, device)
