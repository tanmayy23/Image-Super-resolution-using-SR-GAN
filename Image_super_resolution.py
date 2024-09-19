import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 48 * 48, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Dataset class for SRGAN
class SRGANDataset(Dataset):
    def __init__(self, hr_path, lr_path, transform=None):
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.transform = transform
        self.hr_images = sorted(os.listdir(hr_path))
        self.lr_images = sorted(os.listdir(lr_path))

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, index):
        hr_img = Image.open(os.path.join(self.hr_path, self.hr_images[index]))
        lr_img = Image.open(os.path.join(self.lr_path, self.lr_images[index]))

        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)

        return lr_img, hr_img

# Arguments for path
class Args:
    epochs = 100
    batch_size = 16
    lr = 0.0002
    hr_path = "/home/input/div2k-train-hr-dataset/DIV2K_train_HR"
    lr_path = "/home/input/div2k-train-lr-bicubic/home/tanmay.somkuwar/implementation/chatbot/DIV2K_train_LR_bicubic"
    device = "cuda" if torch.cuda.is_available() else "cpu"

args = Args()

# Image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prepare datasets and dataloaders
train_dataset = SRGANDataset(args.hr_path, args.lr_path, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

# Initialize generator and discriminator
generator = Generator().to(args.device)
discriminator = Discriminator().to(args.device)

# Define loss functions and optimizers
criterion_GAN = nn.BCELoss().to(args.device)  # Binary Cross Entropy Loss for Discriminator
criterion_content = nn.MSELoss().to(args.device)  # MSE Loss for content
optimizer_G = optim.Adam(generator.parameters(), lr=args.lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr)

# Training Loop
g_losses, d_losses = [], []

for epoch in range(args.epochs):
    generator.train()
    discriminator.train()

    g_loss_total, d_loss_total = 0, 0

    for i, (lr_imgs, hr_imgs) in enumerate(tqdm(train_loader)):
        lr_imgs, hr_imgs = lr_imgs.to(args.device), hr_imgs.to(args.device)

        # Generate high-resolution images from low-resolution inputs
        fake_hr_imgs = generator(lr_imgs)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion_GAN(discriminator(hr_imgs), torch.ones_like(discriminator(hr_imgs)))
        fake_loss = criterion_GAN(discriminator(fake_hr_imgs.detach()), torch.zeros_like(discriminator(fake_hr_imgs.detach())))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        gan_loss = criterion_GAN(discriminator(fake_hr_imgs), torch.ones_like(discriminator(fake_hr_imgs)))
        content_loss = criterion_content(fake_hr_imgs, hr_imgs)
        g_loss = content_loss + 1e-3 * gan_loss  # Combined generator loss
        g_loss.backward()
        optimizer_G.step()

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()

    # Record losses
    g_losses.append(g_loss_total / len(train_loader))
    d_losses.append(d_loss_total / len(train_loader))

    print(f"Epoch [{epoch+1}/{args.epochs}]  Generator Loss: {g_losses[-1]:.4f}  Discriminator Loss: {d_losses[-1]:.4f}")

# Save the model weights after training
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

# Plot training losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label="G")
plt.plot(d_losses, label="D")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
