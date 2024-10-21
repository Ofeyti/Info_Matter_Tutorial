# PyTorch Conditional Generative Adversarial Network (CGAN) on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

freq=10

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = img_shape[1] // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2))

        self.deconv_blocks = nn.Sequential(
            nn.BatchNorm1d(128 * self.init_size ** 2),
            nn.ReLU(),
            nn.Unflatten(1, (128, self.init_size, self.init_size)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        img = self.deconv_blocks(out)
        return img

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, img_shape[1] * img_shape[2])

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ds_size = img_shape[1] // 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img, labels):
        label_img = self.label_embedding(labels).view(labels.size(0), 1, img.size(2), img.size(3))
        d_in = torch.cat((img, label_img), 1)
        out = self.conv_blocks(d_in)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity

# Initialize model parameters
latent_dim = 100
num_classes = 10
img_shape = (1, 28, 28)

# Instantiate models
generator = Generator(latent_dim, num_classes, img_shape).to(device)
discriminator = Discriminator(num_classes, img_shape).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Optimizers
#optimizer_G = optim.Adam(generator.parameters(), lr=0.00005, betas=(0.25, 0.9999))
#optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.33, 0.99))

# Loss function
adversarial_loss = nn.BCELoss()

# Training
n_epochs = 50
for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(trainloader):

        batch_size = imgs.size(0)
        valid = torch.ones((batch_size, 1), device=device, dtype=torch.float32)
        fake = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)

        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn((batch_size, latent_dim), device=device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        gen_imgs = generator(z, gen_labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Print progress and plot generated samples every 25 batches
        if i % freq == 0:
            print(f"Epoch [{epoch}/{n_epochs}] Batch {i}/{len(trainloader)} Loss D: {d_loss.item():.4f}, loss G: {g_loss.item():.4f}")
        if i % freq == 0:
            
            # Generate and plot samples for each digit
            z = torch.randn(10, latent_dim, device=device)
            labels = torch.arange(0, 10, device=device)
            gen_imgs = generator(z, labels)
            gen_imgs = gen_imgs.cpu().detach().numpy()
            fig, axs = plt.subplots(2, 5, figsize=(10, 5))
            for idx in range(10):
                row, col = divmod(idx, 5)
                axs[row, col].imshow(-gen_imgs[idx].reshape(28, 28), cmap='gray')
                axs[row, col].axis('off')
                axs[row, col].set_title(f"Label: {idx}")
            plt.show()

# Generate final samples
z = torch.randn(10, latent_dim, device=device)
labels = torch.arange(0, 10, device=device)
gen_imgs = generator(z, labels)

# Plot final generated samples
gen_imgs = gen_imgs.cpu().detach().numpy()
fig, axs = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    row, col = divmod(i, 5)
    axs[row, col].imshow(gen_imgs[i].reshape(28, 28), cmap='gray')
    axs[row, col].axis('off')
    axs[row, col].set_title(f"Label: {i}")
plt.show()
