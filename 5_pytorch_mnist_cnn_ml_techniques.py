# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:40:06 2024

@author: gahlmann
"""

# PyTorch MNIST Classifier
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import numpy as np
import seaborn as sns
import os
import time
total_start_time=time.time()
# Load dataset
print("loading data")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
print("Done")

# Define model
class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
model = ConvNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Starting training...")
losses = []
num_batches = len(trainloader)
epochs = 2

# Start timer
total_start_time = time.time()

total_steps = epochs * num_batches  # Total training steps for all epochs
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(trainloader):
        # Decrease noise factor with each batch
        step = epoch * num_batches + batch_idx
        noise_factor = max(0,  0.2* (1.0 -step*3 / total_steps) )
        
        images, labels = images.to(device), labels.to(device)
        noise = noise_factor * torch.randn_like(images).to(device)
        images = images + noise
        images = torch.clamp(images, 0., 1.)  # Ensure pixel values are in valid range
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:  # Print every 100 batches
            avg_batch_loss = running_loss / 100
            sys.stdout.write(f"\rEpoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_batch_loss:.4f}")
            sys.stdout.flush()
            losses.append(avg_batch_loss)
            running_loss = 0.0

            # Plot loss evolution after every 100 batches
            plt.figure()
            plt.plot(range(1, len(losses) + 1), losses, marker='o')
            plt.xlabel('Batch (per 100)')
            plt.ylabel('Loss')
            plt.title('Training Loss Evolution')
            plt.show()
    print(f"\nFinished Epoch {epoch + 1}")

# Test model
print("Starting testing...")
model.eval()
correct = 0
total = 0
conf_matrix = np.zeros((10, 10), dtype=int)
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            conf_matrix[t, p] += 1

print(f"PyTorch Test Accuracy: {100 * correct / total:.2f}%")

# Normalize confusion matrix
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='binary', fmt='.2f', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print total run time
total_end_time = time.time()
total_run_time = total_end_time - total_start_time
print(f"Total run time: {total_run_time:.2f} seconds")
os.system('echo -e "\a"')