import torch
import torchvision
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import sys
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# Add the current directory to the Python path to find modules
sys.path.append('.')

# Import BackdoorBox components
from core.attacks import BadNets
from cifar10_loader import get_cifar10

# Set a random seed for reproducibility
torch.manual_seed(42)


# Configuration for the stealthy attack
dataset_name = 'cifar10'
attack_name = 'StealthyBadNets'
poisoned_rate = 0.05  # 5% of the training data will be poisoned
y_target = 0    # The target label (airplane in CIFAR-10)

# Create a custom stealthy trigger pattern (subtle circle pattern)
def create_stealthy_trigger(size=32):
    # Create a transparent trigger pattern - must be binary (0 or 1)
    pattern = np.zeros((size, size, 3), dtype=np.float32)

    # Create a circular pattern in the center
    center = size // 2
    radius = size // 4

    # Add a binary circular pattern (1 where trigger should appear)
    for i in range(size):
        for j in range(size):
            # Calculate distance from center
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)

            # Create a thin circular outline
            if radius - 1 <= distance <= radius + 1:
                # Set to 1 for the pattern (binary mask)
                pattern[i, j, :] = 1.0

    # Convert to PyTorch tensor
    pattern_tensor = torch.from_numpy(pattern).permute(2, 0, 1)

    # Create weight tensor - controls intensity (can be between 0 and 1)
    # Make it very subtle by using a small weight value
    weight_tensor = pattern_tensor * 0.03  # Very low intensity for stealth

    return pattern_tensor, weight_tensor


# Define transformation for the images
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Get the CIFAR-10 dataset
train_dataset = get_cifar10(
    root='./data',
    train=True,
    transform=transform_train
)
test_dataset = get_cifar10(
    root='./data',
    train=False,
    transform=transform_test
)

# Create dataloaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples")



# Define a CNN model for CIFAR-10
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Create the model
model = ImprovedCNN()

# Define loss function
criterion = nn.CrossEntropyLoss()


# Generate the stealthy trigger pattern
pattern_tensor, weight_tensor = create_stealthy_trigger(size=32)

# Configure the BadNets attack with our stealthy pattern
badnets = BadNets(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    model=model,
    loss=criterion,
    y_target=y_target,           # Target label to force
    poisoned_rate=poisoned_rate, # Percentage of training data to poison
    pattern=pattern_tensor,      # Our stealthy pattern
    weight=weight_tensor,        # Lower weight for subtle effect
    seed=42,                     # For reproducibility
    deterministic=True           # For reproducibility
)

# Run the attack to create poisoned datasets
poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()

# Create new dataloaders with the poisoned datasets
poisoned_train_loader = DataLoader(
    dataset=poisoned_train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

poisoned_test_loader = DataLoader(
    dataset=poisoned_test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Created poisoned datasets with target class {y_target}")
print(f"Poisoned approximately {int(len(train_dataset) * poisoned_rate)} training samples")



# Find poisoned samples
poisoned_indices = []
num_samples_to_find = 100  # Look through more samples to find good examples

# Scan dataset to find poisoned samples
for i in range(len(train_dataset)):
    if i < len(train_dataset):  # Ensure we don't go out of bounds
        orig_img, orig_label = train_dataset[i]
        pois_img, pois_label = poisoned_train_dataset[i]

        # Check if the image was poisoned
        if torch.sum(torch.abs(orig_img - pois_img)) > 0:
            # Calculate the max difference to find samples with subtle changes
            max_diff = torch.max(torch.abs(orig_img - pois_img)).item()
            # Store the index and the maximum difference
            poisoned_indices.append((i, max_diff))
            if len(poisoned_indices) >= num_samples_to_find:
                break

# Sort by the subtlety of the poisoning (smaller max_diff is more subtle)
poisoned_indices.sort(key=lambda x: x[1])
print(f"Found {len(poisoned_indices)} poisoned samples")



# Function to create an enhanced difference visualization
def create_enhanced_diff_visualization(original, poisoned):
    # Calculate absolute difference
    diff = torch.abs(original - poisoned)

    # Create an RGB visualization where differences are highlighted in cyan
    diff_visualization = torch.zeros_like(original)

    # Amplify the differences dramatically to make them visible
    scaling_factor = 20.0
    diff_visualization[0] = torch.zeros_like(diff[0])  # Red channel = 0
    diff_visualization[1] = diff.mean(dim=0) * scaling_factor  # Green channel
    diff_visualization[2] = diff.mean(dim=0) * scaling_factor  # Blue channel

    # Add the original image at low opacity as background for context
    diff_visualization = diff_visualization + (original * 0.2)

    # Clamp values to [0,1] range
    diff_visualization = torch.clamp(diff_visualization, 0, 1)

    return diff_visualization



# Set up visualization
plt.figure(figsize=(18, 6))

# Choose a sample with subtle poisoning (from the first 10 most subtle examples)
sample_idx = poisoned_indices[5][0]  # Get the 5th most subtle example

# Get original and poisoned samples
orig_img, orig_label = train_dataset[sample_idx]
pois_img, pois_label = poisoned_train_dataset[sample_idx]

# Create enhanced difference visualization
diff_viz = create_enhanced_diff_visualization(orig_img, pois_img)

# Display original, poisoned, and difference images side by side
plt.subplot(1, 3, 1)
plt.imshow(orig_img.permute(1, 2, 0).numpy())
plt.title("Original", fontsize=16, pad=10)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(pois_img.permute(1, 2, 0).numpy())
plt.title("Poisoned Image", fontsize=16, pad=10)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(diff_viz.permute(1, 2, 0).numpy())
plt.title("Difference", fontsize=16, pad=10)
plt.axis('off')

plt.tight_layout(pad=2.0)
plt.savefig('stealthy_backdoor.png', dpi=300, bbox_inches='tight')
print("Saved visualization to stealthy_backdoor.png")



# Calculate poisoning statistics
class_counts = {i: 0 for i in range(10)}  # CIFAR-10 has 10 classes
poisoned_counts = {i: 0 for i in range(10)}

# Count samples in each class
for i in range(len(train_dataset)):
    _, label = train_dataset[i]
    class_counts[label] += 1

# Count poisoned samples by checking for actual modifications
for i in range(len(poisoned_train_dataset)):
    if i < len(train_dataset):  # Ensure we don't go out of bounds
        orig_img, orig_label = train_dataset[i]
        pois_img, pois_label = poisoned_train_dataset[i]

        # Check for changes in the image or label
        if pois_label != orig_label or torch.sum(torch.abs(orig_img - pois_img)) > 0:
            poisoned_counts[orig_label] += 1

# Print statistics
print("\nDistribution of poisoned samples by original class:")
for cls in range(10):
    if class_counts[cls] > 0:
        percent = poisoned_counts[cls]/class_counts[cls]*100
        print(f"Class {cls}: {poisoned_counts[cls]}/{class_counts[cls]} samples poisoned ({percent:.2f}%)")

print(f"\nTotal poisoned samples: {sum(poisoned_counts.values())}")



