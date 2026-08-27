import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import LinearSegmentedColormap
import cv2

# Add the current directory to the Python path
sys.path.append('.')

# Import BackdoorBox components
from core.attacks import BadNets
from cifar10_loader import get_cifar10

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
dataset_name = 'cifar10'
attack_name = 'BadNets'
poisoned_rate = 0.05  # 5% of training data will be poisoned
y_target = 0          # Target label (airplane in CIFAR-10)
trigger_size = 3      # Size of the trigger pattern (3x3 pixels)




# Dataset setup
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Get the CIFAR-10 dataset
train_dataset = get_cifar10(
    root='./data',
    train=True,
    transform=transform
)

test_dataset = get_cifar10(
    root='./data',
    train=False,
    transform=transform
)

# Define a simple model for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model and define loss function
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()




# Configure the BadNets attack
badnets = BadNets(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    model=model,
    loss=criterion,
    y_target=y_target,
    poisoned_rate=poisoned_rate,
    pattern=None,
    weight=None,
    seed=42,
    deterministic=True
)

# Get poisoned datasets
poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()




# Find poisoned samples by comparing original and poisoned images
poisoned_indices = []
# We'll look for up to 5 poisoned samples
num_samples_to_find = 5

for i in range(len(train_dataset)):
    orig_img, orig_label = train_dataset[i]
    pois_img, pois_label = poisoned_train_dataset[i]

    # Check if the image was poisoned (has been modified)
    if torch.sum(torch.abs(orig_img - pois_img)) > 0:
        poisoned_indices.append(i)
        if len(poisoned_indices) >= num_samples_to_find:
            break

print(f"Found {len(poisoned_indices)} poisoned samples for visualization")




# Set up the visualization
# Number of visualization techniques we'll use
num_techniques = 6
num_samples = len(poisoned_indices)

# Create figure with subplots
fig, axes = plt.subplots(num_samples, num_techniques, figsize=(18, 3*num_samples))

# Define a custom colormap for the heatmap (red for positive differences)
diff_cmap = LinearSegmentedColormap.from_list('diff_cmap', ['black', 'yellow', 'red'])

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']





# Process each poisoned sample
for row, idx in enumerate(poisoned_indices):
    # Get original and poisoned images
    orig_img, orig_label = train_dataset[idx]
    pois_img, pois_label = poisoned_train_dataset[idx]

    # Convert to numpy arrays for processing
    orig_np = orig_img.permute(1, 2, 0).numpy()
    pois_np = pois_img.permute(1, 2, 0).numpy()

    # 1. Original image
    axes[row, 0].imshow(orig_np)
    axes[row, 0].set_title(f"Original: {class_names[orig_label]}")
    axes[row, 0].axis('off')

    # 2. Poisoned image
    axes[row, 1].imshow(pois_np)
    axes[row, 1].set_title(f"Poisoned: {class_names[pois_label]}")
    axes[row, 1].axis('off')

    # 3. Enhanced difference visualization
    diff = np.abs(pois_np - orig_np)
    # Amplify differences to make them more visible
    enhanced_diff = np.power(diff * 5, 2)
    # Normalize to [0, 1] range
    if enhanced_diff.max() > 0:
        enhanced_diff = enhanced_diff / enhanced_diff.max()

    axes[row, 2].imshow(enhanced_diff)
    axes[row, 2].set_title("Enhanced Difference")
    axes[row, 2].axis('off')

    # 4. Heatmap of differences
    # Convert to grayscale for heatmap (sum across channels)
    diff_gray = np.sum(diff, axis=2)
    axes[row, 3].imshow(diff_gray, cmap=diff_cmap)
    axes[row, 3].set_title("Difference Heatmap")
    axes[row, 3].axis('off')

    # 5. Zoomed view of trigger region
    # Find the region with the maximum difference
    if diff_gray.max() > 0:
        # Get indices of maximum difference
        trigger_y, trigger_x = np.unravel_index(diff_gray.argmax(), diff_gray.shape)

        # Define zoom region (centered on detected trigger)
        zoom_size = 8  # Size of zoom window
        y_min = max(0, trigger_y - zoom_size//2)
        y_max = min(pois_np.shape[0], trigger_y + zoom_size//2)
        x_min = max(0, trigger_x - zoom_size//2)
        x_max = min(pois_np.shape[1], trigger_x + zoom_size//2)

        # Extract zoomed regions
        zoomed_pois = pois_np[y_min:y_max, x_min:x_max]

        # Show zoomed poisoned image
        axes[row, 4].imshow(zoomed_pois)
        axes[row, 4].set_title("Zoomed Trigger Region")
        axes[row, 4].axis('off')
    else:
        axes[row, 4].text(0.5, 0.5, "No significant difference",
                       horizontalalignment='center', verticalalignment='center')
        axes[row, 4].axis('off')

    # 6. Highlighted trigger in original image
    highlighted = orig_np.copy()

    # Create a mask where differences are significant
    mask_2d = np.sum(diff, axis=2) > 0.05  # Threshold to detect trigger

    # Apply red highlighting to each channel separately
    highlighted[mask_2d, 0] = 1.0  # Red channel - set to max
    highlighted[mask_2d, 1] = 0.0  # Green channel - set to min
    highlighted[mask_2d, 2] = 0.0  # Blue channel - set to min

    axes[row, 5].imshow(highlighted)
    axes[row, 5].set_title("Highlighted Trigger")
    axes[row, 5].axis('off')

# Save the visualization
plt.tight_layout()
plt.savefig('enhanced_trigger_visualization.png', dpi=300)
print("Saved detailed visualization to enhanced_trigger_visualization.png")




# Analyze the first poisoned sample in detail
if len(poisoned_indices) > 0:
    # Use the first poisoned sample for analysis
    idx = poisoned_indices[0]
    orig_img, _ = train_dataset[idx]
    pois_img, _ = poisoned_train_dataset[idx]

    # Convert to numpy arrays
    orig_np = orig_img.permute(1, 2, 0).numpy()
    pois_np = pois_img.permute(1, 2, 0).numpy()

    # Calculate difference
    diff = np.abs(pois_np - orig_np)

    # Identify trigger pixels (where difference is significant)
    trigger_mask = np.sum(diff, axis=2) > 0.05

    # Count trigger pixels and calculate size
    trigger_pixels = np.sum(trigger_mask)
    image_pixels = orig_np.shape[0] * orig_np.shape[1]
    trigger_percentage = (trigger_pixels / image_pixels) * 100

    # Find trigger location
    if trigger_pixels > 0:
        y_indices, x_indices = np.where(trigger_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        # Calculate trigger dimensions
        trigger_height = y_max - y_min + 1
        trigger_width = x_max - x_min + 1

        # Calculate average intensity of trigger
        # Create a 3D mask for indexing the diff array
        mask_3d = trigger_mask[:, :, np.newaxis].repeat(3, axis=2)
        trigger_intensity = np.mean(diff[mask_3d])

        print(f"\\nTrigger Analysis:")
        print(f"- Trigger size: {trigger_pixels} pixels ({trigger_percentage:.2f}% of image)")
        print(f"- Trigger dimensions: {trigger_width}x{trigger_height} pixels")
        print(f"- Trigger location: ({x_min},{y_min}) to ({x_max},{y_max})")
        print(f"- Average intensity: {trigger_intensity:.4f}")
    else:
        print("No significant trigger detected")



