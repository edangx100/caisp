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