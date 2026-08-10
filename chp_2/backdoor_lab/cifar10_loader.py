import torchvision.datasets.cifar as cifar_mod
from torchvision.datasets import CIFAR10

# Alternative mirror for the official CIFAR-10 tarball (same file, same MD5)
cifar_mod.CIFAR10.url = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"


def get_cifar10(root='./data', train=True, download=True, transform=None, target_transform=None):
    return CIFAR10(
        root=root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )