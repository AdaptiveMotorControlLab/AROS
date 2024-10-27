import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from aros_node import (
    LabelChangedDataset,
    get_subsampled_subset,
    get_loaders,
)

# Set up transformations and datasets for tests
transform_tensor = ToTensor()

@pytest.fixture
def cifar10_datasets():
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_tensor)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_tensor)
    return trainset, testset

@pytest.fixture
def cifar100_datasets():
    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_tensor)
    testset = CIFAR100(root='./data', train=False, download=True, transform=transform_tensor)
    return trainset, testset

def test_label_changed_dataset(cifar10_datasets):
    _, testset = cifar10_datasets
    new_label = 99
    relabeled_dataset = LabelChangedDataset(testset, new_label)
    
    assert len(relabeled_dataset) == len(testset), "Relabeled dataset should match the original dataset length"
    
    for img, label in relabeled_dataset:
        assert label == new_label, "All labels should be changed to the new label"

def test_get_subsampled_subset(cifar10_datasets):
    trainset, _ = cifar10_datasets
    subset_ratio = 0.1
    subset = get_subsampled_subset(trainset, subset_ratio=subset_ratio)
    
    expected_size = int(len(trainset) * subset_ratio)
    assert len(subset) == expected_size, f"Subset size should be {expected_size}"

def test_get_loaders_cifar10(cifar10_datasets):
    train_loader, test_loader, test_set, test_loader_vs_other = get_loaders('cifar10')
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert isinstance(test_loader_vs_other, DataLoader)

    for images, labels in test_loader:
        assert images.shape[0] == 16, "Test loader batch size should be 16"
        break

def test_get_loaders_cifar100(cifar100_datasets):
    train_loader, test_loader, test_set, test_loader_vs_other = get_loaders('cifar100')
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert isinstance(test_loader_vs_other, DataLoader)

    for images, labels in test_loader:
        assert images.shape[0] == 16, "Test loader batch size should be 16"
        break

def test_get_loaders_invalid_dataset():
    with pytest.raises(ValueError, match="Dataset 'invalid_dataset' is not supported."):
        get_loaders('invalid_dataset')
