import torch
import torchvision
from transforms import RGB2LAB

def load_normalized_dataset():
    data_path = './data/'
    normalized_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.RandomCrop(64),
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=3),
            torchvision.transforms.ToTensor(),
            RGB2LAB(),
            torchvision.transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
            torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 10), value=0, inplace=False),
            ])
    )
    normalized_loader = torch.utils.data.DataLoader(
        normalized_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return normalized_loader

def load_real_dataset():
    data_path = './data/'
    real_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.RandomCrop(64),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            RGB2LAB()])
    )
    real_loader = torch.utils.data.DataLoader(
        real_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return real_loader
