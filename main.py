import click

import torchvision
import torch

def load_dataset():
    data_path = './data/dogs/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize(64),
            torchvision.transforms.RandomCrop(64),
            torchvision.transforms.ToTensor()])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader


if __name__ == '__main__':
    for batch_idx, (data, target) in enumerate(load_dataset()):
        print(data)
