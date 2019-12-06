import click
import numpy as np

import torchvision
import torch
import matplotlib.pyplot as plt
from transforms import RGB2LAB, LAB2RGB
from PIL import Image

def load_dataset():
    data_path = './data/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.RandomCrop(64),
            torchvision.transforms.ToTensor(),
            RGB2LAB(),
            LAB2RGB()])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader


if __name__ == '__main__':
    i = 0
    for batch_idx, (target, _) in enumerate(load_dataset()):
        target = target.numpy().transpose(0, 2, 3, 1)[0]
        plt.imshow(target)
        plt.show()
        i += 1
        if i == 1:
            break
        
