import click
import numpy as np

import torchvision
import torch
import matplotlib.pyplot as plt
from dataloaders import load_normalized_dataset, load_real_dataset



if __name__ == '__main__':
    i = 0
    for batch_idx, (target, _) in enumerate(load_normalized_dataset()):
        print(target.numpy().transpose(0, 2, 3, 1)[0])
        i += 1
        if i == 1:
            break
    i = 0
    for batch_idx, (target, _) in enumerate(load_real_dataset()):
        print(target.numpy().transpose(0, 2, 3, 1)[0])
        i += 1
        if i == 1:
            break
        
