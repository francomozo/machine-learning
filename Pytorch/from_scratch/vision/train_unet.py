import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import UNet
from src.utils import binarize_map
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# => Parameters
N_CHANNELS = 3
N_CLASSES = 1 # 1 for binary segmentation
BS = 1
LR=0.001
SPLIT = 0.8
MODES = ['train', 'val']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {DEVICE}')

# Initialize network and send to device
unet = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)

# Load data
dataset = datasets.OxfordIIITPet(
    root='data/', 
    download=True, 
    target_types='segmentation', 
    transform=transforms.ToTensor(), 
    target_transform=transforms.ToTensor()
)
# https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
train_size = int(SPLIT * len(dataset))
val_size = len(dataset) - train_size
datasets = random_split(dataset, [train_size, val_size])

datasets = {
    x: datasets[i]
    for i, x in enumerate(MODES)
}
dataloaders = {
    x: DataLoader(datasets[x], batch_size=BS)#, shuffle=(x == 'train'))
    for x in MODES
}

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=unet.parameters(), lr=LR)

for idx, (inputs, maps) in enumerate(dataloaders['train']):
    inputs = inputs.to(DEVICE)
    maps = binarize_map(maps).to(DEVICE)

    scores = unet(inputs)
    loss = criterion(scores, maps)

    loss.backward()
    optimizer.step()

    # each 100 iterations, print loss
    if idx % 100 == 0:
        # show scores map
        scores = scores.squeeze().detach().cpu().numpy()
        scores = np.expand_dims(scores, axis=2)
        scores = cv2.cvtColor(scores, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', scores)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





    # maps = maps.squeeze().numpy()
    # maps = np.expand_dims(maps, axis=2)
    # for i in range(len(values)):
    #     maps = np.where(maps == values[i], colors[i], maps)

    # maps = cv2.cvtColor(maps, cv2.COLOR_RGB2BGR)
    # cv2.imshow('image', maps)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # inputs = inputs.squeeze().permute(1, 2, 0).numpy() 
    # inputs = cv2.cvtColor(inputs, cv2.COLOR_RGB2BGR)
    # cv2.imshow('image', inputs)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()