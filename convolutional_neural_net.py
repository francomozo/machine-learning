'''
    LeNet-5 (the coded version is not strictly the one from the paper)
    (Cx con layer, Sx sub-sampling layers, Fx fully connected layers)
    (Sub-sampling layers is a method of downsampling feature maps (ie: pooling))
    
    Input -> C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> Out
    
    - Input: 32x32 1 channel images 
    - C1:    6 5x5 convolution filters, out features 28x28 (stride=1, padding=0)
    - S2:    out features of size 14x14 (6 feature maps), meaning pooling (kernel=(2, 2))
    - C3:    16 5x5 convolution filters, out features 10x10 (stride=1, padding=0) 
    - S4:    out features of size 5x5 (16 feature maps), meaning pooling (kernel=(2, 2))
    - C5:    120 5x5 convolution filters, out features 1x1 (stride=1, padding=0)
    - F6:    84 dim fully connected layer

    Formula:
    out_features = floor((in_features + 2*padding_size - kernel_size) / (stride_size)) + 1
'''

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets

from src.models import LeNet

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

# Hyperparameters
num_epochs = 10
lr = 0.004
bs = 512

# Load data
modes = ['train', 'val']
transforms = T.Compose([
    T.ToTensor(),
])
datasets = {
    x: datasets.MNIST(root='data/', train=(x == 'train'), download=True, transform=transforms)
    for x in modes
}
dataloaders = {
    x: DataLoader(datasets[x], batch_size=bs, shuffle=(x == 'train'))
    for x in modes
}

# Initialize network and send to device
model = LeNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)

# Train network
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    for phase in modes:
        if phase == 'train':
            model.train()
        else:
            model.eval() 

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = F.pad(inputs, (2, 2, 2, 2))
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                scores = model(inputs)
                loss = criterion(scores, labels)
                
                _, preds = scores.max(1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
                
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print(f'{phase.title()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')    
