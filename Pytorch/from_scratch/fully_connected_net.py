'''
    Modularized version of a script to train a 2 layer neural network
    Go to git history for commented single script + resources (first commit)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import SimpleNN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

# Hyperparameters
input_size = 784 # MNIST: 28x28=784 sized images  
num_classes = 10
hidden_units = 512
num_epochs = 20
lr = 0.004
bs = 512

# Load data
modes = ['train', 'val']
datasets = {
    x: datasets.MNIST(root='data/', train=(x == 'train'), download=True, transform=transforms.ToTensor())
    for x in modes
}
dataloaders = {
    x: DataLoader(datasets[x], batch_size=bs, shuffle=(x == 'train'))
    for x in modes
}

# Initialize network and send to device
model = SimpleNN(input_size, num_classes, hidden_units).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)

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
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Reshape data for fully connected net
            inputs = inputs.reshape(inputs.shape[0], -1)

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

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')    
