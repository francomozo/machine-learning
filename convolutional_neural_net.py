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


    Resources:
    - 1) https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#id1

'''

# Imports
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Create convolutional neural network
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.S2 = self.pool
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.S4 = self.pool
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5))
        self.F6 = nn.Linear(in_features=120, out_features=84)
        self.F7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.C1(x))
        x = self.S2(x)
        x = F.relu(self.C3(x))
        x = self.S4(x)
        x = F.relu(self.C5(x))

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.F6(x))

        x = self.F7(x)

        return x
    
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}.')

# Hyperparameters
num_epochs = 10
lr = 0.004
bs = 512

# Load data
train_dataset = datasets.MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

test_dataset = datasets.MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

# Initialize network and send to device
net = LeNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=lr)

# To check accuracy
def check_accuracy(loader, model, device):
    # -> accuracy = num_corrects / tot_data_points
    model.eval()
    num_corrects = 0
    tot_samples = 0
    
    for x, y in loader:
        x = F.pad(x, (2, 2, 2, 2))

        x, y = x.to(device), y.to(device)
        
        scores = model(x)
        _, preds = scores.max(1)
        num_corrects += torch.sum(preds == y)
        tot_samples += x.shape[0]
    acc = float(num_corrects) / float(tot_samples)
        
    model.train()
    return num_corrects, tot_samples, acc 

# Time management
class TimeMeter():
    def __init__(self):
        self.total_time_elapsed = 0
        self.epoch_time_elapsed = 0
        
    def start_global_timer(self):
        self.total_time_elapsed = perf_counter()

    def end_global_timer(self):
        self.total_time_elapsed = perf_counter() - self.total_time_elapsed

    def get_total_time_elaped(self):
        return self.total_time_elapsed 
    
    def start_epoch_timer(self):
        self.epoch_time_elapsed = perf_counter()

    def end_epoch_timer(self):
        self.epoch_time_elapsed = perf_counter() - self.epoch_time_elapsed

    def get_epoch_time_elaped(self):
        return self.epoch_time_elapsed

time_meter = TimeMeter()

# Train network
time_meter.start_global_timer()
for epoch in range(num_epochs):
    time_meter.start_epoch_timer()
    for idx, (data, targets) in enumerate(train_loader):
        # pad MNIST from 28x28 to 32x32
        data = F.pad(data, (2, 2, 2, 2))
        
        # Move data to available device
        data = data.to(device)
        targets = targets.to(device)
        
        # Zero previous gradients
        optimizer.zero_grad()
        
        # Forward propagate + Loss
        scores = net(data)
        loss = criterion(scores, targets)
        
        # Backpropagate
        loss.backward()
        
        # Take step
        optimizer.step()
    
    time_meter.end_epoch_timer()
    # For each epoch check accuracy on train and test
    print(f'Epoch: {epoch + 1}/{num_epochs} || ', end='')
    with torch.no_grad():
        num_corrects, tot_samples, train_acc = check_accuracy(train_loader, net, device)
        print(f'Train: got {num_corrects}/{int(tot_samples/1000)}K with acc {train_acc:.2f} || ', end='')
        
        num_corrects, tot_samples, test_acc = check_accuracy(test_loader, net, device)
        print(f'Test: got {num_corrects}/{int(tot_samples/1000)}K with acc {test_acc:.2f} || ', end='')
    
    print(f'Time elapsed: {int(time_meter.get_epoch_time_elaped())}s.')    

time_meter.end_global_timer()
print(f'Total time elapsed: {int(time_meter.get_total_time_elaped())}s.')
