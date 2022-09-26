'''
    The CNN is the LeNet
    Since in the original paper the model takes a 32x32 image, I will use a padded MNIST dataset 
    LeNet-5 (Taken from the paper)
        obs: Cx con layer, Sx sub-sampling layers, Fx fully connected layers)
             Sub-sampling layers is a method of downsampling feature maps (ie: pooling)
        Input -> C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> Out
    - Input: 32x32 1 channel images 
    - C1:    6 5x5 convolution filters with 28x28 feature maps as outputs
    - S2:    outputs 6 features maps of size 14x14, meaning pooling of size 2x2
    - C3:    outputs 16 feature maps of size 10x10, 5x5 convolution filters
    - S4:    outputs 16 feature maps of size 5x5, meaning pooling of 2x2
    
    conv in_channels=1, 
    
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
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}.')

# Hyperparameters
num_classes = 10
hidden_units = 512
num_epochs = 20
lr = 0.004
bs = 512

# Load data
train_dataset = datasets.MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

test_dataset = datasets.MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

# Initialize network and send to device
net = CNN(input_size, num_classes, hidden_units).to(device)

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
        x, y = x.to(device), y.to(device)
        x = x.reshape(x.shape[0], -1)
        
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
