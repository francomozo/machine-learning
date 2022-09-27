'''
    Modularized version of a script to train a 2 layer neural network
    Go to git history for commented single script + resources (first commit)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.metrics import check_accuracy
from src.models import SimpleNN
from src.utils.common import TimeMeter

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
train_dataset = datasets.MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

test_dataset = datasets.MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

# Initialize network and send to device
model = SimpleNN(input_size, num_classes, hidden_units).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)

time_meter = TimeMeter()

# Train network
time_meter.start_global_timer()
for epoch in range(num_epochs):
    time_meter.start_epoch_timer()
    for idx, (data, targets) in enumerate(train_loader):
        # Move data to available device
        data = data.to(device)
        targets = targets.to(device)
        
        # Reshape data for fully connected net
        data = data.reshape(data.shape[0], -1)
        
        # Zero previous gradients
        optimizer.zero_grad()
        
        # Forward propagate + Loss
        scores = model(data)
        loss = criterion(scores, targets)
        
        # Backpropagate
        loss.backward()
        
        # Take step
        optimizer.step()
    
    time_meter.end_epoch_timer()

    # For each epoch check accuracy on train and test
    print(f'Epoch: {epoch + 1}/{num_epochs} || ', end='')
    with torch.no_grad():
        num_corrects, tot_samples, train_acc = check_accuracy(train_loader, model, device)
        print(f'Train: got {num_corrects}/{int(tot_samples/1000)}K with acc {train_acc:.2f} || ', end='')
        
        num_corrects, tot_samples, test_acc = check_accuracy(test_loader, model, device)
        print(f'Test: got {num_corrects}/{int(tot_samples/1000)}K with acc {test_acc:.2f} || ', end='')
    
    print(f'Time elapsed: {int(time_meter.get_epoch_time_elaped())}s.')    

time_meter.end_global_timer()
print(f'Total time elapsed: {int(time_meter.get_total_time_elaped())}s.')
