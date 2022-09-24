'''
    Defining a fully connected network with N+1 layers (N hidden layers + 1 output layer)
    with hidden layers with H hidden units
    For this script N = 1
    Trained on MNIST dataset 
    
    Resources:
    - 1)  https://discuss.pytorch.org/t/super-init-vs-super-classname-self-init/148793
    - 2)  https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    - 3)  https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    - 4)  https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
    - 5)  https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    - 6)  https://pytorch.org/docs/stable/optim.html
    - 7)  https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    - 8)  https://pytorch.org/docs/stable/nn.html#loss-functions
    - 9)  https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
    - 10) https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
'''

# Imports
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Create fully connected network
class Net(nn.Module):
    def __init__(self, input_size, num_classes, hidden_units):
        super().__init__() # 1)
        self.fc1 = nn.Linear(input_size, hidden_units) # 2)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}.')


# Hyperparameters
input_size = 784 # MNIST dataset has 28x28=784 sized images, 60k for train and 10k for test   
num_classes = 10 # 10 digits
hidden_units = 512
num_epochs = 10
lr = 0.003
bs = 512

# Load data
train_dataset = datasets.MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor()) # 4)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)# 5)

test_dataset = datasets.MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

# Initialize network and send to device
net = Net(input_size, num_classes, hidden_units).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # 8) From doc: It is useful when training a classification 
                                  #              problem with C classes
optimizer = optim.Adam(params=net.parameters(), lr=lr) # 6) 7)

# To check accuracy
def check_accuracy(loader, model, device):
    # -> accuracy = num_corrects / tot_data_points
    model.eval()
    num_corrects = 0
    tot_samples = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.reshape(x.shape[0], -1)
        
        scores = model(x) # returns torch.Size([batch_size, 10]), need to take the max
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
        
        # Reshape data for fully connected net
        # -> Since data are images, shapes are: torch.Size([64, 1, 28, 28]) torch.Size([64])
        # -> Fully connected nets take one dimensional data (aside from the batch size)
        # -> I want shape [batch_size, 28x28]
        data = data.reshape(data.shape[0], -1)
        
        # Zero previous gradients
        # -> 9)  You can also use model.zero_grad(). This is the same as using optimizer.zero_grad() as 
        #        long as all your model parameters are in that optimizer.
        # -> 10) By default gradients are accumulated on subsequent backward passes (convenient in RNNs 
        #        for example). In other cases I want to set gradients to zero so parameters update
        #        correctly.
        optimizer.zero_grad()
        
        # Forward propagate + Loss
        # -> After propagating the data through the net I get shape torch.Size([64, 10]) 
        # -> Regarding the loss I get a number corresponding to applying the CrossEntropy fn 
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
