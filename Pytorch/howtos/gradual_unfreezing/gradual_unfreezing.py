# @author: Franco Mozo
# @date:   04-04-2023
# @desc:   This script uses a model with layers of a backbone frozen and the last layers unfrozen
#          to train a model. The idea is to train the last layers first and then unfreeze the
#          backbone gradually and train the whole model.
#          Will use this https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#          as a reference for the pretraining
from _init_paths import _init_path

_init_path('Pytorch/from_scratch/src', recursive=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from src.models import LeNet

raise NotImplementedError('This script is not finished yet')

# => Constants
EPOCHS = 10
batch_size = 4
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# => Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# => Model
net = LeNet(in_channels=3)

# => Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# => Train
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d} / {len(trainloader):5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')