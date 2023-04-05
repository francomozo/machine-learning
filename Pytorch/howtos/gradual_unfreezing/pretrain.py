"""
@author: Franco Mozo
@date:   05-04-2023
@desc:   Will use this https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html to pretrain on CIFAR10
"""

from _init_paths import _init_path

_init_path('Pytorch/from_scratch/src', recursive=True)

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from src.models import LeNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from utils import BestModel, check_accuracy

# => Constants
EPOCHS = 2000
BATCH_SIZE = 4
LR = 1e-4
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
MODEL_NAME = Path('lenet_pretrained_cifar.pth')
OUTDIR = Path('exps/pretraining')

# => Paths
# Create the output directory
OUTDIR.mkdir(exist_ok=True)
prev_exps = [str(x.stem) for x in OUTDIR.iterdir()]
if prev_exps:
    exp_num = int(sorted(list(prev_exps), key=lambda x: int(x))[-1]) + 1
else:
    exp_num = 1
OUTDIR = OUTDIR / str(exp_num)
OUTDIR.mkdir(exist_ok=True)

# => Tensorboard writer and device and best model
writer = SummaryWriter(log_dir=OUTDIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = BestModel(pt_path=OUTDIR / MODEL_NAME)

# => Seeds
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# => Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainloader = DataLoader(
    CIFAR10(root='./data', train=True, download=True, transform=transform), 
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=2
)
testloader = DataLoader(
    CIFAR10(root='./data', train=False, download=True, transform=transform), 
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=2
)

# => Model
net = LeNet(in_channels=3).to(device)

# => Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# => Train
for epoch in tqdm(range(EPOCHS), desc='Epochs', leave=False):
    epoch += 1
    
    running_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(trainloader), desc='Batches', total=len(trainloader), leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

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
            writer.add_scalar('training loss', running_loss / 2000, epoch * len(trainloader) + i)       
            running_loss = 0.0
    
    test_acc = check_accuracy(testloader, net, device)
    writer.add_scalar('test accuracy', test_acc, epoch)
      
    # Store best model for saving
    best_model.compare_and_store(net, test_acc, epoch)
    
    if epoch % 10 == 0:
        continue_training = input('Continue training? [y/n]: ')
        if continue_training == 'n':
            break

print('Finished Training')
best_model.save()

