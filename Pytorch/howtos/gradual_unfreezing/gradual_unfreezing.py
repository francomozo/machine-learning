"""
@author: Franco Mozo
@date:   06-04-2023
@desc:   This script allows to finetune a pretrained model (trained with pretrain.py) on a new dataset while unfreezing the  .
"""

from _init_paths import _init_path

_init_path('Pytorch/from_scratch/src', recursive=True)

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from src.models import LeNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import Animals10Dataset, BestModel, check_accuracy

# => Constants
OUTDIR = Path('exps/gradual_unfreezing')
MODEL_NAME = Path('finetuned_lenet_animal10.pth')
PRETRAINED_MODEL = Path('exps/pretraining/3/lenet_pretrained_cifar.pth')

# This config is getting out of hand
IMAGE_SIZE = 32
PRINT_N_PER_EP = 5
BATCH_SIZE = 128
EPOCHS = 120
OPTIMIZER = 'AdamW'
LR = 0.0002

FREEZE_LAYERS = ['C1', 'C3', 'C5'] # Initially freeze the backbone
UNFREEZE_EPS = [90, 70, 50] # Unfreeze the backbone in 3 steps (C5 in epoch 20, C3 in epoch 40, C1 in epoch 50)
LRS = [0.001, 0.0008, 0.0005] # Learning rates for C1, C2 and C3 respectively
LRS_BY_LAYER = True

EXP_DESC = f'Gradual_unfreezing'

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

# => Save config
exp_data = {
    'lr': LR,
    'optimizer': OPTIMIZER,
    'bz': BATCH_SIZE,
    'eps': EPOCHS,
    'freeze_layers': FREEZE_LAYERS,
    'unfreeze_eps': UNFREEZE_EPS,
    'lrs': LRS,
    'lrs_by_layer': LRS_BY_LAYER,
    'desc': EXP_DESC
}
with OUTDIR.joinpath('hyp.json').open('w') as fp:
    json.dump(exp_data, fp)

print(f'Starting experiment {str(exp_num)}')

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
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainloader = DataLoader(
    Animals10Dataset(root='data/animals10/splits', train=True, transforms=transform), 
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=2
)
testloader = DataLoader(
    Animals10Dataset(root='data/animals10/splits', train=False, transforms=transform), 
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=2
)

# => Print each
print_each = int(len(trainloader) / PRINT_N_PER_EP)

# => Model
net = LeNet(in_channels=3).to(device)
net.load_state_dict(torch.load(PRETRAINED_MODEL)['state_dict'])

# => Freeze backbone
# Only C1, C3 and C5 are trainable
for name, param in net.named_parameters():
    if any([x in name for x in FREEZE_LAYERS]):
        param.requires_grad = False

# => Loss and optimizer
criterion = nn.CrossEntropyLoss()
if OPTIMIZER == 'AdamW':
    if LRS_BY_LAYER:
        optimizer = optim.AdamW(
            [
                {'params': net.C1.parameters(), 'lr': LRS[0]},
                {'params': net.C3.parameters(), 'lr': LRS[1]},
                {'params': net.C5.parameters(), 'lr': LRS[2]},
                {'params': net.F6.parameters()},
                {'params': net.F7.parameters()}
            ], 
            lr=LR
        )
    else:
        optimizer = optim.AdamW(net.parameters(), lr=LR)
elif OPTIMIZER == 'SGD':
    if LRS_BY_LAYER:
        optimizer = optim.SGD(
            [
                {'params': net.C1.parameters(), 'lr': LRS[0]},
                {'params': net.C3.parameters(), 'lr': LRS[1]},
                {'params': net.C5.parameters(), 'lr': LRS[2]},
                {'params': net.F6.parameters()},
                {'params': net.F7.parameters()}
            ], 
            lr=LR, 
            momentum=0.9
        )
    else:
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
else:
    raise ValueError(f'Optimizer {OPTIMIZER} not implemented')

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
        if i % print_each == print_each - 1:    # print every print_each mini-batches
            writer.add_scalar('training loss', running_loss / print_each, epoch * len(trainloader) + i)    
            running_loss = 0.0
        
        # Add scalars for each layer: mean and std
        for name, param in net.named_parameters():
            if any([x in name for x in FREEZE_LAYERS]):
                writer.add_scalar(f'Backbone/{name}_mean', param.data.mean(), epoch * len(trainloader) + i)
                writer.add_scalar(f'Backbone/{name}_std', param.data.std(), epoch * len(trainloader) + i)
            elif any([x in name for x in ['F6', 'F7']]):
                writer.add_scalar(f'Head/{name}_mean', param.data.mean(), epoch * len(trainloader) + i)
                writer.add_scalar(f'Head/{name}_std', param.data.std(), epoch * len(trainloader) + i)
    
    if epoch in UNFREEZE_EPS:
        # Get the index of the epoch to unfreeze
        idx = UNFREEZE_EPS.index(epoch)
        layer2unfreeze = FREEZE_LAYERS[idx]
        for name, param in net.named_parameters():
            if any([x in name for x in [layer2unfreeze]]):
                param.requires_grad = True
        print(f'Unfreezing {layer2unfreeze}')
    
    test_acc = check_accuracy(testloader, net, device)
    writer.add_scalar('test accuracy', test_acc, epoch)
    
    # Store best model for saving
    best_model.compare_and_store(net, test_acc, epoch)

print('Finished Training')
best_model.save()