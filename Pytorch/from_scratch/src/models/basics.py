import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_units):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x


class LeNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        
        self.C1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5, 5))
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
