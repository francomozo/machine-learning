import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F


class DoubleConv(nn.Module):
    """ (conv => BN => ReLU) * 2 """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class Down(nn.Module):
    """ red -> blue -> blue arrows (UNet diagram)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # green arrow (UNet diagram)
        # out_channels = in_channels // 2 because we are concatenating the skip connection
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2): #x5x4
        """ x2 is the skip connection """

        x1 = self.up(x1)
        # The papers implements cropping the features map from the skip connection
        # In order to output an image of the same size as the input image, I will
        # resize the x1 feature map to the same size as x2
        if x1.size()[2:] != x2.size()[2:]:
            x1 = F.resize(x1, size=x2.size()[2:])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """ light blue arrow (UNet diagram))"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

""" Note:
For the first implementation I did conv-conv-maxpool for the Down class,
but this does not allow me to use the output of the class as the skip 
connection. A better implementation is to use maxpool-conv-conv. 
"""
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters

        self.inc = DoubleConv(n_channels, n_filters)
        self.down1 = Down(n_filters, n_filters * 2)
        self.down2 = Down(n_filters * 2, n_filters * 4)
        self.down3 = Down(n_filters * 4, n_filters * 8)
        self.bottleneck = Down(n_filters * 8, n_filters * 16)
        self.up1 = Up(n_filters * 16, n_filters * 8)
        self.up2 = Up(n_filters * 8, n_filters * 4)
        self.up3 = Up(n_filters * 4, n_filters * 2)
        self.up4 = Up(n_filters * 2, n_filters)
        self.out = OutConv(n_filters, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_out = self.out(x)
        return x_out

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(n_channels=1, n_classes=1)
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
