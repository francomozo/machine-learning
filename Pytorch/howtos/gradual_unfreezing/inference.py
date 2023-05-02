"""
@author: Franco Mozo
@date:   05-04-2023
@desc:   Script to make inference with trained model
"""
from _init_paths import _init_path

_init_path('Pytorch/from_scratch/vision/src', recursive=True)
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.models import LeNet


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-model', type=str, required=True, help='Path to the model')
    parser.add_argument('-image', type=str, required=True, help='Path to the image')
    return parser.parse_args()
    
def imshow(img, class_name):
    img = img / 2 + 0.5     # unnormalize
    if img.is_cuda:
        img = img.cpu().squeeze(0)
    npimg = img.numpy()
    # show image with opencv 
    # show image of size 250 x 250
    cv2.namedWindow(f'Predicted: {class_name}', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'Predicted: {class_name}', 512, 512)
    cv2.imshow(f'Predicted: {class_name}', np.transpose(npimg, (1, 2, 0)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
    

# => Init
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# => Load model
model = LeNet(in_channels=3).to(device)
model.load_state_dict(torch.load(args.model)['state_dict'])

# => Prepare image
image = cv2.imread(args.image)
# Resize to 32x32
image = cv2.resize(image, (32, 32))
# To torch of shape [1, 3, 32, 32]
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
# Normalize to [-1, 1]
image = image / 255 * 2 - 1
# To device
image = image.to(device)

# => Inference
outputs = model(image)
_, predicted = torch.max(outputs, 1)

imshow(image, CLASSES[predicted.item()])