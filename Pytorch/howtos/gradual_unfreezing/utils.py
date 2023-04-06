from copy import deepcopy
from pathlib import Path

import numpy as np
# import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc='Checking accuracy'):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

class BestModel:
    def __init__(self, pt_path):
        self.pt_path = pt_path
        self.best_state_dict = None
        self.best_test_acc = -1
        self.best_model_epoch = -1
    
    def compare_and_store(self, model, test_acc, epoch):
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            self.best_state_dict = deepcopy(model.state_dict())
            self.best_model_epoch = epoch
        
    def save(self):
        torch.save(
            {
                'state_dict': self.best_state_dict,
                'epoch': self.best_model_epoch,
            }, 
            self.pt_path
        )
        print(f'Saved best model at epoch {self.best_model_epoch} with test accuracy {self.best_test_acc} to {self.pt_path}')
        

class Animals10Dataset(Dataset):
    # https://www.kaggle.com/datasets/alessiocorrado99/animals10?select=raw-img
    def __init__(self, root, train, transforms):
        self.translate_class_names = {'cane': 'dog', 'horse': 'cavallo', 'elefante': 'elephant', 'farfalla': 'butterfly',\
            'gallina': 'chicken', 'gatto': 'cat', 'mucca': 'cow', 'pecora': 'sheep', 'scoiattolo': 'squirrel', 'dog': 'cane',\
            'cavallo': 'horse', 'elephant': 'elefante', 'butterfly': 'farfalla', 'chicken': 'gallina', 'cat': 'gatto',\
            'cow': 'mucca', 'spider': 'ragno', 'squirrel': 'scoiattolo', 'sheep': 'pecora', 'ragno': 'spider'}
        self.classes = ['horse', 'dog', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'spider', 'squirrel', 'sheep']

        self.dir = root / Path('train' if train else 'test')
        self.images = []
        for class_dir in self.dir.iterdir():
            for image_path in class_dir.iterdir():
                self.images.append((str(image_path), class_dir.name))
        
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        im_path, class_name = self.images[idx]
        pil_image = Image.open(im_path).convert('RGB')
        
        image = np.array(pil_image)
        if pil_image.mode != 'RGB':
            if pil_image.mode == 'RGBA':
                image = image[:,:,:3]
            elif pil_image.mode == 'L':
                image = np.stack([image] * 3, axis=-1)
        
        if image is None:
            raise ValueError(f'Could not read image {im_path}')
        if self.transforms:
            image = self.transforms(image)
        
        label = self.classes.index(class_name)
        return image, label