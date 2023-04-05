from copy import deepcopy

import torch


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
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