import torch


def check_accuracy(loader, model, device, fully_net=True):
    # -> accuracy = num_corrects / tot_data_points
    model.eval()
    num_corrects = 0
    tot_samples = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if fully_net:
            x = x.reshape(x.shape[0], -1)
        
        scores = model(x)
        _, preds = scores.max(1)
        num_corrects += torch.sum(preds == y)
        tot_samples += x.shape[0]
    acc = float(num_corrects) / float(tot_samples)
        
    model.train()
    return num_corrects, tot_samples, acc 
