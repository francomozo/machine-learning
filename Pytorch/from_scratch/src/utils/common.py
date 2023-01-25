from time import perf_counter

import torch


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

def binarize_map(image):
    # Pets dataset
    values = {'pet': 0.003921569, 'not_pet': 0.007843138, 'pet_border': 0.011764706} 
    segm_value = 1
    image = torch.where(image == values['pet'], segm_value, image)
    image = torch.where(image == values['pet_border'], segm_value, image)
    image = torch.where(image == values['not_pet'], 0, image)
    
    return image