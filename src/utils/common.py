from time import perf_counter


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
