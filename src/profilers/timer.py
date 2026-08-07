import time
import torch


class Timer:

    def __init__(self, device):
        self.device = device
        self.start_time = None

    def start(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize() #->> for GPU synchronization

        self.start_time = time.perf_counter()

    def stop(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        return (time.perf_counter() - self.start_time) * 1000