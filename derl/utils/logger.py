import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Logger for training metrics"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.step = 0
    
    def log_scalar(self, tag, value, step=None):
        """Log scalar value"""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, values, step=None):
        """Log multiple scalar values"""
        if step is None:
            step = self.step
        self.writer.add_scalars(tag, values, step)
    
    def increment_step(self):
        """Increment global step counter"""
        self.step += 1
    
    def close(self):
        """Close the logger"""
        self.writer.close()
