from collections import deque
import numpy as np

class ErrorMonitor:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.errors = deque(maxlen=window_size)

    def update(self, error):
        self.errors.append(error)

        if len(self.errors) < self.window_size:
            return None  # not enough data yet

        rolling_avg = np.mean(self.errors)
        return rolling_avg

    def is_increasing(self):
        if len(self.errors) < self.window_size:
            return False

        # simple trend check
        return self.errors[-1] > self.errors[0]