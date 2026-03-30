from river.drift import ADWIN

class DriftDetector:
    def __init__(self):
        self.adwin = ADWIN(delta=0.1)  # default is 0.002