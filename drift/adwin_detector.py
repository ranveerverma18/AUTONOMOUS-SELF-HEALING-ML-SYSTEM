from river.drift import ADWIN

class DriftDetector:
    def __init__(self):
        self.adwin = ADWIN(delta=0.1)

    def update(self, value):
        self.adwin.update(value)
        return self.adwin.drift_detected