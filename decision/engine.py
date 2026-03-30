class DecisionEngine:
    def __init__(self, error_threshold=50):
        self.error_threshold = error_threshold

    def decide(self, drift, rolling_avg, trend):
        
        # Case 1: Strong drift → retrain immediately
        if drift:
            return "RETRAIN"

        # Case 2: Error increasing → watch carefully
        if trend and rolling_avg is not None and rolling_avg > self.error_threshold:
            return "MONITOR"

        # Case 3: Everything fine
        return "STABLE"