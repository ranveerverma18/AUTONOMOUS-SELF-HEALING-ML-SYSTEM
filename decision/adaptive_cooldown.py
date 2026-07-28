class AdaptiveCooldown:
    """
    Dynamically adjusts retrain cooldown based on drift severity.
    High drift → shorter cooldown (faster response)
    Low drift → longer cooldown (fewer unnecessary retrains)
    """
    
    def __init__(self, min_cooldown=20, max_cooldown=50):
        self.min_cooldown = min_cooldown
        self.max_cooldown = max_cooldown
        self.last_retrain_idx = -max_cooldown
    
    def should_retrain(self, current_idx, drift_score):
        """
        Return (should_retrain: bool, required_cooldown: int, cycles_elapsed: int).
        """
        cycles_since_retrain = current_idx - self.last_retrain_idx
        
        if drift_score > 0.8:
            required_cooldown = self.min_cooldown
        elif drift_score > 0.5:
            required_cooldown = (self.min_cooldown + self.max_cooldown) // 2
        else:
            required_cooldown = self.max_cooldown
        
        should_retrain = cycles_since_retrain >= required_cooldown
        
        return should_retrain, required_cooldown, cycles_since_retrain
    
    def mark_retrain(self, current_idx):
        """Mark that a retrain occurred at this index."""
        self.last_retrain_idx = current_idx
    
    def get_status(self, current_idx, drift_score):
        """Return human-readable cooldown status dict."""
        should_retrain, required, elapsed = self.should_retrain(current_idx, drift_score)
        
        return {
            "should_retrain": should_retrain,
            "required_cooldown": required,
            "cycles_elapsed": elapsed,
            "cycles_remaining": max(0, required - elapsed),
            "drift_severity": "high" if drift_score > 0.8 else "medium" if drift_score > 0.5 else "low"
        }
