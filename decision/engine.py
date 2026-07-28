class DecisionEngine:
    def __init__(self, error_threshold=45): 
        self.error_threshold = error_threshold

    def decide(self, drift, rolling_avg, trend, drift_score=0.0):
        """
        Enhanced decision engine with multiple states based on drift severity.

        MAE thresholds (all derived from error_threshold=45):
          - ALERT:   MAE > 67.5  (threshold * 1.5) — critically high error
          - MONITOR: MAE > 45.0  (threshold * 1.0) — error above threshold and trending up
          - WATCH:   MAE > 36.0  (threshold * 0.8) — error approaching threshold

        Drift states:
          - RETRAIN_URGENT: drift_score > 0.8
          - RETRAIN:        drift_score > 0.5  or any drift signal
          - STABLE:         no drift, no error concerns
        """
        
        if drift and drift_score > 0.8:
            return "RETRAIN_URGENT"

        if drift and drift_score > 0.5:
            return "RETRAIN"

        if drift:
            return "RETRAIN"

        if rolling_avg is not None and rolling_avg > self.error_threshold * 1.5:
            return "ALERT"

        if trend and rolling_avg is not None and rolling_avg > self.error_threshold:
            return "MONITOR"

        if rolling_avg is not None and rolling_avg > self.error_threshold * 0.8:
            return "WATCH"

        return "STABLE"