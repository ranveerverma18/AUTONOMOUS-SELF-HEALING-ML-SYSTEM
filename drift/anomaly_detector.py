from sklearn.ensemble import IsolationForest
import numpy as np


class AnomalyDetector:
    """
    Detects anomalous data points that might cause bad predictions.
    Uses Isolation Forest to identify outliers in feature space.
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.detector = None
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, train_df):
        """Train anomaly detector on baseline data"""
        features = train_df.drop(columns=['RUL', 'unit', 'cycle'], errors='ignore')
        
        if len(features) < 10:
            return False
        
        self.feature_names = features.columns.tolist()
        
        self.detector = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        self.detector.fit(features)
        self.is_fitted = True
        return True
    
    def is_anomaly(self, data_point):
        """
        Check if a data point is anomalous.

        Returns:
            (is_anomaly: bool, anomaly_score: float)
        """
        if not self.is_fitted or self.detector is None:
            return False, 0.0

        try:
            import pandas as pd
            # Build a single-row DataFrame with the exact column names used at fit time.
            # This eliminates the sklearn UserWarning about missing feature names.
            row = {f: [data_point.get(f, 0.0)] for f in self.feature_names}
            features_df = pd.DataFrame(row, columns=self.feature_names)

            prediction = self.detector.predict(features_df)[0]
            score = self.detector.score_samples(features_df)[0]

            return bool(prediction == -1), float(score)

        except Exception:
            return False, 0.0


