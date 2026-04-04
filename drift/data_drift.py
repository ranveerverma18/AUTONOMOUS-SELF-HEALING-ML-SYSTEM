from scipy.stats import ks_2samp
from collections import deque
import numpy as np


class DataDriftDetector:
    def __init__(
        self,
        window_size=50,
        p_threshold=0.05,
        min_effect_size=0.1,
        correction_method="bonferroni",
        drift_feature_ratio_threshold=0.3,
        carryover_fraction=0.3,
    ):
        self.window_size = window_size
        self.p_threshold = p_threshold
        self.min_effect_size = min_effect_size
        self.correction_method = correction_method
        self.drift_feature_ratio_threshold = drift_feature_ratio_threshold
        self.carryover_fraction = carryover_fraction
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        self.feature_names = None
        self.last_result = {
            "drift_detected": False,
            "phase": "reference_warmup",
            "tested_features": 0,
            "drifted_features": [],
            "min_adjusted_p_value": None,
            "max_statistic": None,
            "drift_score": 0.0,
            "drift_feature_ratio": 0.0,
        }

    def _extract_features(self, data_point):
        numeric_features = {}
        for key, value in data_point.items():
            if key in ['RUL', 'unit', 'cycle']:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                numeric_features[key] = numeric_value
        return numeric_features

    def _set_feature_names_from_reference(self):
        if not self.reference_window:
            self.feature_names = []
            return

        common_keys = set(self.reference_window[0].keys())
        for row in self.reference_window:
            common_keys &= set(row.keys())
        self.feature_names = sorted(common_keys)

    def _adjust_p_values(self, p_values):
        m = len(p_values)
        if m == 0:
            return []

        if self.correction_method.lower() == "bonferroni":
            return [min(p * m, 1.0) for p in p_values]

        # Fallback: no correction for unsupported method names.
        return p_values

    def update_with_details(self, data_point):
        features = self._extract_features(data_point)

        if not features:
            self.last_result = {
                "drift_detected": False,
                "phase": "invalid_point",
                "tested_features": 0,
                "drifted_features": [],
                "min_adjusted_p_value": None,
                "max_statistic": None,
                "drift_score": 0.0,
                "drift_feature_ratio": 0.0,
            }
            return self.last_result

        # Build reference first so baseline and current windows are temporally separated.
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(features)
            if len(self.reference_window) == self.window_size and self.feature_names is None:
                self._set_feature_names_from_reference()
            self.last_result = {
                "drift_detected": False,
                "phase": "reference_warmup",
                "tested_features": 0,
                "drifted_features": [],
                "min_adjusted_p_value": None,
                "max_statistic": None,
                "drift_score": 0.0,
                "drift_feature_ratio": 0.0,
            }
            return self.last_result

        if self.feature_names is None:
            self._set_feature_names_from_reference()

        filtered_current = {k: features[k] for k in self.feature_names if k in features}
        self.current_window.append(filtered_current)

        if len(self.current_window) < self.window_size:
            self.last_result = {
                "drift_detected": False,
                "phase": "current_warmup",
                "tested_features": 0,
                "drifted_features": [],
                "min_adjusted_p_value": None,
                "max_statistic": None,
                "drift_score": 0.0,
                "drift_feature_ratio": 0.0,
            }
            return self.last_result

        stats = []
        p_values = []
        tested_keys = []

        for key in self.feature_names:
            ref_vals = [x[key] for x in self.reference_window if key in x]
            curr_vals = [x[key] for x in self.current_window if key in x]

            # Skip unstable comparisons with too few numeric values.
            if len(ref_vals) < 8 or len(curr_vals) < 8:
                continue

            statistic, p_value = ks_2samp(ref_vals, curr_vals)
            tested_keys.append(key)
            stats.append(float(statistic))
            p_values.append(float(p_value))

        adjusted_p_values = self._adjust_p_values(p_values)
        drifted_features = []
        drift_score = 0.0

        for key, stat, adjusted_p in zip(tested_keys, stats, adjusted_p_values):
            if adjusted_p < self.p_threshold and stat >= self.min_effect_size:
                drifted_features.append(key)
            drift_score = max(drift_score, stat * max(0.0, 1.0 - adjusted_p))

        tested_features = len(tested_keys)
        drift_feature_ratio = (
            len(drifted_features) / tested_features if tested_features > 0 else 0.0
        )
        drift_detected = (
            len(drifted_features) > 0
            or drift_score > 0.5
        )

        if drift_detected:
            # Re-anchor baseline after confirmed drift.
            self.reference_window = deque(self.current_window, maxlen=self.window_size)
            carryover_size = max(1, int(self.window_size * self.carryover_fraction))
            carryover_points = list(self.current_window)[-carryover_size:]
            self.current_window = deque(carryover_points, maxlen=self.window_size)

        self.last_result = {
            "drift_detected": drift_detected,
            "phase": "detecting",
            "tested_features": tested_features,
            "drifted_features": drifted_features,
            "min_adjusted_p_value": min(adjusted_p_values) if adjusted_p_values else None,
            "max_statistic": max(stats) if stats else None,
            "drift_score": float(drift_score),
            "drift_feature_ratio": float(drift_feature_ratio),
        }
        return self.last_result

    def update(self, data_point):
        result = self.update_with_details(data_point)
        return result["drift_detected"]