"""
Drift-Then-Recovery scenario.
First 30 cycles: all sensors shift severely by +6σ.
Next 30 cycles: sensors gradually recover back to baseline.
"""

_ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]


class DriftRecovery:
    META = {
        "id": "drift_recovery",
        "name": "Drift to Recovery",
        "severity": "Critical",
        "duration": 60,
        "description": "Severe +6σ drift for 30 cycles, followed by a gradual return to baseline over 30 cycles.",
        "expectedBehavior": "The system should detect drift, retrain, shadow test, promote if the candidate is better, and stabilize without repeated retraining during recovery.",
        "tags": ["adwin", "data-drift", "recovery", "self-healing"],
    }

    @staticmethod
    def apply(data, cycle_index, baseline_stds, rng):
        if cycle_index < 30:
            shift = 6.0
        else:
            recovery_progress = (cycle_index - 30) / 30.0
            shift = 6.0 * (1.0 - recovery_progress)

        for sensor in _ALL_SENSORS:
            std = baseline_stds.get(sensor, 1.0)
            data[sensor] = float(data[sensor] + shift * std)
