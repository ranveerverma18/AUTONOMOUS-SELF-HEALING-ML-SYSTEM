"""
Correlated Multi-Sensor Drift scenario.
Six temperature-related sensors drift together by +3σ, simulating a
thermal event such as overheating or cooling system failure.
"""

_CORRELATED_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_11",
    "sensor_12",
    "sensor_13",
]


class CorrelatedDrift:
    META = {
        "id": "correlated_drift",
        "name": "Correlated Sensor Drift",
        "severity": "High",
        "duration": 60,
        "description": "Six temperature and pressure sensors drift by +3σ together, simulating a thermal event.",
        "expectedBehavior": "The feature drift ratio should rise quickly, multiple sensors should be highlighted, and retraining should be triggered.",
        "tags": ["data-drift", "correlated", "feature-drift"],
    }

    @staticmethod
    def apply(data, cycle_index, baseline_stds, rng):
        for sensor in _CORRELATED_SENSORS:
            std = baseline_stds.get(sensor, 1.0)
            noise = rng.normal(0, 0.15 * std)
            data[sensor] = float(data[sensor] + 3.0 * std + noise)
