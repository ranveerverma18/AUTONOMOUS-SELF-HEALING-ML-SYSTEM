"""
High Noise scenario.
Multiplies all sensor variance by 5x without shifting the mean.
"""

_ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]


class HighNoise:
    META = {
        "id": "high_noise",
        "name": "High Sensor Noise",
        "severity": "Medium",
        "duration": 60,
        "description": "Sensor variance increases 5x with no mean shift, simulating electrical interference.",
        "expectedBehavior": "Confidence should fall as uncertainty increases. The quality gate should reject a noisy retrain candidate if validation error worsens.",
        "tags": ["confidence", "noise", "gate-rejection"],
    }

    @staticmethod
    def apply(data, cycle_index, baseline_stds, rng):
        for sensor in _ALL_SENSORS:
            std = baseline_stds.get(sensor, 1.0)
            data[sensor] = float(data[sensor] + rng.normal(0, 4.5 * std))
