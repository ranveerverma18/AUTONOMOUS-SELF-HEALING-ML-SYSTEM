"""
Gradual Sensor Drift scenario.
Slowly shifts 4 key sensors by +0.3σ every 10 cycles, simulating
progressive component wear.
"""

_DRIFT_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_11"]


class GradualDrift:
    META = {
        "id": "gradual_drift",
        "name": "Gradual Sensor Drift",
        "severity": "Medium",
        "duration": 100,
        "description": "Four key sensors drift by +0.3σ every 10 cycles, simulating progressive component wear.",
        "expectedBehavior": "Error drift should rise after roughly 40-60 cycles. Feature drift should follow, then the system schedules retraining.",
        "tags": ["adwin", "data-drift", "gradual"],
    }

    @staticmethod
    def apply(data, cycle_index, baseline_stds, rng):
        shift_magnitude = 0.3 * ((cycle_index // 10) + 1)
        for sensor in _DRIFT_SENSORS:
            std = baseline_stds.get(sensor, 1.0)
            data[sensor] = float(data[sensor] + shift_magnitude * std)
