"""
Sudden Drift scenario.
All 21 sensors immediately shift by +8σ, simulating a severe
environmental change or major mechanical failure event.
"""

_ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]


class SuddenSpike:
    META = {
        "id": "sudden_spike",
        "name": "Sudden Sensor Drift",
        "severity": "Critical",
        "duration": 45,
        "description": "All 21 sensors shift by +8σ immediately, simulating a severe environmental or mechanical fault.",
        "expectedBehavior": "Urgent retraining should be queued quickly. The quality gate evaluates the candidate model before shadow testing begins.",
        "tags": ["adwin", "data-drift", "concept-drift", "critical"],
    }

    @staticmethod
    def apply(data, cycle_index, baseline_stds, rng):
        for sensor in _ALL_SENSORS:
            std = baseline_stds.get(sensor, 1.0)
            data[sensor] = float(data[sensor] + 8.0 * std)
