"""
Intermittent Spikes scenario.
Every 7th cycle, 2 randomly selected sensors spike by ±12σ.
All other cycles remain normal.
"""

_ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]


class IntermittentSpikes:
    META = {
        "id": "intermittent_spikes",
        "name": "Intermittent Sensor Spikes",
        "severity": "Low",
        "duration": 90,
        "description": "Two random sensors spike by ±12σ every 7 cycles, with normal readings between spikes.",
        "expectedBehavior": "Isolation Forest should flag spike cycles as anomalies. ADWIN may stay quiet because the issue is intermittent, so retraining is not expected.",
        "tags": ["anomaly", "intermittent", "isolation-forest"],
    }

    @staticmethod
    def apply(data, cycle_index, baseline_stds, rng):
        if cycle_index % 7 == 0:
            chosen = rng.choice(_ALL_SENSORS, size=2, replace=False)
            for sensor in chosen:
                std = baseline_stds.get(sensor, 1.0)
                direction = 1 if rng.random() > 0.5 else -1
                data[sensor] = float(data[sensor] + direction * 12.0 * std)
