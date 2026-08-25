"""
Sensor Failure scenario.
Two sensors are stuck at a constant value (0.0), simulating hardware failure
such as a disconnected sensor wire or a grounded sensor.

The Isolation Forest anomaly detector should flag these points.  Because only
two of the 24 numeric feature channels are affected, fleet-level feature drift
may remain below the retraining threshold unless prediction error also rises.
"""

_STUCK_SENSORS = ["sensor_3", "sensor_9"]


class SensorFailure:
    META = {
        "id": "sensor_failure",
        "name": "Stuck Sensor Failure",
        "severity": "High",
        "duration": 80,
        "description": "Two sensors remain fixed at 0.0, simulating disconnected or grounded hardware.",
        "expectedBehavior": "Isolation Forest should flag the fault. Retraining should occur only if the stuck sensors also cause sustained prediction-error or drift-policy escalation.",
        "tags": ["data-drift", "anomaly", "sensor-fault"],
    }

    @staticmethod
    def apply(data, cycle_index, baseline_stds, rng):
        for sensor in _STUCK_SENSORS:
            data[sensor] = 0.0
