"""
Concept Drift scenario.
Sensor features remain unchanged while the RUL label shifts downward by
60 cycles, simulating a turbofan that degrades faster than the historical fleet.

This tests ADWIN independently: the KS test on features should not detect drift
because the input distribution is unchanged. Only prediction error should rise.
"""


class ConceptDrift:
    META = {
        "id": "concept_drift",
        "name": "Concept Drift",
        "severity": "High",
        "duration": 150,
        "description": "The engine degrades 60 cycles faster than the historical fleet while sensor features remain within normal ranges.",
        "expectedBehavior": "ADWIN should detect rising prediction error after roughly 60-80 cycles. Feature drift should remain low, then error-based retraining is triggered.",
        "tags": ["adwin", "concept-drift", "label-shift"],
    }

    @staticmethod
    def apply(data, cycle_index, baseline_stds, rng):
        data["RUL"] = float(max(0.0, data["RUL"] - 60.0))
