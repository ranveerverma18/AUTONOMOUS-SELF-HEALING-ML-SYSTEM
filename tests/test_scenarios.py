import numpy as np
from scenarios.registry import SCENARIO_REGISTRY, get_scenario_list


EXPECTED_IDS = {
    "gradual_drift",
    "sudden_spike",
    "high_noise",
    "sensor_failure",
    "concept_drift",
    "correlated_drift",
    "intermittent_spikes",
    "drift_recovery",
}

_BASELINE_STDS = {f"sensor_{i}": 1.0 for i in range(1, 22)}
_BASELINE_STDS.update({f"op_setting_{i}": 1.0 for i in range(1, 4)})

RNG = np.random.default_rng(0)


def _base_data():
    data = {f"sensor_{i}": 5.0 for i in range(1, 22)}
    data.update({f"op_setting_{i}": 0.5 for i in range(1, 4)})
    data["RUL"] = 80.0
    data["unit"] = 1
    data["cycle"] = 1
    return data


def test_registry_contains_all_scenarios():
    assert set(SCENARIO_REGISTRY.keys()) == EXPECTED_IDS


def test_get_scenario_list_length():
    lst = get_scenario_list()
    assert len(lst) == len(EXPECTED_IDS)


def test_each_scenario_has_required_meta_fields():
    for scenario_id, cls in SCENARIO_REGISTRY.items():
        meta = cls.META
        assert "id" in meta, f"{scenario_id} missing 'id'"
        assert "name" in meta, f"{scenario_id} missing 'name'"
        assert "severity" in meta, f"{scenario_id} missing 'severity'"
        assert "duration" in meta, f"{scenario_id} missing 'duration'"
        assert "description" in meta, f"{scenario_id} missing 'description'"


def test_each_scenario_apply_mutates_data():
    for scenario_id, cls in SCENARIO_REGISTRY.items():
        data = _base_data()
        original = dict(data)
        # IntermittentSpikes only fires on cycle_index % 7 == 0
        cycle = 7 if scenario_id == "intermittent_spikes" else 5
        cls.apply(data, cycle_index=cycle, baseline_stds=_BASELINE_STDS, rng=RNG)
        changed = any(data[k] != original[k] for k in data)
        assert changed, f"{scenario_id}.apply() did not modify any value at cycle {cycle}"




def test_scenario_apply_does_not_raise_at_cycle_zero():
    for scenario_id, cls in SCENARIO_REGISTRY.items():
        data = _base_data()
        try:
            cls.apply(data, cycle_index=0, baseline_stds=_BASELINE_STDS, rng=RNG)
        except Exception as exc:
            assert False, f"{scenario_id}.apply() raised at cycle 0: {exc}"
