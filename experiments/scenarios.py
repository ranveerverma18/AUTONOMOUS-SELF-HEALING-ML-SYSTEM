"""Experiment-facing scenario registry helpers."""

from __future__ import annotations

from scenarios.registry import SCENARIO_REGISTRY


def scenario_metadata(scenario_id: str) -> dict:
    if scenario_id not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    return dict(SCENARIO_REGISTRY[scenario_id].META)
