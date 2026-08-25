"""Deterministic C-MAPSS stream construction for research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from dataset.processed.preprocess_module import add_rul, load_data
from metrics.calculator import FEATURE_COLUMNS
from scenarios.registry import SCENARIO_REGISTRY


@dataclass(frozen=True)
class StreamEvent:
    sample_index: int
    engine_id: int
    cycle: int
    data: dict
    scenario_active: bool
    scenario_cycle: int | None
    degradation_started: bool


def load_fd001_with_rul(base_path: Path) -> pd.DataFrame:
    raw_path = base_path / "dataset" / "raw" / "train_FD001.txt"
    df = add_rul(load_data(str(raw_path)))
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def split_train_stream_units(
    df: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.76,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    units = df["unit"].unique().tolist()
    shuffled = rng.permutation(units).tolist()
    split_at = int(len(shuffled) * train_fraction)
    train_units = set(shuffled[:split_at])
    stream_units = set(shuffled[split_at:])
    train_df = df[df["unit"].isin(train_units)].reset_index(drop=True)
    stream_df = df[df["unit"].isin(stream_units)].reset_index(drop=True)
    return train_df, stream_df


def build_stream_records(
    stream_df: pd.DataFrame,
    stream_mode: str = "research",
    seed: int = 42,
) -> list[dict]:
    """Return stream records under a documented ordering policy.

    research mode sorts by engine unit and cycle, preserving C-MAPSS temporal
    trajectories (sequential single-asset monitoring).
    
    interleaved mode groups observations by cycle number, sampling all engines
    round-robin per cycle (fleet monitoring).
    
    legacy mode reproduces the dashboard's historical behavior: individual rows
    are randomly permuted using the configured seed.
    """

    if stream_mode == "research":
        ordered = stream_df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        return ordered.to_dict(orient="records")
    elif stream_mode == "interleaved":
        return _build_interleaved_stream(stream_df)
    elif stream_mode == "legacy":
        rng = np.random.default_rng(seed)
        records = stream_df.to_dict(orient="records")
        perm = rng.permutation(len(records)).tolist()
        return [records[i] for i in perm]
    else:
        raise ValueError(f"Unknown stream_mode: {stream_mode}")


def _build_interleaved_stream(stream_df: pd.DataFrame) -> list[dict]:
    """Build interleaved fleet stream preserving within-engine cycle order.
    
    For each cycle c = 1, 2, 3, ..., max_cycle:
        Include all engines that have a cycle-c observation, ordered by unit ID.
    
    This simulates fleet monitoring where the system receives periodic
    observations from multiple assets in parallel.
    """
    max_cycle = int(stream_df["cycle"].max())
    records = []
    
    for cycle in range(1, max_cycle + 1):
        cycle_rows = stream_df[stream_df["cycle"] == cycle].copy()
        cycle_rows = cycle_rows.sort_values("unit")
        records.extend(cycle_rows.to_dict(orient="records"))
    
    return records


def baseline_statistics(train_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    means = {feature: float(train_df[feature].mean()) for feature in FEATURE_COLUMNS}
    stds = {feature: float(max(train_df[feature].std(), 0.5)) for feature in FEATURE_COLUMNS}
    return means, stds


def iter_stream_events(
    records: list[dict],
    baseline_stds: dict[str, float],
    scenario_id: str,
    seed: int,
    stream_length: int,
    scenario_onset_cycle_min: int = 80,
    scenario_onset_cycle_max: int = 100,
    sensor_noise_fraction: float = 0.015,
) -> Iterator[StreamEvent]:
    """Yield reproducible stream events with per-engine scenario lifecycle.
    
    Scenarios are applied per-engine based on each engine's lifecycle cycle,
    not global stream index. Each engine receives a deterministic onset cycle
    within [onset_cycle_min, onset_cycle_max] based on the experiment seed.
    """

    if not records:
        return

    rng = np.random.default_rng(seed)
    scenario_cls = SCENARIO_REGISTRY[scenario_id]
    duration = int(scenario_cls.META["duration"])

    # Assign deterministic per-engine onset cycles
    engine_ids = sorted(set(r["unit"] for r in records))
    engine_onset_map = {}
    onset_rng = np.random.default_rng(seed + 1000)  # Separate RNG for onsets
    for engine_id in engine_ids:
        onset = onset_rng.integers(scenario_onset_cycle_min, scenario_onset_cycle_max + 1)
        engine_onset_map[engine_id] = onset

    for sample_index in range(stream_length):
        row = dict(records[sample_index % len(records)])
        data = dict(row)
        
        # Add sensor noise
        for feature in FEATURE_COLUMNS:
            scale = baseline_stds.get(feature, 1.0)
            data[feature] = float(data[feature] + rng.normal(0, sensor_noise_fraction * scale))

        # Per-engine scenario application
        engine_id = int(data["unit"])
        engine_cycle = int(data["cycle"])
        engine_onset = engine_onset_map.get(engine_id, scenario_onset_cycle_max + 1)
        
        scenario_cycle_for_engine = engine_cycle - engine_onset
        scenario_active = 0 <= scenario_cycle_for_engine < duration
        degradation_started = (engine_cycle == engine_onset)
        
        if scenario_active:
            scenario_cls.apply(data, scenario_cycle_for_engine, baseline_stds, rng)

        yield StreamEvent(
            sample_index=sample_index,
            engine_id=engine_id,
            cycle=engine_cycle,
            data=data,
            scenario_active=scenario_active,
            scenario_cycle=int(scenario_cycle_for_engine) if scenario_active else None,
            degradation_started=degradation_started,
        )
