"""Centralized configuration for reproducible research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_STRATEGIES = {"static", "scheduled", "naive_adaptive", "proposed"}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    strategy: str = "proposed"
    scenario: str = "gradual_drift"
    stream_length: int = 2400
    stream_mode: str = "interleaved"
    scenario_onset_cycle_min: int = 25
    scenario_onset_cycle_max: int = 35
    train_fraction: float = 0.76
    validation_fraction: float = 0.25
    retraining_interval: int = 100
    data_drift_window: int = 20
    data_drift_p_threshold: float = 0.05
    data_drift_feature_ratio_threshold: float = 0.12
    data_drift_min_effect_size: float = 0.08
    error_window: int = 8
    error_threshold: float = 75.0
    retrain_error_threshold: float = 45.0
    retrain_drift_score_threshold: float = 0.55
    performance_gate_threshold: float = 0.95
    shadow_window: int = 20
    cooldown: int = 30
    minimum_retraining_samples: int = 55
    minimum_validation_rows: int = 20
    minimum_validation_units: int = 1
    output_dir: Path = Path("experiments/results")

    def __post_init__(self) -> None:
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        if self.stream_mode not in {"research", "interleaved", "legacy"}:
            raise ValueError(f"Unknown stream_mode: {self.stream_mode}")
        if self.stream_length <= 0:
            raise ValueError("stream_length must be positive")
        if self.minimum_retraining_samples < 10:
            raise ValueError("minimum_retraining_samples must be at least 10")
        if self.cooldown < 0:
            raise ValueError("cooldown must be non-negative")
        if self.minimum_validation_rows < 1:
            raise ValueError("minimum_validation_rows must be at least 1")
        if self.minimum_validation_units < 1:
            raise ValueError("minimum_validation_units must be at least 1")
        if self.scenario_onset_cycle_min < 1:
            raise ValueError("scenario_onset_cycle_min must be at least 1")
        if self.scenario_onset_cycle_max < self.scenario_onset_cycle_min:
            raise ValueError("scenario_onset_cycle_max must be >= scenario_onset_cycle_min")

    @property
    def raw_dir(self) -> Path:
        return self.output_dir / "raw"

    @property
    def aggregated_dir(self) -> Path:
        return self.output_dir / "aggregated"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def adaptation_start_index(self) -> int:
        """First sample where adaptive strategies may update models.

        For interleaved fleet monitoring, scenarios are per-engine lifecycle
        based, so this property is deprecated. Kept for backward compatibility
        with sequential stream experiments.
        """
        return 0  # No global adaptation start for per-engine scenarios

    def ensure_output_dirs(self) -> None:
        for path in [self.raw_dir, self.aggregated_dir, self.figures_dir]:
            path.mkdir(parents=True, exist_ok=True)

