"""CLI runner for deterministic research experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from experiments.baselines import make_strategy
from experiments.config import ExperimentConfig
from experiments.data_stream import (
    baseline_statistics,
    build_stream_records,
    iter_stream_events,
    load_fd001_with_rul,
    split_train_stream_units,
)
from experiments.metrics import summarize_events


def run_experiment(config: ExperimentConfig, base_path: Path | None = None) -> dict:
    base_path = Path.cwd() if base_path is None else base_path
    config.ensure_output_dirs()
    run_id = (
        f"{config.strategy}_{config.scenario}_seed{config.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    df = load_fd001_with_rul(base_path)
    train_df, stream_df = split_train_stream_units(
        df, seed=config.seed, train_fraction=config.train_fraction
    )
    _, baseline_stds = baseline_statistics(train_df)
    records = build_stream_records(stream_df, config.stream_mode, config.seed)
    strategy = make_strategy(config, train_df, baseline_stds)

    rows = [
        strategy.process(event, run_id)
        for event in iter_stream_events(
            records=records,
            baseline_stds=baseline_stds,
            scenario_id=config.scenario,
            seed=config.seed,
            stream_length=config.stream_length,
            scenario_onset_cycle_min=config.scenario_onset_cycle_min,
            scenario_onset_cycle_max=config.scenario_onset_cycle_max,
        )
    ]

    raw_path = config.raw_dir / f"{run_id}_events.csv"
    summary_path = config.aggregated_dir / f"{run_id}_summary.json"
    if rows:
        with raw_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize_events(
        rows,
        config.scenario_onset_cycle_min,
        recovery_error_threshold=config.retrain_error_threshold,
    )
    payload = {
        "run_id": run_id,
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "summary": asdict(summary),
        "raw_events": str(raw_path),
        "summary_json": str(summary_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one self-healing ML experiment")
    # Use ExperimentConfig defaults as single source of truth
    default_config = ExperimentConfig()
    parser.add_argument("--strategy", default=default_config.strategy)
    parser.add_argument("--scenario", default=default_config.scenario)
    parser.add_argument("--seed", type=int, default=default_config.seed)
    parser.add_argument("--stream-length", type=int, default=default_config.stream_length)
    parser.add_argument("--stream-mode", default=default_config.stream_mode, choices=["research", "interleaved", "legacy"])
    parser.add_argument("--scenario-onset-cycle-min", type=int, default=default_config.scenario_onset_cycle_min)
    parser.add_argument("--scenario-onset-cycle-max", type=int, default=default_config.scenario_onset_cycle_max)
    parser.add_argument("--output-dir", default=str(default_config.output_dir))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        strategy=args.strategy,
        scenario=args.scenario,
        stream_length=args.stream_length,
        stream_mode=args.stream_mode,
        scenario_onset_cycle_min=args.scenario_onset_cycle_min,
        scenario_onset_cycle_max=args.scenario_onset_cycle_max,
        output_dir=Path(args.output_dir),
    )
    payload = run_experiment(config)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
