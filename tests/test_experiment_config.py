from pathlib import Path

import pytest

from experiments.config import ExperimentConfig
from experiments.runner import parse_args


def test_config_creates_output_directories(tmp_path):
    config = ExperimentConfig(output_dir=tmp_path)
    config.ensure_output_dirs()

    assert config.raw_dir.exists()
    assert config.aggregated_dir.exists()
    assert config.figures_dir.exists()


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        ExperimentConfig(strategy="unknown")


def test_output_dir_is_path():
    config = ExperimentConfig(output_dir=Path("experiments/results"))

    assert config.raw_dir == Path("experiments/results/raw")


def test_runner_cli_defaults_match_config():
    """Regression test: CLI defaults must match ExperimentConfig defaults.
    
    This test prevents the bug where experiments/runner.py had stale hardcoded
    argparse defaults (stream_length=320, onset 80-100) that diverged from
    ExperimentConfig (stream_length=2400, onset 25-35).
    
    The bug caused the entire 96-run matrix to execute with wrong parameters,
    resulting in zero scenario activations across all runs.
    """
    import sys
    import argparse
    
    # Mock empty command line
    original_argv = sys.argv
    try:
        sys.argv = ['test_program']
        args = parse_args()
        
        # Create default config for comparison
        default_config = ExperimentConfig()
        
        # Verify critical locked protocol parameters
        assert args.stream_length == default_config.stream_length, (
            f"CLI default stream_length={args.stream_length} does not match "
            f"ExperimentConfig default={default_config.stream_length}"
        )
        assert args.scenario_onset_cycle_min == default_config.scenario_onset_cycle_min, (
            f"CLI default scenario_onset_cycle_min={args.scenario_onset_cycle_min} does not match "
            f"ExperimentConfig default={default_config.scenario_onset_cycle_min}"
        )
        assert args.scenario_onset_cycle_max == default_config.scenario_onset_cycle_max, (
            f"CLI default scenario_onset_cycle_max={args.scenario_onset_cycle_max} does not match "
            f"ExperimentConfig default={default_config.scenario_onset_cycle_max}"
        )
        
        # Also verify other important defaults
        assert args.stream_mode == default_config.stream_mode
        assert args.strategy == default_config.strategy
        assert args.scenario == default_config.scenario
        assert args.seed == default_config.seed
        
    finally:
        sys.argv = original_argv
