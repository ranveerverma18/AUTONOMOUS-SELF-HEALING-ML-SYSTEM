"""Generate deterministic experiment manifest before execution.

This script creates a pre-execution manifest CSV for an experiment matrix,
capturing all configuration parameters, provenance information, and planned
execution status.

The manifest uses deterministic logical run_ids (no timestamps) that can be
mapped to actual timestamped output files during execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.config import ExperimentConfig, VALID_STRATEGIES
from scenarios.registry import SCENARIO_REGISTRY


def get_git_commit() -> str:
    """Get current Git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_dataset_checksum(dataset_path: Path) -> str:
    """Get MD5 checksum from PROVENANCE.md or compute if needed.
    
    Returns MD5 checksum for train_FD001.txt or 'unavailable' if cannot be determined.
    """
    provenance_file = dataset_path.parent / "PROVENANCE.md"
    
    # Try to read from PROVENANCE.md first
    if provenance_file.exists():
        try:
            content = provenance_file.read_text(encoding="utf-8")
            # Look for train_FD001.txt MD5 checksum in the markdown table
            for line in content.split("\n"):
                if "train_FD001.txt" in line and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    # Table format: | File | Description | Size | MD5 |
                    if len(parts) >= 5:
                        checksum = parts[4].strip("`")
                        if checksum and len(checksum) == 32:
                            return checksum
        except Exception:
            pass
    
    # If PROVENANCE.md doesn't have it, try to compute
    if dataset_path.exists():
        try:
            md5_hash = hashlib.md5()
            with dataset_path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    md5_hash.update(chunk)
            return md5_hash.hexdigest()
        except Exception:
            pass
    
    return "unavailable"


def generate_manifest(
    strategies: list[str],
    scenarios: list[str],
    seeds: list[int],
    output_dir: Path,
) -> Path:
    """Generate experiment manifest CSV.
    
    Args:
        strategies: List of strategy names
        scenarios: List of scenario IDs
        seeds: List of random seeds
        output_dir: Directory to write manifest
        
    Returns:
        Path to generated manifest file
    """
    # Validate inputs
    for strategy in strategies:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {VALID_STRATEGIES}")
    
    for scenario in scenarios:
        if scenario not in SCENARIO_REGISTRY:
            raise ValueError(f"Invalid scenario: {scenario}. Must be one of {list(SCENARIO_REGISTRY.keys())}")
    
    # Get authoritative config defaults
    default_config = ExperimentConfig(output_dir=output_dir)
    
    # Get provenance information
    git_commit = get_git_commit()
    python_version = get_python_version()
    planned_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get dataset checksum from PROVENANCE.md or compute
    dataset_path = project_root / "dataset" / "raw" / "train_FD001.txt"
    dataset_checksum = get_dataset_checksum(dataset_path)
    
    # Generate manifest rows
    rows = []
    for strategy in strategies:
        for scenario in scenarios:
            for seed in seeds:
                # Deterministic logical run_id (no timestamp)
                run_id = f"{strategy}_{scenario}_seed{seed}"
                
                # Predicted output paths (actual files will have timestamps)
                raw_csv_path = str(default_config.raw_dir / f"{run_id}_*.csv")
                summary_json_path = str(default_config.aggregated_dir / f"{run_id}_*.json")
                
                row = {
                    "run_id": run_id,
                    "strategy": strategy,
                    "scenario": scenario,
                    "seed": seed,
                    # Locked protocol parameters from ExperimentConfig
                    "stream_length": default_config.stream_length,
                    "stream_mode": default_config.stream_mode,
                    "scenario_onset_cycle_min": default_config.scenario_onset_cycle_min,
                    "scenario_onset_cycle_max": default_config.scenario_onset_cycle_max,
                    "train_fraction": default_config.train_fraction,
                    "validation_fraction": default_config.validation_fraction,
                    "retraining_interval": default_config.retraining_interval,
                    "data_drift_window": default_config.data_drift_window,
                    "data_drift_p_threshold": default_config.data_drift_p_threshold,
                    "data_drift_feature_ratio_threshold": default_config.data_drift_feature_ratio_threshold,
                    "data_drift_min_effect_size": default_config.data_drift_min_effect_size,
                    "error_window": default_config.error_window,
                    "error_threshold": default_config.error_threshold,
                    "retrain_error_threshold": default_config.retrain_error_threshold,
                    "retrain_drift_score_threshold": default_config.retrain_drift_score_threshold,
                    "performance_gate_threshold": default_config.performance_gate_threshold,
                    "shadow_window": default_config.shadow_window,
                    "cooldown": default_config.cooldown,
                    "minimum_retraining_samples": default_config.minimum_retraining_samples,
                    "minimum_validation_rows": default_config.minimum_validation_rows,
                    "minimum_validation_units": default_config.minimum_validation_units,
                    # Provenance
                    "git_commit": git_commit,
                    "dataset_checksum": dataset_checksum,
                    "python_version": python_version,
                    "planned_timestamp": planned_timestamp,
                    # Execution tracking
                    "status": "PLANNED",
                    "raw_csv_path": raw_csv_path,
                    "summary_json_path": summary_json_path,
                }
                rows.append(row)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write manifest
    manifest_path = output_dir / "experiment_manifest.csv"
    
    if rows:
        fieldnames = list(rows[0].keys())
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    return manifest_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic experiment manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default 96-run matrix (4 strategies × 8 scenarios × 3 seeds)
  python -m scripts.matrix_orchestration.generate_manifest
  
  # Generate mini-matrix for testing
  python -m scripts.matrix_orchestration.generate_manifest \\
    --strategies static,proposed \\
    --scenarios gradual_drift \\
    --seeds 42,123
  
  # Custom output directory
  python -m scripts.matrix_orchestration.generate_manifest \\
    --output-dir results/custom_matrix
        """
    )
    
    # Default full matrix: 4 strategies × 8 scenarios × 3 seeds = 96 runs
    default_strategies = ["static", "scheduled", "naive_adaptive", "proposed"]
    default_scenarios = [
        "gradual_drift",
        "sudden_spike",
        "high_noise",
        "sensor_failure",
        "concept_drift",
        "correlated_drift",
        "intermittent_spikes",
        "drift_recovery",
    ]
    default_seeds = [42, 123, 456]
    
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(default_strategies),
        help="Comma-separated strategy names (default: all 4 strategies)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(default_scenarios),
        help="Comma-separated scenario IDs (default: all 8 scenarios)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, default_seeds)),
        help="Comma-separated random seeds (default: 42,123,456)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Output directory for manifest (default: experiments/results)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Parse comma-separated arguments
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    
    print(f"Generating experiment manifest...")
    print(f"  Strategies: {len(strategies)} ({', '.join(strategies)})")
    print(f"  Scenarios: {len(scenarios)} ({', '.join(scenarios)})")
    print(f"  Seeds: {len(seeds)} ({', '.join(map(str, seeds))})")
    print(f"  Total runs: {len(strategies) * len(scenarios) * len(seeds)}")
    print(f"  Output: {output_dir}")
    print()
    
    manifest_path = generate_manifest(strategies, scenarios, seeds, output_dir)
    
    print(f"✓ Manifest generated: {manifest_path}")
    print(f"  Status: All runs marked as PLANNED")
    print(f"  Git commit: {get_git_commit()}")
    print(f"  Dataset checksum: {get_dataset_checksum(project_root / 'dataset' / 'raw' / 'train_FD001.txt')}")
    print(f"  Python version: {get_python_version()}")
    print()
    print("Next step: Run experiments with run_matrix.py (not yet implemented)")


if __name__ == "__main__":
    main()
