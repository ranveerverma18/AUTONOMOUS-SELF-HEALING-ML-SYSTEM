#!/usr/bin/env python3
"""
Result Aggregation Tool

Converts verified per-run outputs into analysis-ready aggregated data.
Requires completion verification to pass before aggregation.

Usage:
    python -m scripts.analysis.aggregate_results \
        --manifest <manifest.csv> \
        --results-dir <results-directory> \
        [--output <output-file>]

Exit codes:
    0 = aggregation succeeded
    1 = verification failed or aggregation error
    2 = invalid CLI/input/configuration
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import subprocess

# Import verification functionality
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.matrix_orchestration.verify_completion import (
    load_manifest,
    validate_run,
    find_orphan_files,
    cross_check_status,
    load_status_file,
    parse_run_id_pattern
)


@dataclass
class AggregatedRun:
    """Single row in the aggregated results table."""
    # Identity fields
    run_id: str
    strategy: str
    scenario: str
    seed: int
    
    # Primary performance metrics
    mae: Optional[float]
    rmse: Optional[float]
    
    # Detection metrics
    detection_delay: Optional[int]
    drift_detections: int
    anomaly_detections: int
    false_positive_triggers: int
    
    # Adaptation lifecycle metrics
    retraining_events: int
    validation_skipped_events: int
    candidates_generated: int
    gate_accepts: int
    gate_rejects: int
    shadow_promotions: int
    shadow_rejections: int
    model_promoted_events: int
    degraded_promotions: int
    degraded_promotion_rate: Optional[float]
    
    # Recovery metrics
    time_to_first_error_recovery: Optional[int]
    time_to_sustained_recovery: Optional[int]
    
    # Efficiency metrics
    total_retraining_time: float
    total_shadow_evaluation_time: float
    total_adaptation_time: float
    mean_inference_latency: Optional[float]
    
    # Configuration provenance (key parameters)
    stream_length: int
    stream_mode: str
    scenario_onset_cycle_min: int
    scenario_onset_cycle_max: int
    train_fraction: float
    validation_fraction: float
    retraining_interval: int
    
    # File provenance
    raw_csv: str
    summary_json: str


@dataclass
class AggregationReport:
    """Metadata about the aggregation process."""
    manifest_path: str
    results_dir: str
    aggregation_timestamp: str
    total_runs: int
    successfully_aggregated: int
    
    # Provenance
    git_commit: str
    dataset_checksum: str
    python_version: str
    
    # Verification summary
    verification_status: str
    runs_with_valid_outputs: int
    runs_missing_outputs: int
    runs_with_duplicate_outputs: int
    runs_with_invalid_outputs: int
    orphan_files: int
    status_mismatches: int


def run_verification(manifest_path: Path, results_dir: Path) -> Dict:
    """Run completion verification and return results."""
    print("Running completion verification...")
    
    # Load manifest
    manifest_runs = load_manifest(manifest_path)
    
    # Load status file if exists
    status_path = results_dir / "matrix_execution_status.csv"
    status_data = load_status_file(status_path)
    
    # Validate each run
    raw_dir = results_dir / "raw"
    aggregated_dir = results_dir / "aggregated"
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    if not aggregated_dir.exists():
        raise FileNotFoundError(f"Aggregated directory not found: {aggregated_dir}")
    
    validation_results = {}
    valid_run_ids = set()
    
    for run in manifest_runs:
        run_id = run['run_id']
        valid_run_ids.add(run_id)
        
        strategy, scenario, seed = parse_run_id_pattern(run_id)
        result = validate_run(run_id, strategy, scenario, seed, raw_dir, aggregated_dir)
        validation_results[run_id] = result
    
    # Find orphan files
    orphan_files = find_orphan_files(raw_dir, aggregated_dir, valid_run_ids)
    
    # Cross-check status
    status_mismatches = cross_check_status(manifest_runs, status_data, validation_results)
    
    # Calculate summary
    runs_valid = sum(1 for r in validation_results.values() if r.valid)
    runs_missing = sum(1 for r in validation_results.values() 
                      if not r.valid and any('not found' in e.lower() for e in r.errors))
    runs_duplicate = sum(1 for r in validation_results.values()
                        if not r.valid and any('duplicate' in e.lower() for e in r.errors))
    runs_invalid = len(validation_results) - runs_valid - runs_missing - runs_duplicate
    
    overall_status = "PASS" if runs_valid == len(manifest_runs) and \
                               len(orphan_files) == 0 and \
                               len(status_mismatches) == 0 else "FAIL"
    
    return {
        'overall_status': overall_status,
        'total_runs': len(manifest_runs),
        'runs_valid': runs_valid,
        'runs_missing': runs_missing,
        'runs_duplicate': runs_duplicate,
        'runs_invalid': runs_invalid,
        'orphan_files': len(orphan_files),
        'status_mismatches': len(status_mismatches),
        'validation_results': validation_results,
        'manifest_runs': manifest_runs
    }


def load_summary_json(json_path: Path) -> Dict:
    """Load and parse a summary JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_run(run_id: str, summary_json_path: Path) -> AggregatedRun:
    """Create an aggregated row from a summary JSON file."""
    data = load_summary_json(summary_json_path)
    
    # Extract identity
    config = data['config']
    summary = data['summary']
    
    # Parse run_id
    strategy, scenario, seed = parse_run_id_pattern(run_id)
    
    # Build aggregated row
    return AggregatedRun(
        # Identity
        run_id=run_id,
        strategy=strategy,
        scenario=scenario,
        seed=seed,
        
        # Primary metrics
        mae=summary.get('mae'),
        rmse=summary.get('rmse'),
        
        # Detection metrics
        detection_delay=summary.get('detection_delay'),
        drift_detections=summary.get('drift_detections', 0),
        anomaly_detections=summary.get('anomaly_detections', 0),
        false_positive_triggers=summary.get('false_positive_triggers', 0),
        
        # Adaptation metrics
        retraining_events=summary.get('retraining_events', 0),
        validation_skipped_events=summary.get('validation_skipped_events', 0),
        candidates_generated=summary.get('candidates_generated', 0),
        gate_accepts=summary.get('gate_accepts', 0),
        gate_rejects=summary.get('gate_rejects', 0),
        shadow_promotions=summary.get('shadow_promotions', 0),
        shadow_rejections=summary.get('shadow_rejections', 0),
        model_promoted_events=summary.get('model_promoted_events', 0),
        degraded_promotions=summary.get('degraded_promotions', 0),
        degraded_promotion_rate=summary.get('degraded_promotion_rate'),
        
        # Recovery metrics
        time_to_first_error_recovery=summary.get('time_to_first_error_recovery'),
        time_to_sustained_recovery=summary.get('time_to_sustained_recovery'),
        
        # Efficiency metrics
        total_retraining_time=summary.get('total_retraining_time', 0.0),
        total_shadow_evaluation_time=summary.get('total_shadow_evaluation_time', 0.0),
        total_adaptation_time=summary.get('total_adaptation_time', 0.0),
        mean_inference_latency=summary.get('mean_inference_latency'),
        
        # Configuration provenance
        stream_length=config.get('stream_length'),
        stream_mode=config.get('stream_mode'),
        scenario_onset_cycle_min=config.get('scenario_onset_cycle_min'),
        scenario_onset_cycle_max=config.get('scenario_onset_cycle_max'),
        train_fraction=config.get('train_fraction'),
        validation_fraction=config.get('validation_fraction'),
        retraining_interval=config.get('retraining_interval'),
        
        # File provenance
        raw_csv=data.get('raw_events', ''),
        summary_json=data.get('summary_json', '')
    )


def aggregate_results(manifest_path: Path, results_dir: Path) -> tuple[List[AggregatedRun], AggregationReport]:
    """Aggregate all verified runs into a single table."""
    
    # Step 1: Run verification
    verification = run_verification(manifest_path, results_dir)
    
    if verification['overall_status'] != 'PASS':
        print(f"\n{'='*60}")
        print("VERIFICATION FAILED")
        print(f"{'='*60}")
        print(f"Total Runs:       {verification['total_runs']}")
        print(f"Valid:            {verification['runs_valid']}")
        print(f"Missing:          {verification['runs_missing']}")
        print(f"Duplicate:        {verification['runs_duplicate']}")
        print(f"Invalid:          {verification['runs_invalid']}")
        print(f"Orphan Files:     {verification['orphan_files']}")
        print(f"Status Mismatches: {verification['status_mismatches']}")
        print(f"\nREFUSING TO AGGREGATE - Fix verification issues first")
        raise ValueError("Verification failed - cannot aggregate with invalid outputs")
    
    print(f"✅ Verification passed: {verification['runs_valid']}/{verification['total_runs']} runs valid")
    
    # Step 2: Aggregate each run
    print("\nAggregating runs...")
    aggregated_runs = []
    
    for run_id, validation_result in verification['validation_results'].items():
        if not validation_result.valid:
            continue
        
        if validation_result.summary_json_path is None:
            print(f"  ⚠️  Skipping {run_id}: no summary JSON")
            continue
        
        try:
            agg_run = aggregate_run(run_id, validation_result.summary_json_path)
            aggregated_runs.append(agg_run)
            print(f"  ✅ {run_id}")
        except Exception as e:
            print(f"  ❌ {run_id}: {e}")
            raise
    
    # Sort by strategy, scenario, seed for deterministic output
    aggregated_runs.sort(key=lambda r: (r.strategy, r.scenario, r.seed))
    
    # Step 3: Extract provenance from manifest
    manifest_runs = verification['manifest_runs']
    if manifest_runs:
        first_run = manifest_runs[0]
        git_commit = first_run.get('git_commit', 'unknown')
        dataset_checksum = first_run.get('dataset_checksum', 'unknown')
        python_version = first_run.get('python_version', 'unknown')
    else:
        git_commit = 'unknown'
        dataset_checksum = 'unknown'
        python_version = 'unknown'
    
    # Step 4: Create report
    report = AggregationReport(
        manifest_path=str(manifest_path),
        results_dir=str(results_dir),
        aggregation_timestamp=datetime.now().isoformat(),
        total_runs=verification['total_runs'],
        successfully_aggregated=len(aggregated_runs),
        git_commit=git_commit,
        dataset_checksum=dataset_checksum,
        python_version=python_version,
        verification_status=verification['overall_status'],
        runs_with_valid_outputs=verification['runs_valid'],
        runs_missing_outputs=verification['runs_missing'],
        runs_with_duplicate_outputs=verification['runs_duplicate'],
        runs_with_invalid_outputs=verification['runs_invalid'],
        orphan_files=verification['orphan_files'],
        status_mismatches=verification['status_mismatches']
    )
    
    return aggregated_runs, report


def write_aggregated_csv(runs: List[AggregatedRun], output_path: Path):
    """Write aggregated runs to CSV file."""
    if not runs:
        raise ValueError("No runs to aggregate")
    
    # Get field names from dataclass
    fieldnames = list(asdict(runs[0]).keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))


def write_aggregation_report(report: AggregationReport, runs: List[AggregatedRun], output_path: Path):
    """Write human-readable aggregation report."""
    lines = [
        "# Result Aggregation Report",
        "",
        f"**Generated:** {report.aggregation_timestamp}",
        f"**Manifest:** `{report.manifest_path}`",
        f"**Results Directory:** `{report.results_dir}`",
        "",
        "## Aggregation Summary",
        "",
        f"- **Total Manifest Runs:** {report.total_runs}",
        f"- **Successfully Aggregated:** {report.successfully_aggregated}",
        "",
        "## Provenance",
        "",
        f"- **Git Commit:** `{report.git_commit}`",
        f"- **Dataset Checksum:** `{report.dataset_checksum}`",
        f"- **Python Version:** `{report.python_version}`",
        "",
        "## Verification Status",
        "",
        f"- **Overall Status:** `{report.verification_status}`",
        f"- **Valid Outputs:** {report.runs_with_valid_outputs}",
        f"- **Missing Outputs:** {report.runs_missing_outputs}",
        f"- **Duplicate Outputs:** {report.runs_with_duplicate_outputs}",
        f"- **Invalid Outputs:** {report.runs_with_invalid_outputs}",
        f"- **Orphan Files:** {report.orphan_files}",
        f"- **Status Mismatches:** {report.status_mismatches}",
        "",
        "## Aggregated Runs",
        "",
        "| Run ID | Strategy | Scenario | Seed | MAE | RMSE |",
        "|--------|----------|----------|------|-----|------|"
    ]
    
    for run in runs:
        mae_str = f"{run.mae:.2f}" if run.mae is not None else "N/A"
        rmse_str = f"{run.rmse:.2f}" if run.rmse is not None else "N/A"
        lines.append(f"| {run.run_id} | {run.strategy} | {run.scenario} | {run.seed} | {mae_str} | {rmse_str} |")
    
    lines.extend([
        "",
        "## Metrics Included",
        "",
        "The aggregated CSV contains the following metrics for each run:",
        "",
        "### Identity Fields",
        "- `run_id`: Unique identifier for the run",
        "- `strategy`: Experimental strategy (e.g., static, proposed)",
        "- `scenario`: Drift scenario (e.g., gradual_drift)",
        "- `seed`: Random seed",
        "",
        "### Primary Performance Metrics",
        "- `mae`: Mean Absolute Error",
        "- `rmse`: Root Mean Squared Error",
        "",
        "### Detection Metrics",
        "- `detection_delay`: Samples from scenario onset to first detection",
        "- `drift_detections`: Number of drift detections",
        "- `anomaly_detections`: Number of anomaly detections",
        "- `false_positive_triggers`: Triggers before scenario onset",
        "",
        "### Adaptation Lifecycle Metrics",
        "- `retraining_events`: Number of retraining triggers",
        "- `validation_skipped_events`: Triggers where validation was skipped",
        "- `candidates_generated`: Number of candidate models trained",
        "- `gate_accepts`: Candidates passing performance gate",
        "- `gate_rejects`: Candidates rejected by performance gate",
        "- `shadow_promotions`: Candidates promoted after shadow evaluation",
        "- `shadow_rejections`: Candidates rejected during shadow evaluation",
        "- `model_promoted_events`: Total model promotions",
        "- `degraded_promotions`: Promotions with degraded performance",
        "- `degraded_promotion_rate`: Fraction of degraded promotions",
        "",
        "### Recovery Metrics",
        "- `time_to_first_error_recovery`: Samples to first error recovery",
        "- `time_to_sustained_recovery`: Samples to sustained error recovery",
        "",
        "### Efficiency Metrics",
        "- `total_retraining_time`: Total time spent retraining (seconds)",
        "- `total_shadow_evaluation_time`: Total time in shadow evaluation (seconds)",
        "- `total_adaptation_time`: Total adaptation time (seconds)",
        "- `mean_inference_latency`: Mean inference latency (seconds)",
        "",
        "### Configuration Provenance",
        "- `stream_length`: Length of data stream",
        "- `stream_mode`: Stream generation mode",
        "- `scenario_onset_cycle_min`: Minimum scenario onset cycle",
        "- `scenario_onset_cycle_max`: Maximum scenario onset cycle",
        "- `train_fraction`: Training data fraction",
        "- `validation_fraction`: Validation data fraction",
        "- `retraining_interval`: Retraining interval",
        "",
        "### File Provenance",
        "- `raw_csv`: Path to raw events CSV",
        "- `summary_json`: Path to summary JSON",
        ""
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate verified experiment results"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to experiment manifest CSV"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to results directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file (default: results-dir/aggregated_results.csv)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.manifest.exists():
        print(f"ERROR: Manifest file not found: {args.manifest}", file=sys.stderr)
        return 2
    
    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}", file=sys.stderr)
        return 2
    
    # Set default output
    if args.output is None:
        args.output = args.results_dir / "aggregated_results.csv"
    
    report_path = args.output.with_suffix('.md')
    
    try:
        # Run aggregation
        print(f"Aggregating results from: {args.results_dir}")
        print(f"Using manifest: {args.manifest}")
        
        runs, report = aggregate_results(args.manifest, args.results_dir)
        
        # Write outputs
        write_aggregated_csv(runs, args.output)
        write_aggregation_report(report, runs, report_path)
        
        print(f"\n{'='*60}")
        print("AGGREGATION COMPLETE")
        print(f"{'='*60}")
        print(f"Runs Aggregated:  {len(runs)}/{report.total_runs}")
        print(f"Output CSV:       {args.output}")
        print(f"Report:           {report_path}")
        print(f"\nProvenance:")
        print(f"  Git commit:     {report.git_commit}")
        print(f"  Dataset:        {report.dataset_checksum}")
        print(f"  Python:         {report.python_version}")
        
        return 0
        
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nERROR: Aggregation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
