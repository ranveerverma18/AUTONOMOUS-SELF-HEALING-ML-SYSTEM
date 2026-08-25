#!/usr/bin/env python3
"""
Experiment Completion Verifier

Validates that every manifest run has exactly the expected outputs.
Does not trust the status file alone - independently inspects the filesystem.

Usage:
    python -m scripts.matrix_orchestration.verify_completion \
        --manifest <manifest.csv> \
        --results-dir <results-directory> \
        [--strict]

Exit codes:
    0 = all checks passed
    1 = validation failure
    2 = invalid CLI/input/configuration
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re


@dataclass
class ValidationResult:
    """Result of validating a single manifest run."""
    run_id: str
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_csv_path: Optional[Path] = None
    summary_json_path: Optional[Path] = None


@dataclass
class VerificationReport:
    """Complete verification report."""
    manifest_path: str
    results_dir: str
    timestamp: str
    total_manifest_runs: int
    runs_with_valid_outputs: int
    runs_missing_outputs: int
    runs_with_duplicate_outputs: int
    runs_with_invalid_outputs: int
    orphan_raw_files: int
    orphan_summary_files: int
    status_mismatches: int
    overall_status: str
    run_details: Dict[str, ValidationResult] = field(default_factory=dict)
    orphan_files: List[str] = field(default_factory=list)
    status_mismatch_details: List[str] = field(default_factory=list)


# Expected CSV columns (55 columns)
REQUIRED_CSV_COLUMNS = [
    "run_id", "seed", "strategy", "scenario",
    "sample_index", "engine_id", "cycle", "event_index",
    "scenario_active", "degradation_started",
    "actual_rul", "predicted_rul", "absolute_error", "squared_error",
    "rolling_mae", "rolling_rmse",
    "anomaly_detected", "feature_drift_detected", "concept_drift_detected",
    "drift_score", "drift_trigger",
    "retraining_triggered", "retraining_started", "retraining_completed",
    "validation_skipped", "validation_skip_reason",
    "candidate_generated", "candidate_id",
    "candidate_mae", "candidate_rmse",
    "production_mae", "production_rmse", "improvement",
    "gate_passed", "gate_rejected",
    "shadow_started", "shadow_completed", "shadow_passed", "shadow_rejected", "shadow_result",
    "model_promoted", "promotion_decision", "degraded_promotion",
    "model_version", "training_time", "shadow_evaluation_time", "inference_latency",
    "val_buffer_rows", "val_train_rows", "val_validation_rows",
    "val_buffer_units", "val_train_units", "val_validation_units",
    "val_unit_disjoint", "val_unit_intersection_count", "val_training_time"
]


def parse_run_id_pattern(run_id: str) -> Tuple[str, str, int]:
    """Parse run_id into strategy, scenario, seed.
    
    Example: 'static_gradual_drift_seed42' -> ('static', 'gradual_drift', 42)
    Example: 'naive_adaptive_gradual_drift_seed42' -> ('naive_adaptive', 'gradual_drift', 42)
    
    Uses authoritative strategy vocabulary to correctly handle multi-word strategies.
    """
    # Authoritative strategy vocabulary
    KNOWN_STRATEGIES = ['static', 'scheduled', 'naive_adaptive', 'proposed']
    
    # Extract seed first (unambiguous)
    seed_match = re.search(r'_seed(\d+)$', run_id)
    if not seed_match:
        raise ValueError(f"Invalid run_id format (no seed): {run_id}")
    
    seed = int(seed_match.group(1))
    # Remove seed suffix to get strategy_scenario
    strategy_scenario = run_id[:seed_match.start()]
    
    # Try each known strategy (longest first to handle prefixes correctly)
    for strategy in sorted(KNOWN_STRATEGIES, key=len, reverse=True):
        if strategy_scenario.startswith(strategy + '_'):
            scenario = strategy_scenario[len(strategy) + 1:]  # +1 for underscore
            return strategy, scenario, seed
    
    raise ValueError(f"Invalid run_id format (unknown strategy): {run_id}")


def find_matching_files(run_id: str, directory: Path, extension: str) -> List[Path]:
    """Find files matching run_id pattern with timestamp.
    
    Pattern: {run_id}_YYYYMMDD_HHMMSS{extension}
    """
    pattern = f"{run_id}_*{extension}"
    return list(directory.glob(pattern))


def validate_csv_file(csv_path: Path) -> Tuple[bool, List[str]]:
    """Validate raw CSV file structure and content."""
    errors = []
    
    # Check file exists and is readable
    if not csv_path.exists():
        errors.append(f"CSV file does not exist: {csv_path}")
        return False, errors
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check columns
            if reader.fieldnames is None:
                errors.append("CSV has no header row")
                return False, errors
            
            actual_columns = set(reader.fieldnames)
            required_columns = set(REQUIRED_CSV_COLUMNS)
            
            missing_columns = required_columns - actual_columns
            if missing_columns:
                errors.append(f"CSV missing required columns: {sorted(missing_columns)}")
                return False, errors
            
            # Check non-zero rows
            row_count = sum(1 for _ in reader)
            if row_count == 0:
                errors.append("CSV file is empty (no data rows)")
                return False, errors
            
    except Exception as e:
        errors.append(f"Failed to read CSV: {str(e)}")
        return False, errors
    
    return True, []


def validate_json_file(json_path: Path, expected_strategy: str, 
                       expected_scenario: str, expected_seed: int) -> Tuple[bool, List[str]]:
    """Validate summary JSON file structure and identity."""
    errors = []
    
    # Check file exists and is readable
    if not json_path.exists():
        errors.append(f"JSON file does not exist: {json_path}")
        return False, errors
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check top-level structure
        if not isinstance(data, dict):
            errors.append("JSON root is not an object")
            return False, errors
        
        # Check required fields
        if "run_id" not in data:
            errors.append("JSON missing 'run_id' field")
            return False, errors
        
        if "config" not in data:
            errors.append("JSON missing 'config' field")
            return False, errors
        
        if "summary" not in data:
            errors.append("JSON missing 'summary' field")
            return False, errors
        
        config = data["config"]
        if not isinstance(config, dict):
            errors.append("JSON 'config' is not an object")
            return False, errors
        
        # Validate identity fields
        if config.get("strategy") != expected_strategy:
            errors.append(f"JSON strategy mismatch: expected {expected_strategy}, got {config.get('strategy')}")
        
        if config.get("scenario") != expected_scenario:
            errors.append(f"JSON scenario mismatch: expected {expected_scenario}, got {config.get('scenario')}")
        
        if config.get("seed") != expected_seed:
            errors.append(f"JSON seed mismatch: expected {expected_seed}, got {config.get('seed')}")
        
        # Check summary is dict
        if not isinstance(data["summary"], dict):
            errors.append("JSON 'summary' is not an object")
            return False, errors
        
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {str(e)}")
        return False, errors
    except Exception as e:
        errors.append(f"Failed to read JSON: {str(e)}")
        return False, errors
    
    return len(errors) == 0, errors


def validate_run(run_id: str, strategy: str, scenario: str, seed: int,
                 raw_dir: Path, aggregated_dir: Path) -> ValidationResult:
    """Validate outputs for a single manifest run."""
    result = ValidationResult(run_id=run_id, valid=True)
    
    # Find matching files
    raw_files = find_matching_files(run_id, raw_dir, "_events.csv")
    json_files = find_matching_files(run_id, aggregated_dir, "_summary.json")
    
    # Check for duplicates
    if len(raw_files) > 1:
        result.valid = False
        result.errors.append(f"Duplicate raw CSV files found: {[f.name for f in raw_files]}")
    elif len(raw_files) == 0:
        result.valid = False
        result.errors.append("No raw CSV file found")
    else:
        result.raw_csv_path = raw_files[0]
    
    if len(json_files) > 1:
        result.valid = False
        result.errors.append(f"Duplicate summary JSON files found: {[f.name for f in json_files]}")
    elif len(json_files) == 0:
        result.valid = False
        result.errors.append("No summary JSON file found")
    else:
        result.summary_json_path = json_files[0]
    
    # If files found, validate content
    if result.raw_csv_path:
        csv_valid, csv_errors = validate_csv_file(result.raw_csv_path)
        if not csv_valid:
            result.valid = False
            result.errors.extend(csv_errors)
    
    if result.summary_json_path:
        json_valid, json_errors = validate_json_file(
            result.summary_json_path, strategy, scenario, seed
        )
        if not json_valid:
            result.valid = False
            result.errors.extend(json_errors)
    
    return result


def load_manifest(manifest_path: Path) -> List[Dict]:
    """Load experiment manifest."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_status_file(status_path: Path) -> Dict[str, Dict]:
    """Load execution status file."""
    if not status_path.exists():
        return {}
    
    with open(status_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return {row['run_id']: row for row in reader}


def find_orphan_files(raw_dir: Path, aggregated_dir: Path, 
                      valid_run_ids: Set[str]) -> List[str]:
    """Find output files that don't correspond to any manifest run."""
    orphans = []
    
    # Check raw CSV files
    for csv_file in raw_dir.glob("*_events.csv"):
        # Extract run_id pattern from filename
        # Pattern: {strategy}_{scenario}_seed{seed}_{timestamp}_events.csv
        match = re.match(r'^(.+?)_(\d{8}_\d{6})_events\.csv$', csv_file.name)
        if match:
            run_id_prefix = match.group(1)
            # Check if this matches any valid run_id
            matched = any(run_id.startswith(run_id_prefix.rsplit('_', 2)[0] + '_' + 
                                           run_id_prefix.rsplit('_', 2)[1] + '_' +
                                           run_id_prefix.rsplit('_', 1)[0].split('_')[-1])
                         for run_id in valid_run_ids)
            if not matched:
                # Try simpler pattern: extract everything before timestamp
                base_pattern = re.match(r'^(.+?)_\d{8}_\d{6}_events\.csv$', csv_file.name)
                if base_pattern:
                    base_run_id = base_pattern.group(1)
                    if base_run_id not in valid_run_ids:
                        orphans.append(str(csv_file.relative_to(raw_dir.parent.parent)))
    
    # Check summary JSON files
    for json_file in aggregated_dir.glob("*_summary.json"):
        match = re.match(r'^(.+?)_\d{8}_\d{6}_summary\.json$', json_file.name)
        if match:
            base_run_id = match.group(1)
            if base_run_id not in valid_run_ids:
                orphans.append(str(json_file.relative_to(aggregated_dir.parent.parent)))
    
    return orphans


def cross_check_status(manifest_runs: List[Dict], status_data: Dict[str, Dict],
                       validation_results: Dict[str, ValidationResult]) -> List[str]:
    """Cross-check manifest, status file, and filesystem.
    
    Note: Manifest status represents the PLANNED state and remains immutable.
    Actual execution state is tracked in matrix_execution_status.csv.
    """
    mismatches = []
    
    for run in manifest_runs:
        run_id = run['run_id']
        exec_status = status_data.get(run_id, {}).get('status', 'NOT_IN_STATUS')
        validation = validation_results.get(run_id)
        
        if not validation:
            continue
        
        # Execution status says SUCCESS but output missing
        if exec_status == 'SUCCESS' and not validation.valid:
            if 'No raw CSV file found' in ' '.join(validation.errors) or \
               'No summary JSON file found' in ' '.join(validation.errors):
                mismatches.append(
                    f"{run_id}: Execution status=SUCCESS but outputs missing"
                )
        
        # Execution status says FAILED but output exists and is valid
        if exec_status == 'FAILED' and validation.valid:
            mismatches.append(
                f"{run_id}: Execution status=FAILED but valid outputs exist"
            )
        
        # Valid output exists but no execution status entry
        if exec_status == 'NOT_IN_STATUS' and validation.valid:
            mismatches.append(
                f"{run_id}: Valid outputs exist but no execution status entry"
            )
        
        # NOTE: Manifest status=PLANNED is NOT a mismatch.
        # The manifest represents the immutable plan created before execution.
        # Actual execution state is tracked in matrix_execution_status.csv.
    
    return mismatches


def generate_json_report(report: VerificationReport, output_path: Path):
    """Generate machine-readable JSON report."""
    data = {
        "manifest": report.manifest_path,
        "results_dir": report.results_dir,
        "timestamp": report.timestamp,
        "total_manifest_runs": report.total_manifest_runs,
        "runs_with_valid_outputs": report.runs_with_valid_outputs,
        "runs_missing_outputs": report.runs_missing_outputs,
        "runs_with_duplicate_outputs": report.runs_with_duplicate_outputs,
        "runs_with_invalid_outputs": report.runs_with_invalid_outputs,
        "orphan_raw_files": report.orphan_raw_files,
        "orphan_summary_files": report.orphan_summary_files,
        "status_mismatches": report.status_mismatches,
        "overall_status": report.overall_status,
        "run_details": {
            run_id: {
                "valid": result.valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "raw_csv": str(result.raw_csv_path) if result.raw_csv_path else None,
                "summary_json": str(result.summary_json_path) if result.summary_json_path else None
            }
            for run_id, result in report.run_details.items()
        },
        "orphan_files": report.orphan_files,
        "status_mismatch_details": report.status_mismatch_details
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def generate_markdown_report(report: VerificationReport, output_path: Path):
    """Generate human-readable markdown report."""
    lines = [
        "# Experiment Completion Verification Report",
        "",
        f"**Generated:** {report.timestamp}",
        f"**Manifest:** `{report.manifest_path}`",
        f"**Results Directory:** `{report.results_dir}`",
        "",
        "## Summary",
        "",
        f"- **Total Manifest Runs:** {report.total_manifest_runs}",
        f"- **Valid Outputs:** {report.runs_with_valid_outputs}",
        f"- **Missing Outputs:** {report.runs_missing_outputs}",
        f"- **Duplicate Outputs:** {report.runs_with_duplicate_outputs}",
        f"- **Invalid Outputs:** {report.runs_with_invalid_outputs}",
        f"- **Orphan Files:** {report.orphan_raw_files + report.orphan_summary_files}",
        f"- **Status Mismatches:** {report.status_mismatches}",
        "",
        f"**Overall Status:** `{report.overall_status}`",
        "",
    ]
    
    # Run details
    if report.run_details:
        lines.extend([
            "## Run Validation Details",
            ""
        ])
        
        for run_id, result in sorted(report.run_details.items()):
            status_icon = "✅" if result.valid else "❌"
            lines.append(f"### {status_icon} {run_id}")
            lines.append("")
            
            if result.valid:
                lines.append("**Status:** VALID")
                if result.raw_csv_path:
                    lines.append(f"- Raw CSV: `{result.raw_csv_path.name}`")
                if result.summary_json_path:
                    lines.append(f"- Summary JSON: `{result.summary_json_path.name}`")
            else:
                lines.append("**Status:** INVALID")
                lines.append("")
                lines.append("**Errors:**")
                for error in result.errors:
                    lines.append(f"- {error}")
            
            if result.warnings:
                lines.append("")
                lines.append("**Warnings:**")
                for warning in result.warnings:
                    lines.append(f"- {warning}")
            
            lines.append("")
    
    # Orphan files
    if report.orphan_files:
        lines.extend([
            "## Orphan Files",
            "",
            "The following files do not correspond to any manifest run:",
            ""
        ])
        for orphan in sorted(report.orphan_files):
            lines.append(f"- `{orphan}`")
        lines.append("")
    
    # Status mismatches
    if report.status_mismatch_details:
        lines.extend([
            "## Status Mismatches",
            "",
            "Inconsistencies between manifest, status file, and filesystem:",
            ""
        ])
        for mismatch in report.status_mismatch_details:
            lines.append(f"- {mismatch}")
        lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Verify experiment completion against manifest"
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
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any validation fails"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for verification reports (default: results-dir)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.manifest.exists():
        print(f"ERROR: Manifest file not found: {args.manifest}", file=sys.stderr)
        return 2
    
    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}", file=sys.stderr)
        return 2
    
    raw_dir = args.results_dir / "raw"
    aggregated_dir = args.results_dir / "aggregated"
    
    if not raw_dir.exists():
        print(f"ERROR: Raw directory not found: {raw_dir}", file=sys.stderr)
        return 2
    
    if not aggregated_dir.exists():
        print(f"ERROR: Aggregated directory not found: {aggregated_dir}", file=sys.stderr)
        return 2
    
    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    manifest_runs = load_manifest(args.manifest)
    print(f"Found {len(manifest_runs)} runs in manifest")
    
    # Load status file if exists
    status_path = args.results_dir / "matrix_execution_status.csv"
    status_data = load_status_file(status_path)
    if status_data:
        print(f"Loaded {len(status_data)} entries from status file")
    
    # Validate each run
    print("\nValidating outputs...")
    validation_results = {}
    valid_run_ids = set()
    
    for run in manifest_runs:
        run_id = run['run_id']
        valid_run_ids.add(run_id)
        
        try:
            strategy, scenario, seed = parse_run_id_pattern(run_id)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        
        result = validate_run(run_id, strategy, scenario, seed, raw_dir, aggregated_dir)
        validation_results[run_id] = result
        
        status = "✅ VALID" if result.valid else "❌ INVALID"
        print(f"  {run_id}: {status}")
        if not result.valid:
            for error in result.errors:
                print(f"    - {error}")
    
    # Find orphan files
    print("\nChecking for orphan files...")
    orphan_files = find_orphan_files(raw_dir, aggregated_dir, valid_run_ids)
    if orphan_files:
        print(f"Found {len(orphan_files)} orphan files:")
        for orphan in orphan_files:
            print(f"  - {orphan}")
    else:
        print("No orphan files found")
    
    # Cross-check status
    print("\nCross-checking manifest, status, and filesystem...")
    status_mismatches = cross_check_status(manifest_runs, status_data, validation_results)
    if status_mismatches:
        print(f"Found {len(status_mismatches)} mismatches:")
        for mismatch in status_mismatches:
            print(f"  - {mismatch}")
    else:
        print("No mismatches found")
    
    # Generate report
    runs_valid = sum(1 for r in validation_results.values() if r.valid)
    runs_missing = sum(1 for r in validation_results.values() 
                      if not r.valid and any('not found' in e.lower() for e in r.errors))
    runs_duplicate = sum(1 for r in validation_results.values()
                        if not r.valid and any('duplicate' in e.lower() for e in r.errors))
    runs_invalid = len(validation_results) - runs_valid - runs_missing - runs_duplicate
    
    # Count orphan types
    orphan_csv = sum(1 for o in orphan_files if o.endswith('.csv'))
    orphan_json = sum(1 for o in orphan_files if o.endswith('.json'))
    
    overall_status = "PASS" if runs_valid == len(manifest_runs) and \
                               len(orphan_files) == 0 and \
                               len(status_mismatches) == 0 else "FAIL"
    
    report = VerificationReport(
        manifest_path=str(args.manifest),
        results_dir=str(args.results_dir),
        timestamp=datetime.now().isoformat(),
        total_manifest_runs=len(manifest_runs),
        runs_with_valid_outputs=runs_valid,
        runs_missing_outputs=runs_missing,
        runs_with_duplicate_outputs=runs_duplicate,
        runs_with_invalid_outputs=runs_invalid,
        orphan_raw_files=orphan_csv,
        orphan_summary_files=orphan_json,
        status_mismatches=len(status_mismatches),
        overall_status=overall_status,
        run_details=validation_results,
        orphan_files=orphan_files,
        status_mismatch_details=status_mismatches
    )
    
    # Save reports
    output_dir = args.output_dir if args.output_dir else args.results_dir
    json_path = output_dir / "verification_report.json"
    md_path = output_dir / "verification_report.md"
    
    generate_json_report(report, json_path)
    generate_markdown_report(report, md_path)
    
    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Runs:       {report.total_manifest_runs}")
    print(f"Valid:            {report.runs_with_valid_outputs}")
    print(f"Missing:          {report.runs_missing_outputs}")
    print(f"Duplicate:        {report.runs_with_duplicate_outputs}")
    print(f"Invalid:          {report.runs_with_invalid_outputs}")
    print(f"Orphan Files:     {report.orphan_raw_files + report.orphan_summary_files}")
    print(f"Status Mismatches: {report.status_mismatches}")
    print(f"\nOverall Status:   {report.overall_status}")
    print(f"\nReports saved:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    
    # Exit code
    if overall_status == "FAIL" and args.strict:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
