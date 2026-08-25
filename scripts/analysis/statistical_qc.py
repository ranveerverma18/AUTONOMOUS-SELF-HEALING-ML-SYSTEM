#!/usr/bin/env python3
"""
Statistical Quality Control Tool

Validates aggregated experiment results before statistical hypothesis testing.
Checks structural validity, data completeness, and protocol consistency.

Usage:
    python -m scripts.analysis.statistical_qc \
        --aggregated <aggregated_results.csv> \
        --manifest <experiment_manifest.csv> \
        [--output <qc_report.json>]

Exit codes:
    0 = QC passed
    1 = QC failed
    2 = invalid CLI/input/configuration
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
import math

# Valid values from source code
VALID_STRATEGIES = {"static", "scheduled", "naive_adaptive", "proposed"}
VALID_SCENARIOS = {
    "gradual_drift", "sudden_spike", "high_noise", "sensor_failure",
    "concept_drift", "correlated_drift", "intermittent_spikes", "drift_recovery"
}
VALID_STREAM_MODES = {"research", "interleaved", "legacy"}

# Required columns from AggregatedRun dataclass
REQUIRED_COLUMNS = [
    # Identity
    "run_id", "strategy", "scenario", "seed",
    # Primary metrics
    "mae", "rmse",
    # Detection
    "detection_delay", "drift_detections", "anomaly_detections", "false_positive_triggers",
    # Adaptation
    "retraining_events", "validation_skipped_events", "candidates_generated",
    "gate_accepts", "gate_rejects", "shadow_promotions", "shadow_rejections",
    "model_promoted_events", "degraded_promotions", "degraded_promotion_rate",
    # Recovery
    "time_to_first_error_recovery", "time_to_sustained_recovery",
    # Efficiency
    "total_retraining_time", "total_shadow_evaluation_time", "total_adaptation_time",
    "mean_inference_latency",
    # Configuration
    "stream_length", "stream_mode", "scenario_onset_cycle_min", "scenario_onset_cycle_max",
    "train_fraction", "validation_fraction", "retraining_interval",
    # Provenance
    "raw_csv", "summary_json"
]

# Columns that are allowed to be NULL/None (semantically valid)
NULLABLE_COLUMNS = {
    "mae",  # Can be None if no errors computed
    "rmse",  # Can be None if no errors computed
    "detection_delay",  # Can be None if no detection occurred
    "degraded_promotion_rate",  # Can be None if no promotions
    "time_to_first_error_recovery",  # Can be None if no recovery
    "time_to_sustained_recovery",  # Can be None if no sustained recovery
    "mean_inference_latency"  # Can be None if no latencies recorded
}

# Columns that must be non-negative integers (counts)
COUNT_COLUMNS = {
    "drift_detections", "anomaly_detections", "false_positive_triggers",
    "retraining_events", "validation_skipped_events", "candidates_generated",
    "gate_accepts", "gate_rejects", "shadow_promotions", "shadow_rejections",
    "model_promoted_events", "degraded_promotions"
}

# Columns that must be non-negative floats (time/duration)
NONNEGATIVE_FLOAT_COLUMNS = {
    "total_retraining_time", "total_shadow_evaluation_time", "total_adaptation_time"
}

# Columns that must be in [0, 1] range (fractions/rates)
FRACTION_COLUMNS = {
    "degraded_promotion_rate", "train_fraction", "validation_fraction"
}


@dataclass
class QCIssue:
    """Single QC validation issue."""
    severity: str  # "ERROR" or "WARNING"
    category: str  # e.g., "structural", "numeric", "consistency"
    run_id: Optional[str]
    column: Optional[str]
    message: str
    value: Optional[str] = None


@dataclass
class QCReport:
    """Statistical QC validation report."""
    aggregated_file: str
    manifest_file: Optional[str]
    total_rows: int
    passed: bool
    issues: List[QCIssue] = field(default_factory=list)
    
    # Validation summaries
    missing_columns: List[str] = field(default_factory=list)
    duplicate_run_ids: List[str] = field(default_factory=list)
    duplicate_combinations: List[str] = field(default_factory=list)
    unexpected_strategies: Set[str] = field(default_factory=set)
    unexpected_scenarios: Set[str] = field(default_factory=set)
    unexpected_seeds: Set[int] = field(default_factory=set)
    nan_values: int = 0
    inf_values: int = 0
    config_inconsistencies: int = 0
    untraceable_runs: int = 0
    missing_from_manifest: List[str] = field(default_factory=list)
    extra_in_aggregated: List[str] = field(default_factory=list)
    
    def add_error(self, category: str, message: str, run_id: Optional[str] = None,
                  column: Optional[str] = None, value: Optional[str] = None):
        """Add an error issue."""
        self.issues.append(QCIssue("ERROR", category, run_id, column, message, value))
        self.passed = False
    
    def add_warning(self, category: str, message: str, run_id: Optional[str] = None,
                    column: Optional[str] = None, value: Optional[str] = None):
        """Add a warning issue."""
        self.issues.append(QCIssue("WARNING", category, run_id, column, message, value))


def is_null_value(value: str) -> bool:
    """Check if a CSV value represents null/None."""
    return value == "" or value.lower() in ("none", "null", "na", "n/a")


def is_numeric_valid(value: str) -> bool:
    """Check if a non-null value is a valid number."""
    if is_null_value(value):
        return True  # Null is handled separately
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def check_required_columns(headers: List[str]) -> List[str]:
    """Check for missing required columns."""
    headers_set = set(headers)
    required_set = set(REQUIRED_COLUMNS)
    return sorted(required_set - headers_set)


def check_duplicate_run_ids(rows: List[Dict]) -> List[str]:
    """Find duplicate run_ids."""
    seen = {}
    duplicates = []
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id in seen:
            if run_id not in duplicates:
                duplicates.append(run_id)
        else:
            seen[run_id] = True
    return duplicates


def check_duplicate_combinations(rows: List[Dict]) -> List[str]:
    """Find duplicate (strategy, scenario, seed) combinations."""
    seen = {}
    duplicates = []
    for row in rows:
        key = (row.get("strategy", ""), row.get("scenario", ""), row.get("seed", ""))
        if key in seen:
            desc = f"{key[0]}_{key[1]}_seed{key[2]}"
            if desc not in duplicates:
                duplicates.append(desc)
        else:
            seen[key] = True
    return duplicates


def validate_numeric_values(row: Dict, report: QCReport):
    """Validate numeric values in a row."""
    run_id = row.get("run_id", "unknown")
    
    for column in row.keys():
        value = row[column]
        
        # Skip non-numeric columns
        if column in {"run_id", "strategy", "scenario", "stream_mode", "raw_csv", "summary_json"}:
            continue
        
        # Check if null is allowed
        if is_null_value(value):
            if column not in NULLABLE_COLUMNS:
                report.add_error(
                    "missing_value",
                    f"Required column has missing value",
                    run_id, column, value
                )
            continue
        
        # Check if numeric
        if not is_numeric_valid(value):
            report.add_error(
                "numeric_invalid",
                f"Non-numeric value in numeric column",
                run_id, column, value
            )
            continue
        
        try:
            num_value = float(value)
            
            # Check for NaN
            if math.isnan(num_value):
                report.nan_values += 1
                report.add_error(
                    "numeric_invalid",
                    f"NaN value detected",
                    run_id, column, value
                )
                continue
            
            # Check for infinity
            if math.isinf(num_value):
                report.inf_values += 1
                report.add_error(
                    "numeric_invalid",
                    f"Infinity value detected",
                    run_id, column, value
                )
                continue
            
            # Check count columns (must be non-negative integers)
            if column in COUNT_COLUMNS:
                if num_value < 0:
                    report.add_error(
                        "numeric_range",
                        f"Count column has negative value",
                        run_id, column, value
                    )
                if not float(value).is_integer():
                    report.add_error(
                        "numeric_type",
                        f"Count column has non-integer value",
                        run_id, column, value
                    )
            
            # Check non-negative float columns
            if column in NONNEGATIVE_FLOAT_COLUMNS:
                if num_value < 0:
                    report.add_error(
                        "numeric_range",
                        f"Non-negative column has negative value",
                        run_id, column, value
                    )
            
            # Check fraction columns (must be in [0, 1])
            if column in FRACTION_COLUMNS:
                if not (0 <= num_value <= 1):
                    report.add_error(
                        "numeric_range",
                        f"Fraction column out of [0,1] range",
                        run_id, column, value
                    )
            
            # Check seed (must be positive integer)
            if column == "seed":
                if num_value <= 0 or not float(value).is_integer():
                    report.add_error(
                        "numeric_range",
                        f"Seed must be positive integer",
                        run_id, column, value
                    )
            
            # Check stream_length (must be positive)
            if column == "stream_length":
                if num_value <= 0:
                    report.add_error(
                        "numeric_range",
                        f"stream_length must be positive",
                        run_id, column, value
                    )
            
            # Check scenario onset cycles
            if column in {"scenario_onset_cycle_min", "scenario_onset_cycle_max"}:
                if num_value < 1:
                    report.add_error(
                        "numeric_range",
                        f"Scenario onset cycle must be at least 1",
                        run_id, column, value
                    )
            
            # Check retraining_interval (must be positive)
            if column == "retraining_interval":
                if num_value <= 0:
                    report.add_error(
                        "numeric_range",
                        f"retraining_interval must be positive",
                        run_id, column, value
                    )
        
        except Exception as e:
            report.add_error(
                "numeric_invalid",
                f"Error validating numeric value: {str(e)}",
                run_id, column, value
            )


def validate_categorical_values(row: Dict, report: QCReport):
    """Validate categorical/enum values."""
    run_id = row.get("run_id", "unknown")
    
    # Check strategy
    strategy = row.get("strategy", "")
    if strategy and strategy not in VALID_STRATEGIES:
        report.unexpected_strategies.add(strategy)
        report.add_error(
            "categorical_invalid",
            f"Unknown strategy: {strategy}",
            run_id, "strategy", strategy
        )
    
    # Check scenario
    scenario = row.get("scenario", "")
    if scenario and scenario not in VALID_SCENARIOS:
        report.unexpected_scenarios.add(scenario)
        report.add_error(
            "categorical_invalid",
            f"Unknown scenario: {scenario}",
            run_id, "scenario", scenario
        )
    
    # Check stream_mode
    stream_mode = row.get("stream_mode", "")
    if stream_mode and stream_mode not in VALID_STREAM_MODES:
        report.add_error(
            "categorical_invalid",
            f"Unknown stream_mode: {stream_mode}",
            run_id, "stream_mode", stream_mode
        )


def validate_configuration_consistency(rows: List[Dict], report: QCReport):
    """Check that configuration parameters are consistent across runs."""
    if not rows:
        return
    
    # Configuration parameters that should be consistent across all runs
    config_params = [
        "stream_length", "stream_mode", "scenario_onset_cycle_min", "scenario_onset_cycle_max",
        "train_fraction", "validation_fraction", "retraining_interval"
    ]
    
    reference = rows[0]
    for param in config_params:
        ref_value = reference.get(param, "")
        
        for row in rows[1:]:
            value = row.get(param, "")
            if value != ref_value:
                report.config_inconsistencies += 1
                report.add_warning(
                    "config_inconsistent",
                    f"Configuration parameter '{param}' varies across runs: "
                    f"{ref_value} vs {value}",
                    row.get("run_id", "unknown"), param
                )


def validate_against_manifest(aggregated_rows: List[Dict], manifest_path: Path,
                              report: QCReport):
    """Validate aggregated results against manifest."""
    # Load manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)
    
    # Extract run_ids
    manifest_run_ids = set(row['run_id'] for row in manifest_rows)
    aggregated_run_ids = set(row['run_id'] for row in aggregated_rows)
    
    # Check for missing runs
    missing = manifest_run_ids - aggregated_run_ids
    if missing:
        report.missing_from_manifest = sorted(missing)
        for run_id in missing:
            report.add_error(
                "coverage",
                f"Run in manifest but not in aggregated results",
                run_id
            )
    
    # Check for extra runs
    extra = aggregated_run_ids - manifest_run_ids
    if extra:
        report.extra_in_aggregated = sorted(extra)
        for run_id in extra:
            report.add_error(
                "coverage",
                f"Run in aggregated results but not in manifest",
                run_id
            )
    
    # Check provenance consistency
    manifest_lookup = {row['run_id']: row for row in manifest_rows}
    for agg_row in aggregated_rows:
        run_id = agg_row['run_id']
        if run_id not in manifest_lookup:
            continue
        
        manifest_row = manifest_lookup[run_id]
        
        # Check strategy/scenario/seed match
        for field in ['strategy', 'scenario', 'seed']:
            agg_val = str(agg_row.get(field, ''))
            man_val = str(manifest_row.get(field, ''))
            if agg_val != man_val:
                report.add_error(
                    "provenance",
                    f"Identity field mismatch with manifest: {field}",
                    run_id, field, f"aggregated={agg_val}, manifest={man_val}"
                )


def validate_traceability(rows: List[Dict], report: QCReport):
    """Validate that file provenance paths are present."""
    for row in rows:
        run_id = row.get("run_id", "unknown")
        
        # Check raw_csv path
        raw_csv = row.get("raw_csv", "")
        if not raw_csv:
            report.untraceable_runs += 1
            report.add_error(
                "traceability",
                f"Missing raw_csv path",
                run_id, "raw_csv"
            )
        elif run_id not in raw_csv:
            report.add_warning(
                "traceability",
                f"run_id not found in raw_csv path",
                run_id, "raw_csv", raw_csv
            )
        
        # Check summary_json path
        summary_json = row.get("summary_json", "")
        if not summary_json:
            report.untraceable_runs += 1
            report.add_error(
                "traceability",
                f"Missing summary_json path",
                run_id, "summary_json"
            )
        elif run_id not in summary_json:
            report.add_warning(
                "traceability",
                f"run_id not found in summary_json path",
                run_id, "summary_json", summary_json
            )


def perform_qc(aggregated_file: Path, manifest_file: Optional[Path] = None) -> QCReport:
    """Perform statistical QC on aggregated results."""
    report = QCReport(
        aggregated_file=str(aggregated_file),
        manifest_file=str(manifest_file) if manifest_file else None,
        total_rows=0,
        passed=True
    )
    
    # Load aggregated results
    try:
        with open(aggregated_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
    except Exception as e:
        report.add_error("structural", f"Failed to read aggregated file: {e}")
        return report
    
    report.total_rows = len(rows)
    
    if not headers:
        report.add_error("structural", "No headers found in aggregated file")
        return report
    
    if report.total_rows == 0:
        report.add_error("structural", "No data rows in aggregated file")
        return report
    
    # 1. Check required columns
    missing_cols = check_required_columns(headers)
    if missing_cols:
        report.missing_columns = missing_cols
        for col in missing_cols:
            report.add_error("structural", f"Missing required column: {col}", column=col)
    
    # 2. Check for duplicate run_ids
    duplicate_ids = check_duplicate_run_ids(rows)
    if duplicate_ids:
        report.duplicate_run_ids = duplicate_ids
        for run_id in duplicate_ids:
            report.add_error("structural", f"Duplicate run_id", run_id)
    
    # 3. Check for duplicate (strategy, scenario, seed) combinations
    duplicate_combos = check_duplicate_combinations(rows)
    if duplicate_combos:
        report.duplicate_combinations = duplicate_combos
        for combo in duplicate_combos:
            report.add_error("structural", f"Duplicate strategy/scenario/seed combination: {combo}")
    
    # 4-5. Validate each row
    for row in rows:
        validate_numeric_values(row, report)
        validate_categorical_values(row, report)
    
    validate_traceability(rows, report)
    
    # 6. Check configuration consistency
    validate_configuration_consistency(rows, report)
    
    # 7. Validate against manifest if provided
    if manifest_file and manifest_file.exists():
        validate_against_manifest(rows, manifest_file, report)
    
    return report


def write_report_json(report: QCReport, output_path: Path):
    """Write machine-readable JSON report."""
    data = {
        "aggregated_file": report.aggregated_file,
        "manifest_file": report.manifest_file,
        "total_rows": report.total_rows,
        "passed": report.passed,
        "summary": {
            "missing_columns": report.missing_columns,
            "duplicate_run_ids": report.duplicate_run_ids,
            "duplicate_combinations": report.duplicate_combinations,
            "unexpected_strategies": sorted(report.unexpected_strategies),
            "unexpected_scenarios": sorted(report.unexpected_scenarios),
            "unexpected_seeds": sorted(report.unexpected_seeds),
            "nan_values": report.nan_values,
            "inf_values": report.inf_values,
            "config_inconsistencies": report.config_inconsistencies,
            "untraceable_runs": report.untraceable_runs,
            "missing_from_manifest": report.missing_from_manifest,
            "extra_in_aggregated": report.extra_in_aggregated
        },
        "issues": [
            {
                "severity": issue.severity,
                "category": issue.category,
                "run_id": issue.run_id,
                "column": issue.column,
                "message": issue.message,
                "value": issue.value
            }
            for issue in report.issues
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def write_report_text(report: QCReport, output_path: Path):
    """Write human-readable text report."""
    lines = [
        "# Statistical Quality Control Report",
        "",
        f"**Aggregated File:** `{report.aggregated_file}`",
        f"**Manifest File:** `{report.manifest_file or 'N/A'}`",
        f"**Total Rows:** {report.total_rows}",
        "",
        f"**QC Status:** {'✅ PASS' if report.passed else '❌ FAIL'}",
        "",
        "## Summary",
        "",
        f"- **Missing Columns:** {len(report.missing_columns)}",
        f"- **Duplicate run_ids:** {len(report.duplicate_run_ids)}",
        f"- **Duplicate Combinations:** {len(report.duplicate_combinations)}",
        f"- **Unexpected Strategies:** {len(report.unexpected_strategies)}",
        f"- **Unexpected Scenarios:** {len(report.unexpected_scenarios)}",
        f"- **NaN Values:** {report.nan_values}",
        f"- **Infinity Values:** {report.inf_values}",
        f"- **Config Inconsistencies:** {report.config_inconsistencies}",
        f"- **Untraceable Runs:** {report.untraceable_runs}",
        f"- **Missing from Manifest:** {len(report.missing_from_manifest)}",
        f"- **Extra in Aggregated:** {len(report.extra_in_aggregated)}",
        f"- **Total Issues:** {len(report.issues)}",
        ""
    ]
    
    if report.issues:
        lines.extend([
            "## Issues",
            ""
        ])
        
        # Group by severity
        errors = [i for i in report.issues if i.severity == "ERROR"]
        warnings = [i for i in report.issues if i.severity == "WARNING"]
        
        if errors:
            lines.append(f"### Errors ({len(errors)})")
            lines.append("")
            for issue in errors:
                loc = f"{issue.run_id or 'N/A'}:{issue.column or 'N/A'}"
                lines.append(f"- [{issue.category}] {loc}: {issue.message}")
                if issue.value:
                    lines.append(f"  Value: `{issue.value}`")
            lines.append("")
        
        if warnings:
            lines.append(f"### Warnings ({len(warnings)})")
            lines.append("")
            for issue in warnings:
                loc = f"{issue.run_id or 'N/A'}:{issue.column or 'N/A'}"
                lines.append(f"- [{issue.category}] {loc}: {issue.message}")
                if issue.value:
                    lines.append(f"  Value: `{issue.value}`")
            lines.append("")
    else:
        lines.extend([
            "## Issues",
            "",
            "No issues detected. ✅",
            ""
        ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Statistical QC for aggregated experiment results"
    )
    parser.add_argument(
        "--aggregated",
        type=Path,
        required=True,
        help="Path to aggregated results CSV"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to experiment manifest CSV (optional)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: same as aggregated with .qc.json extension)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.aggregated.exists():
        print(f"ERROR: Aggregated file not found: {args.aggregated}", file=sys.stderr)
        return 2
    
    if args.manifest and not args.manifest.exists():
        print(f"ERROR: Manifest file not found: {args.manifest}", file=sys.stderr)
        return 2
    
    # Set default output
    if args.output is None:
        args.output = args.aggregated.with_suffix('.qc.json')
    
    text_output = args.output.with_suffix('.qc.md')
    
    # Run QC
    print(f"Running statistical QC on: {args.aggregated}")
    if args.manifest:
        print(f"Using manifest: {args.manifest}")
    
    report = perform_qc(args.aggregated, args.manifest)
    
    # Write reports
    write_report_json(report, args.output)
    write_report_text(report, text_output)
    
    print(f"\n{'='*60}")
    print("QC COMPLETE")
    print(f"{'='*60}")
    print(f"Total Rows:      {report.total_rows}")
    print(f"Total Issues:    {len(report.issues)}")
    print(f"  Errors:        {sum(1 for i in report.issues if i.severity == 'ERROR')}")
    print(f"  Warnings:      {sum(1 for i in report.issues if i.severity == 'WARNING')}")
    print(f"\nQC Status:       {'✅ PASS' if report.passed else '❌ FAIL'}")
    print(f"\nReports saved:")
    print(f"  - {args.output}")
    print(f"  - {text_output}")
    
    # Exit code
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
