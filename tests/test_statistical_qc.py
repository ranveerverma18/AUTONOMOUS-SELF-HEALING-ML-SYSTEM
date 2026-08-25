"""
Tests for statistical QC tool.

Uses synthetic test data only.
"""

import csv
import pytest
from pathlib import Path
from scripts.analysis.statistical_qc import (
    perform_qc,
    check_required_columns,
    check_duplicate_run_ids,
    check_duplicate_combinations,
    is_null_value,
    is_numeric_valid,
    REQUIRED_COLUMNS
)


@pytest.fixture
def temp_files(tmp_path):
    """Create temporary file paths."""
    return {
        'aggregated': tmp_path / "aggregated.csv",
        'manifest': tmp_path / "manifest.csv"
    }


def create_valid_aggregated_csv(path: Path, rows: list):
    """Create a valid aggregated CSV."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def create_valid_manifest(path: Path, run_ids: list):
    """Create a valid manifest CSV."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['run_id', 'strategy', 'scenario', 'seed', 'git_commit',
                     'dataset_checksum', 'python_version', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_id in run_ids:
            parts = run_id.split('_')
            strategy = parts[0]
            scenario = '_'.join(parts[1:-1])
            seed = parts[-1].replace('seed', '')
            writer.writerow({
                'run_id': run_id,
                'strategy': strategy,
                'scenario': scenario,
                'seed': seed,
                'git_commit': 'abc123',
                'dataset_checksum': 'def456',
                'python_version': '3.12.0',
                'status': 'SUCCESS'
            })


def sample_row(run_id="static_gradual_drift_seed42", strategy="static",
               scenario="gradual_drift", seed=42):
    """Create a sample aggregated row with valid values."""
    return {
        "run_id": run_id,
        "strategy": strategy,
        "scenario": scenario,
        "seed": str(seed),
        "mae": "10.5",
        "rmse": "12.3",
        "detection_delay": "100",
        "drift_detections": "5",
        "anomaly_detections": "10",
        "false_positive_triggers": "0",
        "retraining_events": "2",
        "validation_skipped_events": "0",
        "candidates_generated": "2",
        "gate_accepts": "1",
        "gate_rejects": "1",
        "shadow_promotions": "1",
        "shadow_rejections": "0",
        "model_promoted_events": "1",
        "degraded_promotions": "0",
        "degraded_promotion_rate": "0.0",
        "time_to_first_error_recovery": "",  # Null is OK
        "time_to_sustained_recovery": "",  # Null is OK
        "total_retraining_time": "5.0",
        "total_shadow_evaluation_time": "10.0",
        "total_adaptation_time": "15.0",
        "mean_inference_latency": "0.001",
        "stream_length": "2400",
        "stream_mode": "interleaved",
        "scenario_onset_cycle_min": "25",
        "scenario_onset_cycle_max": "35",
        "train_fraction": "0.76",
        "validation_fraction": "0.25",
        "retraining_interval": "100",
        "raw_csv": f"raw/{run_id}_events.csv",
        "summary_json": f"aggregated/{run_id}_summary.json"
    }


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_is_null_value(self):
        assert is_null_value("")
        assert is_null_value("None")
        assert is_null_value("null")
        assert is_null_value("NA")
        assert not is_null_value("0")
        assert not is_null_value("10.5")
    
    def test_is_numeric_valid(self):
        assert is_numeric_valid("10.5")
        assert is_numeric_valid("-5")
        assert is_numeric_valid("0")
        assert is_numeric_valid("")  # Null is valid
        assert not is_numeric_valid("abc")
        assert not is_numeric_valid("10.5.3")
    
    def test_check_required_columns(self):
        headers = REQUIRED_COLUMNS.copy()
        missing = check_required_columns(headers)
        assert len(missing) == 0
        
        headers = ["run_id", "strategy"]
        missing = check_required_columns(headers)
        assert len(missing) > 0
        assert "scenario" in missing
    
    def test_check_duplicate_run_ids(self):
        rows = [
            {"run_id": "run1"},
            {"run_id": "run2"},
            {"run_id": "run1"}
        ]
        dupes = check_duplicate_run_ids(rows)
        assert "run1" in dupes
        assert len(dupes) == 1
    
    def test_check_duplicate_combinations(self):
        rows = [
            {"strategy": "static", "scenario": "test", "seed": "42"},
            {"strategy": "static", "scenario": "test", "seed": "42"}
        ]
        dupes = check_duplicate_combinations(rows)
        assert len(dupes) == 1


class TestStructuralValidation:
    """Test structural validation."""
    
    def test_valid_structure(self, temp_files):
        """Test that a valid CSV passes structural checks."""
        rows = [sample_row()]
        create_valid_aggregated_csv(temp_files['aggregated'], rows)
        
        report = perform_qc(temp_files['aggregated'])
        
        assert report.passed
        assert len(report.missing_columns) == 0
        assert len(report.duplicate_run_ids) == 0
        assert len(report.duplicate_combinations) == 0
    
    def test_missing_required_column(self, temp_files):
        """Test that missing column is detected."""
        rows = [sample_row()]
        # Write CSV without 'mae' column
        columns = [c for c in REQUIRED_COLUMNS if c != 'mae']
        
        with open(temp_files['aggregated'], 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            row = sample_row()
            row.pop('mae', None)
            writer.writerow(row)
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert 'mae' in report.missing_columns
    
    def test_duplicate_run_id(self, temp_files):
        """Test that duplicate run_id is detected."""
        rows = [
            sample_row("static_test_seed42", "static", "test", 42),
            sample_row("static_test_seed42", "static", "test", 42)
        ]
        create_valid_aggregated_csv(temp_files['aggregated'], rows)
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert "static_test_seed42" in report.duplicate_run_ids
    
    def test_duplicate_combination(self, temp_files):
        """Test that duplicate strategy/scenario/seed is detected."""
        rows = [
            sample_row("static_test_seed42_v1", "static", "test", 42),
            sample_row("static_test_seed42_v2", "static", "test", 42)
        ]
        create_valid_aggregated_csv(temp_files['aggregated'], rows)
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert len(report.duplicate_combinations) > 0


class TestNumericValidation:
    """Test numeric validation."""
    
    def test_nan_value_detected(self, temp_files):
        """Test that NaN values are detected."""
        row = sample_row()
        row['mae'] = 'nan'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert report.nan_values > 0
    
    def test_infinity_value_detected(self, temp_files):
        """Test that infinity values are detected."""
        row = sample_row()
        row['rmse'] = 'inf'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert report.inf_values > 0
    
    def test_malformed_numeric_value(self, temp_files):
        """Test that non-numeric values in numeric columns are detected."""
        row = sample_row()
        row['mae'] = 'not_a_number'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        errors = [i for i in report.issues if i.severity == 'ERROR']
        assert any('numeric' in e.category.lower() for e in errors)
    
    def test_negative_count_rejected(self, temp_files):
        """Test that negative counts are rejected."""
        row = sample_row()
        row['drift_detections'] = '-5'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert any('negative' in i.message.lower() for i in report.issues)
    
    def test_non_integer_count_rejected(self, temp_files):
        """Test that non-integer counts are rejected."""
        row = sample_row()
        row['drift_detections'] = '5.5'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert any('integer' in i.message.lower() for i in report.issues)
    
    def test_negative_time_rejected(self, temp_files):
        """Test that negative time values are rejected."""
        row = sample_row()
        row['total_retraining_time'] = '-1.0'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert any('negative' in i.message.lower() for i in report.issues)
    
    def test_fraction_out_of_range(self, temp_files):
        """Test that fractions outside [0,1] are rejected."""
        row = sample_row()
        row['train_fraction'] = '1.5'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert any('range' in i.message.lower() for i in report.issues)
    
    def test_legitimate_null_accepted(self, temp_files):
        """Test that legitimate null values are accepted."""
        row = sample_row()
        row['detection_delay'] = ''  # Null is OK for detection_delay
        row['degraded_promotion_rate'] = ''  # Null is OK
        row['time_to_first_error_recovery'] = ''  # Null is OK
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        # Should pass - these nulls are legitimate
        assert report.passed


class TestCategoricalValidation:
    """Test categorical validation."""
    
    def test_unexpected_strategy(self, temp_files):
        """Test that unexpected strategy is detected."""
        row = sample_row()
        row['strategy'] = 'unknown_strategy'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert 'unknown_strategy' in report.unexpected_strategies
    
    def test_unexpected_scenario(self, temp_files):
        """Test that unexpected scenario is detected."""
        row = sample_row()
        row['scenario'] = 'unknown_scenario'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert 'unknown_scenario' in report.unexpected_scenarios
    
    def test_invalid_stream_mode(self, temp_files):
        """Test that invalid stream_mode is detected."""
        row = sample_row()
        row['stream_mode'] = 'invalid_mode'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert any('stream_mode' in i.message for i in report.issues)


class TestConfigurationConsistency:
    """Test configuration consistency checks."""
    
    def test_inconsistent_configuration(self, temp_files):
        """Test that configuration inconsistencies are detected."""
        rows = [
            sample_row("static_test_seed42", "static", "test", 42),
            sample_row("static_test_seed123", "static", "test", 123)
        ]
        rows[1]['stream_length'] = '3000'  # Different from row[0]
        create_valid_aggregated_csv(temp_files['aggregated'], rows)
        
        report = perform_qc(temp_files['aggregated'])
        
        # Should have warnings about inconsistent config
        assert report.config_inconsistencies > 0
        warnings = [i for i in report.issues if i.severity == 'WARNING']
        assert len(warnings) > 0


class TestManifestValidation:
    """Test validation against manifest."""
    
    def test_missing_run_from_manifest(self, temp_files):
        """Test that missing runs are detected."""
        # Create manifest with 2 runs
        create_valid_manifest(temp_files['manifest'], [
            "static_test_seed42",
            "static_test_seed123"
        ])
        
        # Create aggregated with only 1 run
        rows = [sample_row("static_test_seed42", "static", "test", 42)]
        create_valid_aggregated_csv(temp_files['aggregated'], rows)
        
        report = perform_qc(temp_files['aggregated'], temp_files['manifest'])
        
        assert not report.passed
        assert "static_test_seed123" in report.missing_from_manifest
    
    def test_extra_run_not_in_manifest(self, temp_files):
        """Test that extra runs are detected."""
        # Create manifest with 1 run
        create_valid_manifest(temp_files['manifest'], ["static_test_seed42"])
        
        # Create aggregated with 2 runs
        rows = [
            sample_row("static_test_seed42", "static", "test", 42),
            sample_row("static_test_seed123", "static", "test", 123)
        ]
        create_valid_aggregated_csv(temp_files['aggregated'], rows)
        
        report = perform_qc(temp_files['aggregated'], temp_files['manifest'])
        
        assert not report.passed
        assert "static_test_seed123" in report.extra_in_aggregated


class TestTraceability:
    """Test traceability validation."""
    
    def test_missing_raw_csv_path(self, temp_files):
        """Test that missing raw_csv is detected."""
        row = sample_row()
        row['raw_csv'] = ''
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert report.untraceable_runs > 0
    
    def test_missing_summary_json_path(self, temp_files):
        """Test that missing summary_json is detected."""
        row = sample_row()
        row['summary_json'] = ''
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        assert not report.passed
        assert report.untraceable_runs > 0
    
    def test_run_id_not_in_path(self, temp_files):
        """Test warning when run_id not in file path."""
        row = sample_row("static_test_seed42")
        row['raw_csv'] = 'raw/different_run_events.csv'
        create_valid_aggregated_csv(temp_files['aggregated'], [row])
        
        report = perform_qc(temp_files['aggregated'])
        
        # Should have warning
        warnings = [i for i in report.issues if i.severity == 'WARNING']
        assert any('not found in' in w.message for w in warnings)


class TestCompleteValidation:
    """Test complete validation scenarios."""
    
    def test_fully_valid_mini_matrix(self, temp_files):
        """Test that a valid 4-run mini-matrix passes all checks."""
        rows = [
            sample_row("static_gradual_drift_seed42", "static", "gradual_drift", 42),
            sample_row("static_gradual_drift_seed123", "static", "gradual_drift", 123),
            sample_row("proposed_gradual_drift_seed42", "proposed", "gradual_drift", 42),
            sample_row("proposed_gradual_drift_seed123", "proposed", "gradual_drift", 123)
        ]
        create_valid_aggregated_csv(temp_files['aggregated'], rows)
        
        create_valid_manifest(temp_files['manifest'], [
            "static_gradual_drift_seed42",
            "static_gradual_drift_seed123",
            "proposed_gradual_drift_seed42",
            "proposed_gradual_drift_seed123"
        ])
        
        report = perform_qc(temp_files['aggregated'], temp_files['manifest'])
        
        assert report.passed
        assert report.total_rows == 4
        assert len(report.issues) == 0
