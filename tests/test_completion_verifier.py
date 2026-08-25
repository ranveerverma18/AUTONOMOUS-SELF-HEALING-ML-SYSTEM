"""
Tests for experiment completion verifier.

Uses synthetic test data only - does not run real experiments.
"""

import csv
import json
import pytest
from pathlib import Path
from scripts.matrix_orchestration.verify_completion import (
    validate_csv_file,
    validate_json_file,
    validate_run,
    find_orphan_files,
    cross_check_status,
    find_matching_files,
    parse_run_id_pattern,
    ValidationResult,
    REQUIRED_CSV_COLUMNS
)


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create temporary results directory structure."""
    results_dir = tmp_path / "results"
    raw_dir = results_dir / "raw"
    aggregated_dir = results_dir / "aggregated"
    
    raw_dir.mkdir(parents=True)
    aggregated_dir.mkdir(parents=True)
    
    return results_dir


@pytest.fixture
def sample_csv_data():
    """Sample CSV row data."""
    return {
        "run_id": "static_test_seed42_20260101_120000",
        "seed": "42",
        "strategy": "static",
        "scenario": "test",
        "sample_index": "0",
        "engine_id": "1",
        "cycle": "1",
        "event_index": "0",
        "scenario_active": "True",
        "degradation_started": "False",
        "actual_rul": "100",
        "predicted_rul": "95",
        "absolute_error": "5",
        "squared_error": "25",
        "rolling_mae": "5.0",
        "rolling_rmse": "5.0",
        "anomaly_detected": "False",
        "feature_drift_detected": "False",
        "concept_drift_detected": "False",
        "drift_score": "0.1",
        "drift_trigger": "False",
        "retraining_triggered": "False",
        "retraining_started": "False",
        "retraining_completed": "False",
        "validation_skipped": "False",
        "validation_skip_reason": "",
        "candidate_generated": "False",
        "candidate_id": "",
        "candidate_mae": "",
        "candidate_rmse": "",
        "production_mae": "10.0",
        "production_rmse": "12.0",
        "improvement": "",
        "gate_passed": "False",
        "gate_rejected": "False",
        "shadow_started": "False",
        "shadow_completed": "False",
        "shadow_passed": "False",
        "shadow_rejected": "False",
        "shadow_result": "",
        "model_promoted": "False",
        "promotion_decision": "",
        "degraded_promotion": "False",
        "model_version": "1",
        "training_time": "0.0",
        "shadow_evaluation_time": "0.0",
        "inference_latency": "0.001",
        "val_buffer_rows": "0",
        "val_train_rows": "0",
        "val_validation_rows": "0",
        "val_buffer_units": "0",
        "val_train_units": "0",
        "val_validation_units": "0",
        "val_unit_disjoint": "True",
        "val_unit_intersection_count": "0",
        "val_training_time": "0.0"
    }


@pytest.fixture
def sample_json_data():
    """Sample summary JSON data."""
    return {
        "run_id": "static_test_seed42_20260101_120000",
        "config": {
            "seed": 42,
            "strategy": "static",
            "scenario": "test",
            "stream_length": 100
        },
        "summary": {
            "mae": 10.5,
            "rmse": 12.3,
            "detection_delay": 50
        }
    }


def create_valid_csv(path: Path, csv_data: dict):
    """Create a valid CSV file."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(csv_data)


def create_valid_json(path: Path, json_data: dict):
    """Create a valid JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)


class TestParseRunId:
    """Test run_id parsing."""
    
    def test_valid_run_id_simple(self):
        strategy, scenario, seed = parse_run_id_pattern("static_gradual_drift_seed42")
        assert strategy == "static"
        assert scenario == "gradual_drift"
        assert seed == 42
    
    def test_valid_run_id_scheduled(self):
        strategy, scenario, seed = parse_run_id_pattern("scheduled_sudden_spike_seed123")
        assert strategy == "scheduled"
        assert scenario == "sudden_spike"
        assert seed == 123
    
    def test_valid_run_id_naive_adaptive(self):
        """Regression test for naive_adaptive strategy (contains underscore)."""
        strategy, scenario, seed = parse_run_id_pattern("naive_adaptive_gradual_drift_seed42")
        assert strategy == "naive_adaptive"
        assert scenario == "gradual_drift"
        assert seed == 42
    
    def test_valid_run_id_naive_adaptive_concept_drift(self):
        """Test naive_adaptive with multi-word scenario."""
        strategy, scenario, seed = parse_run_id_pattern("naive_adaptive_concept_drift_seed456")
        assert strategy == "naive_adaptive"
        assert scenario == "concept_drift"
        assert seed == 456
    
    def test_valid_run_id_proposed(self):
        strategy, scenario, seed = parse_run_id_pattern("proposed_correlated_drift_seed42")
        assert strategy == "proposed"
        assert scenario == "correlated_drift"
        assert seed == 42
    
    def test_all_real_strategies(self):
        """Test all actual strategy names from the research matrix."""
        test_cases = [
            ("static_gradual_drift_seed42", "static", "gradual_drift", 42),
            ("scheduled_sudden_spike_seed123", "scheduled", "sudden_spike", 123),
            ("naive_adaptive_high_noise_seed456", "naive_adaptive", "high_noise", 456),
            ("proposed_sensor_failure_seed42", "proposed", "sensor_failure", 42),
        ]
        for run_id, exp_strategy, exp_scenario, exp_seed in test_cases:
            strategy, scenario, seed = parse_run_id_pattern(run_id)
            assert strategy == exp_strategy, f"Failed for {run_id}"
            assert scenario == exp_scenario, f"Failed for {run_id}"
            assert seed == exp_seed, f"Failed for {run_id}"
    
    def test_all_real_scenarios(self):
        """Test all actual scenario names from the research matrix."""
        scenarios = [
            "gradual_drift", "sudden_spike", "high_noise", "sensor_failure",
            "concept_drift", "correlated_drift", "intermittent_spikes", "drift_recovery"
        ]
        for scenario in scenarios:
            run_id = f"naive_adaptive_{scenario}_seed42"
            strategy, parsed_scenario, seed = parse_run_id_pattern(run_id)
            assert strategy == "naive_adaptive"
            assert parsed_scenario == scenario
            assert seed == 42
    
    def test_invalid_run_id_format(self):
        with pytest.raises(ValueError, match="no seed"):
            parse_run_id_pattern("invalid_format")
    
    def test_invalid_run_id_no_seed(self):
        with pytest.raises(ValueError, match="no seed"):
            parse_run_id_pattern("static_gradual_drift")
    
    def test_invalid_run_id_unknown_strategy(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            parse_run_id_pattern("unknown_strategy_test_seed42")


class TestFindMatchingFiles:
    """Test file matching logic."""
    
    def test_find_single_file(self, temp_results_dir):
        raw_dir = temp_results_dir / "raw"
        file_path = raw_dir / "static_test_seed42_20260101_120000_events.csv"
        file_path.touch()
        
        matches = find_matching_files("static_test_seed42", raw_dir, "_events.csv")
        assert len(matches) == 1
        assert matches[0] == file_path
    
    def test_find_no_files(self, temp_results_dir):
        raw_dir = temp_results_dir / "raw"
        matches = find_matching_files("nonexistent_run_seed42", raw_dir, "_events.csv")
        assert len(matches) == 0
    
    def test_find_duplicate_files(self, temp_results_dir):
        raw_dir = temp_results_dir / "raw"
        file1 = raw_dir / "static_test_seed42_20260101_120000_events.csv"
        file2 = raw_dir / "static_test_seed42_20260101_130000_events.csv"
        file1.touch()
        file2.touch()
        
        matches = find_matching_files("static_test_seed42", raw_dir, "_events.csv")
        assert len(matches) == 2


class TestValidateCsvFile:
    """Test CSV file validation."""
    
    def test_valid_csv(self, temp_results_dir, sample_csv_data):
        csv_path = temp_results_dir / "test.csv"
        create_valid_csv(csv_path, sample_csv_data)
        
        valid, errors = validate_csv_file(csv_path)
        assert valid
        assert len(errors) == 0
    
    def test_missing_csv(self, temp_results_dir):
        csv_path = temp_results_dir / "nonexistent.csv"
        
        valid, errors = validate_csv_file(csv_path)
        assert not valid
        assert len(errors) > 0
        assert "does not exist" in errors[0]
    
    def test_empty_csv(self, temp_results_dir):
        csv_path = temp_results_dir / "empty.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_CSV_COLUMNS)
            writer.writeheader()
        
        valid, errors = validate_csv_file(csv_path)
        assert not valid
        assert any("empty" in e.lower() for e in errors)
    
    def test_missing_columns(self, temp_results_dir):
        csv_path = temp_results_dir / "incomplete.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", "seed"])
            writer.writeheader()
            writer.writerow({"run_id": "test", "seed": "42"})
        
        valid, errors = validate_csv_file(csv_path)
        assert not valid
        assert any("missing required columns" in e.lower() for e in errors)
    
    def test_no_header(self, temp_results_dir):
        csv_path = temp_results_dir / "no_header.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("some,data,without,proper,header\n")
        
        valid, errors = validate_csv_file(csv_path)
        assert not valid


class TestValidateJsonFile:
    """Test JSON file validation."""
    
    def test_valid_json(self, temp_results_dir, sample_json_data):
        json_path = temp_results_dir / "test.json"
        create_valid_json(json_path, sample_json_data)
        
        valid, errors = validate_json_file(json_path, "static", "test", 42)
        assert valid
        assert len(errors) == 0
    
    def test_missing_json(self, temp_results_dir):
        json_path = temp_results_dir / "nonexistent.json"
        
        valid, errors = validate_json_file(json_path, "static", "test", 42)
        assert not valid
        assert any("does not exist" in e for e in errors)
    
    def test_invalid_json_syntax(self, temp_results_dir):
        json_path = temp_results_dir / "invalid.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write("{invalid json}")
        
        valid, errors = validate_json_file(json_path, "static", "test", 42)
        assert not valid
        assert any("Invalid JSON" in e for e in errors)
    
    def test_missing_run_id_field(self, temp_results_dir):
        json_path = temp_results_dir / "no_run_id.json"
        data = {"config": {}, "summary": {}}
        create_valid_json(json_path, data)
        
        valid, errors = validate_json_file(json_path, "static", "test", 42)
        assert not valid
        assert any("missing 'run_id'" in e.lower() for e in errors)
    
    def test_missing_config_field(self, temp_results_dir):
        json_path = temp_results_dir / "no_config.json"
        data = {"run_id": "test", "summary": {}}
        create_valid_json(json_path, data)
        
        valid, errors = validate_json_file(json_path, "static", "test", 42)
        assert not valid
        assert any("missing 'config'" in e.lower() for e in errors)
    
    def test_missing_summary_field(self, temp_results_dir):
        json_path = temp_results_dir / "no_summary.json"
        data = {"run_id": "test", "config": {}}
        create_valid_json(json_path, data)
        
        valid, errors = validate_json_file(json_path, "static", "test", 42)
        assert not valid
        assert any("missing 'summary'" in e.lower() for e in errors)
    
    def test_strategy_mismatch(self, temp_results_dir, sample_json_data):
        json_path = temp_results_dir / "wrong_strategy.json"
        create_valid_json(json_path, sample_json_data)
        
        valid, errors = validate_json_file(json_path, "proposed", "test", 42)
        assert not valid
        assert any("strategy mismatch" in e.lower() for e in errors)
    
    def test_scenario_mismatch(self, temp_results_dir, sample_json_data):
        json_path = temp_results_dir / "wrong_scenario.json"
        create_valid_json(json_path, sample_json_data)
        
        valid, errors = validate_json_file(json_path, "static", "different", 42)
        assert not valid
        assert any("scenario mismatch" in e.lower() for e in errors)
    
    def test_seed_mismatch(self, temp_results_dir, sample_json_data):
        json_path = temp_results_dir / "wrong_seed.json"
        create_valid_json(json_path, sample_json_data)
        
        valid, errors = validate_json_file(json_path, "static", "test", 999)
        assert not valid
        assert any("seed mismatch" in e.lower() for e in errors)


class TestValidateRun:
    """Test complete run validation."""
    
    def test_all_outputs_present_valid(self, temp_results_dir, sample_csv_data, sample_json_data):
        raw_dir = temp_results_dir / "raw"
        agg_dir = temp_results_dir / "aggregated"
        
        csv_path = raw_dir / "static_test_seed42_20260101_120000_events.csv"
        json_path = agg_dir / "static_test_seed42_20260101_120000_summary.json"
        
        create_valid_csv(csv_path, sample_csv_data)
        create_valid_json(json_path, sample_json_data)
        
        result = validate_run("static_test_seed42", "static", "test", 42, raw_dir, agg_dir)
        assert result.valid
        assert len(result.errors) == 0
        assert result.raw_csv_path == csv_path
        assert result.summary_json_path == json_path
    
    def test_missing_raw_csv(self, temp_results_dir, sample_json_data):
        raw_dir = temp_results_dir / "raw"
        agg_dir = temp_results_dir / "aggregated"
        
        json_path = agg_dir / "static_test_seed42_20260101_120000_summary.json"
        create_valid_json(json_path, sample_json_data)
        
        result = validate_run("static_test_seed42", "static", "test", 42, raw_dir, agg_dir)
        assert not result.valid
        assert any("No raw CSV file found" in e for e in result.errors)
    
    def test_missing_summary_json(self, temp_results_dir, sample_csv_data):
        raw_dir = temp_results_dir / "raw"
        agg_dir = temp_results_dir / "aggregated"
        
        csv_path = raw_dir / "static_test_seed42_20260101_120000_events.csv"
        create_valid_csv(csv_path, sample_csv_data)
        
        result = validate_run("static_test_seed42", "static", "test", 42, raw_dir, agg_dir)
        assert not result.valid
        assert any("No summary JSON file found" in e for e in result.errors)
    
    def test_duplicate_raw_csv(self, temp_results_dir, sample_csv_data, sample_json_data):
        raw_dir = temp_results_dir / "raw"
        agg_dir = temp_results_dir / "aggregated"
        
        csv1 = raw_dir / "static_test_seed42_20260101_120000_events.csv"
        csv2 = raw_dir / "static_test_seed42_20260101_130000_events.csv"
        json_path = agg_dir / "static_test_seed42_20260101_120000_summary.json"
        
        create_valid_csv(csv1, sample_csv_data)
        create_valid_csv(csv2, sample_csv_data)
        create_valid_json(json_path, sample_json_data)
        
        result = validate_run("static_test_seed42", "static", "test", 42, raw_dir, agg_dir)
        assert not result.valid
        assert any("Duplicate raw CSV" in e for e in result.errors)
    
    def test_duplicate_summary_json(self, temp_results_dir, sample_csv_data, sample_json_data):
        raw_dir = temp_results_dir / "raw"
        agg_dir = temp_results_dir / "aggregated"
        
        csv_path = raw_dir / "static_test_seed42_20260101_120000_events.csv"
        json1 = agg_dir / "static_test_seed42_20260101_120000_summary.json"
        json2 = agg_dir / "static_test_seed42_20260101_130000_summary.json"
        
        create_valid_csv(csv_path, sample_csv_data)
        create_valid_json(json1, sample_json_data)
        create_valid_json(json2, sample_json_data)
        
        result = validate_run("static_test_seed42", "static", "test", 42, raw_dir, agg_dir)
        assert not result.valid
        assert any("Duplicate summary JSON" in e for e in result.errors)


class TestOrphanDetection:
    """Test orphan file detection."""
    
    def test_no_orphans(self, temp_results_dir, sample_csv_data, sample_json_data):
        raw_dir = temp_results_dir / "raw"
        agg_dir = temp_results_dir / "aggregated"
        
        csv_path = raw_dir / "static_test_seed42_20260101_120000_events.csv"
        json_path = agg_dir / "static_test_seed42_20260101_120000_summary.json"
        
        create_valid_csv(csv_path, sample_csv_data)
        create_valid_json(json_path, sample_json_data)
        
        valid_run_ids = {"static_test_seed42"}
        orphans = find_orphan_files(raw_dir, agg_dir, valid_run_ids)
        assert len(orphans) == 0
    
    def test_orphan_raw_file(self, temp_results_dir, sample_csv_data):
        raw_dir = temp_results_dir / "raw"
        agg_dir = temp_results_dir / "aggregated"
        
        csv_path = raw_dir / "orphan_run_seed99_20260101_120000_events.csv"
        create_valid_csv(csv_path, sample_csv_data)
        
        valid_run_ids = {"static_test_seed42"}
        orphans = find_orphan_files(raw_dir, agg_dir, valid_run_ids)
        assert len(orphans) > 0
        assert any("orphan_run_seed99" in o for o in orphans)
    
    def test_orphan_summary_file(self, temp_results_dir, sample_json_data):
        raw_dir = temp_results_dir / "raw"
        agg_dir = temp_results_dir / "aggregated"
        
        json_path = agg_dir / "orphan_run_seed99_20260101_120000_summary.json"
        create_valid_json(json_path, sample_json_data)
        
        valid_run_ids = {"static_test_seed42"}
        orphans = find_orphan_files(raw_dir, agg_dir, valid_run_ids)
        assert len(orphans) > 0
        assert any("orphan_run_seed99" in o for o in orphans)


class TestStatusCrossCheck:
    """Test manifest/status/filesystem cross-checking."""
    
    def test_success_with_missing_output(self):
        """Execution status=SUCCESS but outputs missing is a mismatch."""
        manifest = [{"run_id": "test_run_seed42", "status": "PLANNED"}]
        status_data = {"test_run_seed42": {"status": "SUCCESS"}}
        
        result = ValidationResult(run_id="test_run_seed42", valid=False)
        result.errors.append("No raw CSV file found")
        validation_results = {"test_run_seed42": result}
        
        mismatches = cross_check_status(manifest, status_data, validation_results)
        assert len(mismatches) > 0
        assert any("Execution status=SUCCESS but outputs missing" in m for m in mismatches)
    
    def test_failed_with_valid_output(self):
        """Execution status=FAILED but valid outputs exist is a mismatch."""
        manifest = [{"run_id": "test_run_seed42", "status": "PLANNED"}]
        status_data = {"test_run_seed42": {"status": "FAILED"}}
        
        result = ValidationResult(run_id="test_run_seed42", valid=True)
        validation_results = {"test_run_seed42": result}
        
        mismatches = cross_check_status(manifest, status_data, validation_results)
        assert len(mismatches) > 0
        assert any("Execution status=FAILED but valid outputs exist" in m for m in mismatches)
    
    def test_no_status_entry_with_output(self):
        """Valid outputs but no execution status entry is a mismatch."""
        manifest = [{"run_id": "test_run_seed42", "status": "PLANNED"}]
        status_data = {}
        
        result = ValidationResult(run_id="test_run_seed42", valid=True)
        validation_results = {"test_run_seed42": result}
        
        mismatches = cross_check_status(manifest, status_data, validation_results)
        assert len(mismatches) > 0
        assert any("no execution status entry" in m for m in mismatches)
    
    def test_planned_with_existing_output_not_mismatch(self):
        """Manifest status=PLANNED with existing outputs is NOT a mismatch.
        
        The manifest represents the immutable plan created before execution.
        Actual execution state is tracked in matrix_execution_status.csv.
        """
        manifest = [{"run_id": "test_run_seed42", "status": "PLANNED"}]
        status_data = {"test_run_seed42": {"status": "SUCCESS"}}
        
        result = ValidationResult(run_id="test_run_seed42", valid=True)
        validation_results = {"test_run_seed42": result}
        
        mismatches = cross_check_status(manifest, status_data, validation_results)
        # Should NOT be a mismatch - manifest stays PLANNED, execution status shows SUCCESS
        assert len(mismatches) == 0


class TestIntegration:
    """Integration tests with complete scenarios."""
    
    def test_valid_mini_matrix(self, temp_results_dir, sample_csv_data, sample_json_data):
        """Simulate a valid 4-run mini-matrix."""
        raw_dir = temp_results_dir / "raw"
        agg_dir = temp_results_dir / "aggregated"
        
        runs = [
            ("static", "gradual_drift", 42),
            ("static", "gradual_drift", 123),
            ("proposed", "gradual_drift", 42),
            ("proposed", "gradual_drift", 123)
        ]
        
        validation_results = {}
        for strategy, scenario, seed in runs:
            run_id = f"{strategy}_{scenario}_seed{seed}"
            csv_path = raw_dir / f"{run_id}_20260101_120000_events.csv"
            json_path = agg_dir / f"{run_id}_20260101_120000_summary.json"
            
            # Update sample data
            csv_data = sample_csv_data.copy()
            csv_data["strategy"] = strategy
            csv_data["scenario"] = scenario
            csv_data["seed"] = str(seed)
            
            json_data = sample_json_data.copy()
            json_data["config"]["strategy"] = strategy
            json_data["config"]["scenario"] = scenario
            json_data["config"]["seed"] = seed
            
            create_valid_csv(csv_path, csv_data)
            create_valid_json(json_path, json_data)
            
            result = validate_run(run_id, strategy, scenario, seed, raw_dir, agg_dir)
            validation_results[run_id] = result
        
        # All should be valid
        assert all(r.valid for r in validation_results.values())
        assert len(validation_results) == 4
        
        # No orphans
        valid_run_ids = set(validation_results.keys())
        orphans = find_orphan_files(raw_dir, agg_dir, valid_run_ids)
        assert len(orphans) == 0
