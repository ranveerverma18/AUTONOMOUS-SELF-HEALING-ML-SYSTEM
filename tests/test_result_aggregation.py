"""
Tests for result aggregation tool.

Uses synthetic test data only - does not run real experiments.
"""

import csv
import json
import pytest
from pathlib import Path
from scripts.analysis.aggregate_results import (
    aggregate_run,
    aggregate_results,
    write_aggregated_csv,
    AggregatedRun
)


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create temporary results directory structure."""
    results_dir = tmp_path / "results"
    raw_dir = results_dir / "raw"
    aggregated_dir = results_dir / "aggregated"
    logs_dir = results_dir / "logs"
    
    raw_dir.mkdir(parents=True)
    aggregated_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    
    return results_dir


@pytest.fixture
def sample_summary_json():
    """Sample summary JSON matching actual schema."""
    return {
        "run_id": "static_test_seed42_20260101_120000",
        "config": {
            "seed": 42,
            "strategy": "static",
            "scenario": "test",
            "stream_length": 2400,
            "stream_mode": "interleaved",
            "scenario_onset_cycle_min": 25,
            "scenario_onset_cycle_max": 35,
            "train_fraction": 0.76,
            "validation_fraction": 0.25,
            "retraining_interval": 100
        },
        "summary": {
            "mae": 33.5,
            "rmse": 41.2,
            "detection_delay": 518,
            "drift_detections": 15,
            "anomaly_detections": 341,
            "false_positive_triggers": 0,
            "retraining_events": 0,
            "validation_skipped_events": 0,
            "candidates_generated": 0,
            "gate_accepts": 0,
            "gate_rejects": 0,
            "shadow_promotions": 0,
            "shadow_rejections": 0,
            "model_promoted_events": 0,
            "degraded_promotions": 0,
            "degraded_promotion_rate": None,
            "time_to_first_error_recovery": None,
            "time_to_sustained_recovery": None,
            "total_retraining_time": 0.0,
            "total_shadow_evaluation_time": 0.0,
            "total_adaptation_time": 0.0,
            "mean_inference_latency": 0.008
        },
        "raw_events": "test/path/events.csv",
        "summary_json": "test/path/summary.json"
    }


@pytest.fixture
def sample_csv_header():
    """CSV header matching actual schema."""
    return [
        "run_id", "seed", "strategy", "scenario", "sample_index", "engine_id", "cycle",
        "event_index", "scenario_active", "degradation_started", "actual_rul",
        "predicted_rul", "absolute_error", "squared_error", "rolling_mae", "rolling_rmse",
        "anomaly_detected", "feature_drift_detected", "concept_drift_detected",
        "drift_score", "drift_trigger", "retraining_triggered", "retraining_started",
        "retraining_completed", "validation_skipped", "validation_skip_reason",
        "candidate_generated", "candidate_id", "candidate_mae", "candidate_rmse",
        "production_mae", "production_rmse", "improvement", "gate_passed", "gate_rejected",
        "shadow_started", "shadow_completed", "shadow_passed", "shadow_rejected",
        "shadow_result", "model_promoted", "promotion_decision", "degraded_promotion",
        "model_version", "training_time", "shadow_evaluation_time", "inference_latency",
        "val_buffer_rows", "val_train_rows", "val_validation_rows", "val_buffer_units",
        "val_train_units", "val_validation_units", "val_unit_disjoint",
        "val_unit_intersection_count", "val_training_time"
    ]


def create_valid_manifest(path: Path, runs: list):
    """Create a valid manifest CSV."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'run_id', 'strategy', 'scenario', 'seed', 'stream_length', 'stream_mode',
            'scenario_onset_cycle_min', 'scenario_onset_cycle_max', 'train_fraction',
            'validation_fraction', 'retraining_interval', 'data_drift_window',
            'data_drift_p_threshold', 'data_drift_feature_ratio_threshold',
            'data_drift_min_effect_size', 'error_window', 'error_threshold',
            'retrain_error_threshold', 'retrain_drift_score_threshold',
            'performance_gate_threshold', 'shadow_window', 'cooldown',
            'minimum_retraining_samples', 'minimum_validation_rows',
            'minimum_validation_units', 'git_commit', 'dataset_checksum',
            'python_version', 'planned_timestamp', 'status', 'raw_csv_path',
            'summary_json_path'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for run in runs:
            row = {
                'run_id': run['run_id'],
                'strategy': run['strategy'],
                'scenario': run['scenario'],
                'seed': run['seed'],
                'stream_length': 2400,
                'stream_mode': 'interleaved',
                'scenario_onset_cycle_min': 25,
                'scenario_onset_cycle_max': 35,
                'train_fraction': 0.76,
                'validation_fraction': 0.25,
                'retraining_interval': 100,
                'data_drift_window': 20,
                'data_drift_p_threshold': 0.05,
                'data_drift_feature_ratio_threshold': 0.12,
                'data_drift_min_effect_size': 0.08,
                'error_window': 8,
                'error_threshold': 75.0,
                'retrain_error_threshold': 45.0,
                'retrain_drift_score_threshold': 0.55,
                'performance_gate_threshold': 0.95,
                'shadow_window': 20,
                'cooldown': 30,
                'minimum_retraining_samples': 55,
                'minimum_validation_rows': 20,
                'minimum_validation_units': 1,
                'git_commit': 'abc123',
                'dataset_checksum': 'def456',
                'python_version': '3.12.0',
                'planned_timestamp': '2026-01-01 12:00:00',
                'status': 'SUCCESS',
                'raw_csv_path': f"results/raw/{run['run_id']}_*.csv",
                'summary_json_path': f"results/aggregated/{run['run_id']}_*.json"
            }
            writer.writerow(row)


def create_valid_csv(path: Path, run_id: str, strategy: str, scenario: str, seed: int, header: list):
    """Create a valid CSV file with minimal data."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # Write one data row
        row = [run_id, seed, strategy, scenario] + [''] * (len(header) - 4)
        row[10] = '100.0'  # actual_rul
        row[11] = '95.0'   # predicted_rul
        row[12] = '5.0'    # absolute_error
        row[13] = '25.0'   # squared_error
        writer.writerow(row)


def create_valid_summary_json(path: Path, run_id: str, strategy: str, scenario: str, seed: int):
    """Create a valid summary JSON file."""
    data = {
        "run_id": run_id,
        "config": {
            "seed": seed,
            "strategy": strategy,
            "scenario": scenario,
            "stream_length": 2400,
            "stream_mode": "interleaved",
            "scenario_onset_cycle_min": 25,
            "scenario_onset_cycle_max": 35,
            "train_fraction": 0.76,
            "validation_fraction": 0.25,
            "retraining_interval": 100
        },
        "summary": {
            "mae": 10.5,
            "rmse": 12.3,
            "detection_delay": 50,
            "drift_detections": 5,
            "anomaly_detections": 10,
            "false_positive_triggers": 0,
            "retraining_events": 2,
            "validation_skipped_events": 0,
            "candidates_generated": 2,
            "gate_accepts": 1,
            "gate_rejects": 1,
            "shadow_promotions": 1,
            "shadow_rejections": 0,
            "model_promoted_events": 1,
            "degraded_promotions": 0,
            "degraded_promotion_rate": 0.0,
            "time_to_first_error_recovery": 100,
            "time_to_sustained_recovery": 150,
            "total_retraining_time": 5.0,
            "total_shadow_evaluation_time": 10.0,
            "total_adaptation_time": 15.0,
            "mean_inference_latency": 0.001
        },
        "raw_events": f"raw/{run_id}_events.csv",
        "summary_json": f"aggregated/{run_id}_summary.json"
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def create_status_file(path: Path, runs: list):
    """Create a valid status file."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['run_id', 'status', 'started_at', 'completed_at', 
                     'runtime_seconds', 'exit_code', 'raw_csv_path', 'summary_json_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow({
                'run_id': run['run_id'],
                'status': 'SUCCESS',
                'started_at': '2026-01-01T12:00:00',
                'completed_at': '2026-01-01T12:01:00',
                'runtime_seconds': '60.0',
                'exit_code': '0',
                'raw_csv_path': f"results/raw/{run['run_id']}_*.csv",
                'summary_json_path': f"results/aggregated/{run['run_id']}_*.json"
            })


class TestAggregateRun:
    """Test individual run aggregation."""
    
    def test_aggregate_static_run(self, tmp_path, sample_summary_json):
        """Test aggregating a static strategy run."""
        json_path = tmp_path / "summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample_summary_json, f)
        
        result = aggregate_run("static_test_seed42", json_path)
        
        assert result.run_id == "static_test_seed42"
        assert result.strategy == "static"
        assert result.scenario == "test"
        assert result.seed == 42
        assert result.mae == 33.5
        assert result.rmse == 41.2
        assert result.detection_delay == 518
        assert result.retraining_events == 0
        assert result.candidates_generated == 0
    
    def test_aggregate_proposed_run(self, tmp_path):
        """Test aggregating a proposed strategy run."""
        data = {
            "run_id": "proposed_test_seed123_20260101_120000",
            "config": {
                "seed": 123,
                "strategy": "proposed",
                "scenario": "test",
                "stream_length": 2400,
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": 25,
                "scenario_onset_cycle_max": 35,
                "train_fraction": 0.76,
                "validation_fraction": 0.25,
                "retraining_interval": 100
            },
            "summary": {
                "mae": 10.3,
                "rmse": 18.0,
                "detection_delay": 500,
                "drift_detections": 12,
                "anomaly_detections": 300,
                "false_positive_triggers": 0,
                "retraining_events": 12,
                "validation_skipped_events": 0,
                "candidates_generated": 12,
                "gate_accepts": 2,
                "gate_rejects": 10,
                "shadow_promotions": 2,
                "shadow_rejections": 0,
                "model_promoted_events": 2,
                "degraded_promotions": 1,
                "degraded_promotion_rate": 0.5,
                "time_to_first_error_recovery": None,
                "time_to_sustained_recovery": None,
                "total_retraining_time": 2.5,
                "total_shadow_evaluation_time": 42.0,
                "total_adaptation_time": 44.5,
                "mean_inference_latency": 0.008
            },
            "raw_events": "test/path/events.csv",
            "summary_json": "test/path/summary.json"
        }
        
        json_path = tmp_path / "summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        result = aggregate_run("proposed_test_seed123", json_path)
        
        assert result.strategy == "proposed"
        assert result.seed == 123
        assert result.retraining_events == 12
        assert result.candidates_generated == 12
        assert result.shadow_promotions == 2
        assert result.degraded_promotions == 1
        assert result.degraded_promotion_rate == 0.5
    
    def test_aggregate_with_null_metrics(self, tmp_path):
        """Test aggregating run with null/None metrics."""
        data = {
            "run_id": "static_test_seed42_20260101_120000",
            "config": {
                "seed": 42,
                "strategy": "static",
                "scenario": "test",
                "stream_length": 2400,
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": 25,
                "scenario_onset_cycle_max": 35,
                "train_fraction": 0.76,
                "validation_fraction": 0.25,
                "retraining_interval": 100
            },
            "summary": {
                "mae": None,
                "rmse": None,
                "detection_delay": None,
                "drift_detections": 0,
                "anomaly_detections": 0,
                "false_positive_triggers": 0,
                "retraining_events": 0,
                "validation_skipped_events": 0,
                "candidates_generated": 0,
                "gate_accepts": 0,
                "gate_rejects": 0,
                "shadow_promotions": 0,
                "shadow_rejections": 0,
                "model_promoted_events": 0,
                "degraded_promotions": 0,
                "degraded_promotion_rate": None,
                "time_to_first_error_recovery": None,
                "time_to_sustained_recovery": None,
                "total_retraining_time": 0.0,
                "total_shadow_evaluation_time": 0.0,
                "total_adaptation_time": 0.0,
                "mean_inference_latency": None
            },
            "raw_events": "test/path/events.csv",
            "summary_json": "test/path/summary.json"
        }
        
        json_path = tmp_path / "summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        result = aggregate_run("static_test_seed42", json_path)
        
        assert result.mae is None
        assert result.rmse is None
        assert result.detection_delay is None
        assert result.mean_inference_latency is None


class TestAggregateResults:
    """Test complete aggregation workflow."""
    
    def test_successful_aggregation_4_runs(self, temp_results_dir, sample_csv_header):
        """Test successful aggregation of 4-run mini-matrix."""
        runs = [
            {'run_id': 'static_test_seed42', 'strategy': 'static', 'scenario': 'test', 'seed': 42},
            {'run_id': 'static_test_seed123', 'strategy': 'static', 'scenario': 'test', 'seed': 123},
            {'run_id': 'proposed_test_seed42', 'strategy': 'proposed', 'scenario': 'test', 'seed': 42},
            {'run_id': 'proposed_test_seed123', 'strategy': 'proposed', 'scenario': 'test', 'seed': 123}
        ]
        
        # Create manifest
        manifest_path = temp_results_dir / "manifest.csv"
        create_valid_manifest(manifest_path, runs)
        
        # Create outputs for each run
        for run in runs:
            csv_path = temp_results_dir / "raw" / f"{run['run_id']}_20260101_120000_events.csv"
            json_path = temp_results_dir / "aggregated" / f"{run['run_id']}_20260101_120000_summary.json"
            
            create_valid_csv(csv_path, run['run_id'], run['strategy'], run['scenario'], run['seed'], sample_csv_header)
            create_valid_summary_json(json_path, run['run_id'], run['strategy'], run['scenario'], run['seed'])
        
        # Create status file
        create_status_file(temp_results_dir / "matrix_execution_status.csv", runs)
        
        # Run aggregation
        aggregated_runs, report = aggregate_results(manifest_path, temp_results_dir)
        
        assert len(aggregated_runs) == 4
        assert report.successfully_aggregated == 4
        assert report.verification_status == "PASS"
        
        # Check sorting (by strategy, scenario, seed)
        assert aggregated_runs[0].strategy == "proposed"
        assert aggregated_runs[0].seed == 42
        assert aggregated_runs[1].strategy == "proposed"
        assert aggregated_runs[1].seed == 123
        assert aggregated_runs[2].strategy == "static"
        assert aggregated_runs[2].seed == 42
        assert aggregated_runs[3].strategy == "static"
        assert aggregated_runs[3].seed == 123
    
    def test_refuse_missing_output(self, temp_results_dir, sample_csv_header):
        """Test that aggregation refuses when output is missing."""
        runs = [
            {'run_id': 'static_test_seed42', 'strategy': 'static', 'scenario': 'test', 'seed': 42}
        ]
        
        manifest_path = temp_results_dir / "manifest.csv"
        create_valid_manifest(manifest_path, runs)
        
        # Don't create outputs - verification should fail
        
        with pytest.raises(ValueError, match="Verification failed"):
            aggregate_results(manifest_path, temp_results_dir)
    
    def test_refuse_duplicate_output(self, temp_results_dir, sample_csv_header):
        """Test that aggregation refuses when duplicate outputs exist."""
        runs = [
            {'run_id': 'static_test_seed42', 'strategy': 'static', 'scenario': 'test', 'seed': 42}
        ]
        
        manifest_path = temp_results_dir / "manifest.csv"
        create_valid_manifest(manifest_path, runs)
        
        # Create duplicate outputs
        for i in range(2):
            csv_path = temp_results_dir / "raw" / f"static_test_seed42_2026010{i}_120000_events.csv"
            json_path = temp_results_dir / "aggregated" / f"static_test_seed42_2026010{i}_120000_summary.json"
            
            create_valid_csv(csv_path, "static_test_seed42", "static", "test", 42, sample_csv_header)
            create_valid_summary_json(json_path, "static_test_seed42", "static", "test", 42)
        
        create_status_file(temp_results_dir / "matrix_execution_status.csv", runs)
        
        with pytest.raises(ValueError, match="Verification failed"):
            aggregate_results(manifest_path, temp_results_dir)
    
    def test_refuse_orphan_files(self, temp_results_dir, sample_csv_header):
        """Test that aggregation refuses when orphan files exist."""
        runs = [
            {'run_id': 'static_test_seed42', 'strategy': 'static', 'scenario': 'test', 'seed': 42}
        ]
        
        manifest_path = temp_results_dir / "manifest.csv"
        create_valid_manifest(manifest_path, runs)
        
        # Create expected outputs
        csv_path = temp_results_dir / "raw" / "static_test_seed42_20260101_120000_events.csv"
        json_path = temp_results_dir / "aggregated" / "static_test_seed42_20260101_120000_summary.json"
        create_valid_csv(csv_path, "static_test_seed42", "static", "test", 42, sample_csv_header)
        create_valid_summary_json(json_path, "static_test_seed42", "static", "test", 42)
        
        # Create orphan file
        orphan_csv = temp_results_dir / "raw" / "orphan_run_seed99_20260101_120000_events.csv"
        create_valid_csv(orphan_csv, "orphan_run_seed99", "static", "test", 99, sample_csv_header)
        
        create_status_file(temp_results_dir / "matrix_execution_status.csv", runs)
        
        with pytest.raises(ValueError, match="Verification failed"):
            aggregate_results(manifest_path, temp_results_dir)
    
    def test_refuse_identity_mismatch(self, temp_results_dir, sample_csv_header):
        """Test that aggregation refuses when identity fields mismatch."""
        runs = [
            {'run_id': 'static_test_seed42', 'strategy': 'static', 'scenario': 'test', 'seed': 42}
        ]
        
        manifest_path = temp_results_dir / "manifest.csv"
        create_valid_manifest(manifest_path, runs)
        
        # Create outputs with wrong strategy in JSON
        csv_path = temp_results_dir / "raw" / "static_test_seed42_20260101_120000_events.csv"
        json_path = temp_results_dir / "aggregated" / "static_test_seed42_20260101_120000_summary.json"
        
        create_valid_csv(csv_path, "static_test_seed42", "static", "test", 42, sample_csv_header)
        # Create JSON with WRONG strategy
        create_valid_summary_json(json_path, "static_test_seed42", "proposed", "test", 42)
        
        create_status_file(temp_results_dir / "matrix_execution_status.csv", runs)
        
        with pytest.raises(ValueError, match="Verification failed"):
            aggregate_results(manifest_path, temp_results_dir)


class TestWriteAggregatedCSV:
    """Test CSV output writing."""
    
    def test_write_csv_deterministic(self, tmp_path):
        """Test that CSV output is deterministic."""
        runs = [
            AggregatedRun(
                run_id="proposed_test_seed42",
                strategy="proposed",
                scenario="test",
                seed=42,
                mae=10.5,
                rmse=12.3,
                detection_delay=50,
                drift_detections=5,
                anomaly_detections=10,
                false_positive_triggers=0,
                retraining_events=2,
                validation_skipped_events=0,
                candidates_generated=2,
                gate_accepts=1,
                gate_rejects=1,
                shadow_promotions=1,
                shadow_rejections=0,
                model_promoted_events=1,
                degraded_promotions=0,
                degraded_promotion_rate=0.0,
                time_to_first_error_recovery=100,
                time_to_sustained_recovery=150,
                total_retraining_time=5.0,
                total_shadow_evaluation_time=10.0,
                total_adaptation_time=15.0,
                mean_inference_latency=0.001,
                stream_length=2400,
                stream_mode="interleaved",
                scenario_onset_cycle_min=25,
                scenario_onset_cycle_max=35,
                train_fraction=0.76,
                validation_fraction=0.25,
                retraining_interval=100,
                raw_csv="raw/proposed_test_seed42_events.csv",
                summary_json="aggregated/proposed_test_seed42_summary.json"
            ),
            AggregatedRun(
                run_id="static_test_seed42",
                strategy="static",
                scenario="test",
                seed=42,
                mae=33.5,
                rmse=41.2,
                detection_delay=518,
                drift_detections=15,
                anomaly_detections=341,
                false_positive_triggers=0,
                retraining_events=0,
                validation_skipped_events=0,
                candidates_generated=0,
                gate_accepts=0,
                gate_rejects=0,
                shadow_promotions=0,
                shadow_rejections=0,
                model_promoted_events=0,
                degraded_promotions=0,
                degraded_promotion_rate=None,
                time_to_first_error_recovery=None,
                time_to_sustained_recovery=None,
                total_retraining_time=0.0,
                total_shadow_evaluation_time=0.0,
                total_adaptation_time=0.0,
                mean_inference_latency=0.008,
                stream_length=2400,
                stream_mode="interleaved",
                scenario_onset_cycle_min=25,
                scenario_onset_cycle_max=35,
                train_fraction=0.76,
                validation_fraction=0.25,
                retraining_interval=100,
                raw_csv="raw/static_test_seed42_events.csv",
                summary_json="aggregated/static_test_seed42_summary.json"
            )
        ]
        
        # Write twice
        output1 = tmp_path / "output1.csv"
        output2 = tmp_path / "output2.csv"
        
        write_aggregated_csv(runs, output1)
        write_aggregated_csv(runs, output2)
        
        # Compare byte-for-byte
        assert output1.read_bytes() == output2.read_bytes()
    
    def test_write_csv_format(self, tmp_path):
        """Test that CSV has correct format."""
        runs = [
            AggregatedRun(
                run_id="static_test_seed42",
                strategy="static",
                scenario="test",
                seed=42,
                mae=33.5,
                rmse=41.2,
                detection_delay=518,
                drift_detections=15,
                anomaly_detections=341,
                false_positive_triggers=0,
                retraining_events=0,
                validation_skipped_events=0,
                candidates_generated=0,
                gate_accepts=0,
                gate_rejects=0,
                shadow_promotions=0,
                shadow_rejections=0,
                model_promoted_events=0,
                degraded_promotions=0,
                degraded_promotion_rate=None,
                time_to_first_error_recovery=None,
                time_to_sustained_recovery=None,
                total_retraining_time=0.0,
                total_shadow_evaluation_time=0.0,
                total_adaptation_time=0.0,
                mean_inference_latency=0.008,
                stream_length=2400,
                stream_mode="interleaved",
                scenario_onset_cycle_min=25,
                scenario_onset_cycle_max=35,
                train_fraction=0.76,
                validation_fraction=0.25,
                retraining_interval=100,
                raw_csv="raw/static_test_seed42_events.csv",
                summary_json="aggregated/static_test_seed42_summary.json"
            )
        ]
        
        output = tmp_path / "output.csv"
        write_aggregated_csv(runs, output)
        
        # Read and check
        with open(output, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        assert rows[0]['run_id'] == 'static_test_seed42'
        assert rows[0]['strategy'] == 'static'
        assert rows[0]['seed'] == '42'
        assert float(rows[0]['mae']) == 33.5


class TestTraceability:
    """Test that traceability is preserved."""
    
    def test_traceability_preserved(self, temp_results_dir, sample_csv_header):
        """Test that all runs can be traced back to source files."""
        runs = [
            {'run_id': 'static_test_seed42', 'strategy': 'static', 'scenario': 'test', 'seed': 42}
        ]
        
        manifest_path = temp_results_dir / "manifest.csv"
        create_valid_manifest(manifest_path, runs)
        
        csv_path = temp_results_dir / "raw" / "static_test_seed42_20260101_120000_events.csv"
        json_path = temp_results_dir / "aggregated" / "static_test_seed42_20260101_120000_summary.json"
        
        create_valid_csv(csv_path, "static_test_seed42", "static", "test", 42, sample_csv_header)
        create_valid_summary_json(json_path, "static_test_seed42", "static", "test", 42)
        create_status_file(temp_results_dir / "matrix_execution_status.csv", runs)
        
        aggregated_runs, report = aggregate_results(manifest_path, temp_results_dir)
        
        assert len(aggregated_runs) == 1
        run = aggregated_runs[0]
        
        # Check identity preserved
        assert run.run_id == "static_test_seed42"
        assert run.strategy == "static"
        assert run.scenario == "test"
        assert run.seed == 42
        
        # Check file provenance preserved
        assert "static_test_seed42" in run.raw_csv
        assert "static_test_seed42" in run.summary_json
        
        # Check config provenance preserved
        assert run.stream_length == 2400
        assert run.stream_mode == "interleaved"
        assert run.train_fraction == 0.76
