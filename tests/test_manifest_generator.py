"""Tests for experiment manifest generator."""

from pathlib import Path
import csv
import tempfile
import pytest

from scripts.matrix_orchestration.generate_manifest import (
    generate_manifest,
    get_git_commit,
    get_python_version,
    get_dataset_checksum,
)
from experiments.config import ExperimentConfig


class TestManifestGenerator:
    """Test manifest generation functionality."""
    
    def test_default_matrix_produces_96_rows(self, tmp_path):
        """Default 4×8×3 matrix should produce exactly 96 runs."""
        strategies = ["static", "scheduled", "naive_adaptive", "proposed"]
        scenarios = [
            "gradual_drift", "sudden_spike", "high_noise", "sensor_failure",
            "concept_drift", "correlated_drift", "intermittent_spikes", "drift_recovery"
        ]
        seeds = [42, 123, 456]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 96, f"Expected 96 runs, got {len(rows)}"
    
    def test_all_four_strategies_occur(self, tmp_path):
        """All 4 strategies should appear in default matrix."""
        strategies = ["static", "scheduled", "naive_adaptive", "proposed"]
        scenarios = ["gradual_drift"]
        seeds = [42]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        strategies_in_manifest = {row["strategy"] for row in rows}
        assert strategies_in_manifest == {"static", "scheduled", "naive_adaptive", "proposed"}
    
    def test_all_eight_scenarios_occur(self, tmp_path):
        """All 8 scenarios should appear in default matrix."""
        strategies = ["static"]
        scenarios = [
            "gradual_drift", "sudden_spike", "high_noise", "sensor_failure",
            "concept_drift", "correlated_drift", "intermittent_spikes", "drift_recovery"
        ]
        seeds = [42]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        scenarios_in_manifest = {row["scenario"] for row in rows}
        assert len(scenarios_in_manifest) == 8
        assert scenarios_in_manifest == set(scenarios)
    
    def test_all_three_seeds_occur(self, tmp_path):
        """All 3 seeds should appear in default matrix."""
        strategies = ["static"]
        scenarios = ["gradual_drift"]
        seeds = [42, 123, 456]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        seeds_in_manifest = {int(row["seed"]) for row in rows}
        assert seeds_in_manifest == {42, 123, 456}
    
    def test_no_duplicate_run_ids(self, tmp_path):
        """Each run_id should be unique."""
        strategies = ["static", "proposed"]
        scenarios = ["gradual_drift", "sudden_spike"]
        seeds = [42, 123]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        run_ids = [row["run_id"] for row in rows]
        assert len(run_ids) == len(set(run_ids)), "Duplicate run_ids found"
    
    def test_run_id_is_deterministic(self, tmp_path):
        """run_id should be deterministic: {strategy}_{scenario}_seed{seed}."""
        strategies = ["proposed"]
        scenarios = ["gradual_drift"]
        seeds = [42]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        assert rows[0]["run_id"] == "proposed_gradual_drift_seed42"
        
        # Verify no timestamp in run_id
        assert "_20" not in rows[0]["run_id"], "run_id should not contain timestamp"
    
    def test_all_rows_have_planned_status(self, tmp_path):
        """All newly generated rows should have status=PLANNED."""
        strategies = ["static", "proposed"]
        scenarios = ["gradual_drift"]
        seeds = [42, 123]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        for row in rows:
            assert row["status"] == "PLANNED", f"Row {row['run_id']} has status {row['status']}"
    
    def test_locked_config_parameters_copied_correctly(self, tmp_path):
        """Locked ExperimentConfig parameters should match defaults."""
        strategies = ["static"]
        scenarios = ["gradual_drift"]
        seeds = [42]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        # Get authoritative defaults
        default_config = ExperimentConfig()
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        
        # Verify critical locked parameters
        assert int(row["stream_length"]) == default_config.stream_length
        assert row["stream_mode"] == default_config.stream_mode
        assert int(row["scenario_onset_cycle_min"]) == default_config.scenario_onset_cycle_min
        assert int(row["scenario_onset_cycle_max"]) == default_config.scenario_onset_cycle_max
        assert float(row["train_fraction"]) == default_config.train_fraction
        assert float(row["validation_fraction"]) == default_config.validation_fraction
        assert int(row["retraining_interval"]) == default_config.retraining_interval
        assert float(row["retrain_error_threshold"]) == default_config.retrain_error_threshold
        assert float(row["performance_gate_threshold"]) == default_config.performance_gate_threshold
        assert int(row["shadow_window"]) == default_config.shadow_window
        assert int(row["cooldown"]) == default_config.cooldown
        assert int(row["minimum_retraining_samples"]) == default_config.minimum_retraining_samples
        assert int(row["minimum_validation_rows"]) == default_config.minimum_validation_rows
        assert int(row["minimum_validation_units"]) == default_config.minimum_validation_units
    
    def test_git_commit_recorded(self, tmp_path):
        """Git commit should be recorded in manifest."""
        strategies = ["static"]
        scenarios = ["gradual_drift"]
        seeds = [42]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        
        assert "git_commit" in row
        assert row["git_commit"] != ""
        # Either valid commit hash or "unknown"
        assert len(row["git_commit"]) == 40 or row["git_commit"] == "unknown"
    
    def test_dataset_provenance_recorded(self, tmp_path):
        """Dataset checksum should be recorded from PROVENANCE.md."""
        strategies = ["static"]
        scenarios = ["gradual_drift"]
        seeds = [42]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        
        assert "dataset_checksum" in row
        assert row["dataset_checksum"] != ""
        # Should be MD5 hash or "unavailable"
        assert len(row["dataset_checksum"]) == 32 or row["dataset_checksum"] == "unavailable"
    
    def test_custom_mini_matrix(self, tmp_path):
        """Custom mini-matrix arguments should work."""
        strategies = ["static"]
        scenarios = ["gradual_drift", "sudden_spike"]
        seeds = [42]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # 1 strategy × 2 scenarios × 1 seed = 2 runs
        assert len(rows) == 2
        assert {row["scenario"] for row in rows} == {"gradual_drift", "sudden_spike"}
    
    def test_manifest_column_schema_stable(self, tmp_path):
        """Manifest column schema should be stable and complete."""
        strategies = ["static"]
        scenarios = ["gradual_drift"]
        seeds = [42]
        
        manifest_path = generate_manifest(strategies, scenarios, seeds, tmp_path)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
        
        # Required columns
        required_columns = {
            "run_id", "strategy", "scenario", "seed",
            "stream_length", "stream_mode",
            "scenario_onset_cycle_min", "scenario_onset_cycle_max",
            "train_fraction", "validation_fraction",
            "retraining_interval",
            "data_drift_window", "data_drift_p_threshold",
            "data_drift_feature_ratio_threshold", "data_drift_min_effect_size",
            "error_window", "error_threshold",
            "retrain_error_threshold", "retrain_drift_score_threshold",
            "performance_gate_threshold", "shadow_window", "cooldown",
            "minimum_retraining_samples", "minimum_validation_rows", "minimum_validation_units",
            "git_commit", "dataset_checksum", "python_version", "planned_timestamp",
            "status", "raw_csv_path", "summary_json_path",
        }
        
        assert set(columns) == required_columns, f"Column mismatch. Expected: {required_columns}, Got: {set(columns)}"
    
    def test_invalid_strategy_raises_error(self, tmp_path):
        """Invalid strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            generate_manifest(["invalid_strategy"], ["gradual_drift"], [42], tmp_path)
    
    def test_invalid_scenario_raises_error(self, tmp_path):
        """Invalid scenario should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid scenario"):
            generate_manifest(["static"], ["invalid_scenario"], [42], tmp_path)


class TestProvenanceFunctions:
    """Test provenance helper functions."""
    
    def test_get_git_commit_returns_string(self):
        """get_git_commit should return non-empty string."""
        commit = get_git_commit()
        assert isinstance(commit, str)
        assert len(commit) > 0
        # Either valid hash or "unknown"
        assert len(commit) == 40 or commit == "unknown"
    
    def test_get_python_version_format(self):
        """get_python_version should return major.minor.micro format."""
        version = get_python_version()
        assert isinstance(version, str)
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)
    
    def test_get_dataset_checksum_from_provenance(self):
        """get_dataset_checksum should read from PROVENANCE.md if available."""
        # Use actual project dataset path
        dataset_path = Path(__file__).parent.parent / "dataset" / "raw" / "train_FD001.txt"
        
        if dataset_path.exists():
            checksum = get_dataset_checksum(dataset_path)
            assert isinstance(checksum, str)
            # Should be MD5 hash or "unavailable"
            assert len(checksum) == 32 or checksum == "unavailable"
            # If PROVENANCE.md exists, should match the documented checksum
            provenance = dataset_path.parent.parent / "PROVENANCE.md"
            if provenance.exists():
                content = provenance.read_text()
                if "1721c96c01e188569f0e7bb16b1ea493" in content:
                    assert checksum == "1721c96c01e188569f0e7bb16b1ea493"
