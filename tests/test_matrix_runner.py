"""Tests for manifest-driven matrix runner."""

import csv
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from scripts.matrix_orchestration.run_matrix import (
    validate_manifest,
    load_execution_status,
    save_execution_status,
    execute_run,
)


class TestManifestValidation:
    """Test manifest validation logic."""
    
    def test_missing_manifest_fails(self, tmp_path):
        """Missing manifest should raise ValueError."""
        manifest = tmp_path / "nonexistent.csv"
        with pytest.raises(ValueError, match="Manifest not found"):
            validate_manifest(manifest)
    
    def test_empty_manifest_fails(self, tmp_path):
        """Empty manifest should raise ValueError."""
        manifest = tmp_path / "empty.csv"
        manifest.write_text("run_id,strategy,scenario,seed,status\n")
        
        with pytest.raises(ValueError, match="Manifest is empty"):
            validate_manifest(manifest)
    
    def test_missing_required_columns_fails(self, tmp_path):
        """Manifest missing required columns should fail."""
        manifest = tmp_path / "invalid.csv"
        manifest.write_text("run_id,strategy\ntest,static\n")
        
        with pytest.raises(ValueError, match="missing required columns"):
            validate_manifest(manifest)
    
    def test_duplicate_run_id_fails(self, tmp_path):
        """Duplicate run_id should raise ValueError."""
        manifest = tmp_path / "duplicate.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_id", "strategy", "scenario", "seed", "status",
                "stream_length", "stream_mode",
                "scenario_onset_cycle_min", "scenario_onset_cycle_max",
                "git_commit",
            ])
            writer.writeheader()
            writer.writerow({
                "run_id": "static_gradual_drift_seed42",
                "strategy": "static",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "stream_length": "2400",
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": "25",
                "scenario_onset_cycle_max": "35",
                "git_commit": "abc123",
            })
            writer.writerow({
                "run_id": "static_gradual_drift_seed42",  # Duplicate
                "strategy": "static",
                "scenario": "gradual_drift",
                "seed": "123",
                "status": "PLANNED",
                "stream_length": "2400",
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": "25",
                "scenario_onset_cycle_max": "35",
                "git_commit": "abc123",
            })
        
        with pytest.raises(ValueError, match="Duplicate run_id"):
            validate_manifest(manifest)
    
    def test_invalid_strategy_fails(self, tmp_path):
        """Invalid strategy should raise ValueError."""
        manifest = tmp_path / "invalid_strategy.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_id", "strategy", "scenario", "seed", "status",
                "stream_length", "stream_mode",
                "scenario_onset_cycle_min", "scenario_onset_cycle_max",
                "git_commit",
            ])
            writer.writeheader()
            writer.writerow({
                "run_id": "invalid_gradual_drift_seed42",
                "strategy": "invalid_strategy",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "stream_length": "2400",
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": "25",
                "scenario_onset_cycle_max": "35",
                "git_commit": "abc123",
            })
        
        with pytest.raises(ValueError, match="Invalid strategy"):
            validate_manifest(manifest)
    
    def test_invalid_scenario_fails(self, tmp_path):
        """Invalid scenario should raise ValueError."""
        manifest = tmp_path / "invalid_scenario.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_id", "strategy", "scenario", "seed", "status",
                "stream_length", "stream_mode",
                "scenario_onset_cycle_min", "scenario_onset_cycle_max",
                "git_commit",
            ])
            writer.writeheader()
            writer.writerow({
                "run_id": "static_invalid_seed42",
                "strategy": "static",
                "scenario": "invalid_scenario",
                "seed": "42",
                "status": "PLANNED",
                "stream_length": "2400",
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": "25",
                "scenario_onset_cycle_max": "35",
                "git_commit": "abc123",
            })
        
        with pytest.raises(ValueError, match="Invalid scenario"):
            validate_manifest(manifest)
    
    def test_invalid_seed_fails(self, tmp_path):
        """Invalid seed should raise ValueError."""
        manifest = tmp_path / "invalid_seed.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_id", "strategy", "scenario", "seed", "status",
                "stream_length", "stream_mode",
                "scenario_onset_cycle_min", "scenario_onset_cycle_max",
                "git_commit",
            ])
            writer.writeheader()
            writer.writerow({
                "run_id": "static_gradual_drift_seedXYZ",
                "strategy": "static",
                "scenario": "gradual_drift",
                "seed": "not_a_number",
                "status": "PLANNED",
                "stream_length": "2400",
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": "25",
                "scenario_onset_cycle_max": "35",
                "git_commit": "abc123",
            })
        
        with pytest.raises(ValueError, match="Invalid seed"):
            validate_manifest(manifest)
    
    @patch("scripts.matrix_orchestration.run_matrix.get_git_commit")
    def test_git_mismatch_fails_by_default(self, mock_git, tmp_path):
        """Git mismatch should fail without --allow-git-mismatch."""
        mock_git.return_value = "current_commit_xyz"
        
        manifest = tmp_path / "git_mismatch.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_id", "strategy", "scenario", "seed", "status",
                "stream_length", "stream_mode",
                "scenario_onset_cycle_min", "scenario_onset_cycle_max",
                "git_commit",
            ])
            writer.writeheader()
            writer.writerow({
                "run_id": "static_gradual_drift_seed42",
                "strategy": "static",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "stream_length": "2400",
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": "25",
                "scenario_onset_cycle_max": "35",
                "git_commit": "old_commit_abc",
            })
        
        with pytest.raises(ValueError, match="Git commit mismatch"):
            validate_manifest(manifest, allow_git_mismatch=False)
    
    @patch("scripts.matrix_orchestration.run_matrix.get_git_commit")
    def test_allow_git_mismatch_permits_execution(self, mock_git, tmp_path):
        """--allow-git-mismatch should permit execution despite mismatch."""
        mock_git.return_value = "current_commit_xyz"
        
        manifest = tmp_path / "git_mismatch_allowed.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_id", "strategy", "scenario", "seed", "status",
                "stream_length", "stream_mode",
                "scenario_onset_cycle_min", "scenario_onset_cycle_max",
                "git_commit",
            ])
            writer.writeheader()
            writer.writerow({
                "run_id": "static_gradual_drift_seed42",
                "strategy": "static",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "stream_length": "2400",
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": "25",
                "scenario_onset_cycle_max": "35",
                "git_commit": "old_commit_abc",
            })
        
        # Should not raise
        rows = validate_manifest(manifest, allow_git_mismatch=True)
        assert len(rows) == 1
    
    def test_valid_manifest_returns_rows(self, tmp_path):
        """Valid manifest should return rows."""
        manifest = tmp_path / "valid.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_id", "strategy", "scenario", "seed", "status",
                "stream_length", "stream_mode",
                "scenario_onset_cycle_min", "scenario_onset_cycle_max",
                "git_commit",
            ])
            writer.writeheader()
            writer.writerow({
                "run_id": "static_gradual_drift_seed42",
                "strategy": "static",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "stream_length": "2400",
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": "25",
                "scenario_onset_cycle_max": "35",
                "git_commit": "unknown",
            })
            writer.writerow({
                "run_id": "proposed_sudden_spike_seed123",
                "strategy": "proposed",
                "scenario": "sudden_spike",
                "seed": "123",
                "status": "PLANNED",
                "stream_length": "2400",
                "stream_mode": "interleaved",
                "scenario_onset_cycle_min": "25",
                "scenario_onset_cycle_max": "35",
                "git_commit": "unknown",
            })
        
        rows = validate_manifest(manifest, allow_git_mismatch=True)
        assert len(rows) == 2
        assert rows[0]["run_id"] == "static_gradual_drift_seed42"
        assert rows[1]["run_id"] == "proposed_sudden_spike_seed123"


class TestExecutionStatus:
    """Test execution status tracking."""
    
    def test_load_nonexistent_status_returns_empty(self, tmp_path):
        """Loading nonexistent status file should return empty dict."""
        status_file = tmp_path / "nonexistent_status.csv"
        statuses = load_execution_status(status_file)
        assert statuses == {}
    
    def test_save_and_load_status(self, tmp_path):
        """Saved status should be loadable."""
        status_file = tmp_path / "status.csv"
        
        statuses = {
            "run1": {
                "run_id": "run1",
                "status": "SUCCESS",
                "started_at": "2026-08-24T10:00:00",
                "completed_at": "2026-08-24T10:05:00",
                "runtime_seconds": "300.0",
                "exit_code": "0",
                "raw_csv_path": "path/to/raw.csv",
                "summary_json_path": "path/to/summary.json",
            },
            "run2": {
                "run_id": "run2",
                "status": "FAILED",
                "started_at": "2026-08-24T10:05:00",
                "completed_at": "2026-08-24T10:06:00",
                "runtime_seconds": "60.0",
                "exit_code": "1",
                "raw_csv_path": "path/to/raw2.csv",
                "summary_json_path": "path/to/summary2.json",
            },
        }
        
        save_execution_status(status_file, statuses)
        
        loaded = load_execution_status(status_file)
        assert len(loaded) == 2
        assert loaded["run1"]["status"] == "SUCCESS"
        assert loaded["run2"]["status"] == "FAILED"


class TestDryRun:
    """Test dry run mode."""
    
    @patch("scripts.matrix_orchestration.run_matrix.validate_manifest")
    def test_dry_run_executes_zero_experiments(self, mock_validate, tmp_path):
        """Dry run should execute zero experiments."""
        mock_validate.return_value = [
            {
                "run_id": "static_gradual_drift_seed42",
                "strategy": "static",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "git_commit": "abc123",
            }
        ]
        
        from scripts.matrix_orchestration.run_matrix import run_matrix
        
        summary = run_matrix(
            manifest_path=tmp_path / "fake.csv",
            output_dir=tmp_path / "output",
            dry_run=True,
            require_confirmation=False,
        )
        
        assert summary["dry_run"] is True
        assert summary["planned_count"] == 1
        
        # Verify no outputs created
        assert not (tmp_path / "output" / "raw").exists()
        assert not (tmp_path / "output" / "aggregated").exists()


class TestResumeBehavior:
    """Test resume functionality."""
    
    @patch("scripts.matrix_orchestration.run_matrix.validate_manifest")
    @patch("scripts.matrix_orchestration.run_matrix.load_execution_status")
    def test_resume_skips_successful_runs(self, mock_load, mock_validate, tmp_path):
        """Resume should skip runs marked SUCCESS."""
        mock_validate.return_value = [
            {
                "run_id": "run1",
                "strategy": "static",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "git_commit": "abc",
            },
            {
                "run_id": "run2",
                "strategy": "proposed",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "git_commit": "abc",
            },
        ]
        
        # run1 already successful
        mock_load.return_value = {
            "run1": {"status": "SUCCESS"}
        }
        
        from scripts.matrix_orchestration.run_matrix import run_matrix
        
        summary = run_matrix(
            manifest_path=tmp_path / "fake.csv",
            output_dir=tmp_path / "output",
            resume=True,
            dry_run=True,
            require_confirmation=False,
        )
        
        # Only run2 should be planned (run1 skipped)
        assert summary["planned_count"] == 1


class TestManifestRowHandling:
    """Test that manifest rows are not expanded."""
    
    @patch("scripts.matrix_orchestration.run_matrix.validate_manifest")
    def test_manifest_rows_not_expanded(self, mock_validate, tmp_path):
        """Runner should execute exactly the rows in manifest, no expansion."""
        # Manifest has 2 rows
        mock_validate.return_value = [
            {
                "run_id": "static_gradual_drift_seed42",
                "strategy": "static",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "git_commit": "abc",
            },
            {
                "run_id": "proposed_sudden_spike_seed123",
                "strategy": "proposed",
                "scenario": "sudden_spike",
                "seed": "123",
                "status": "PLANNED",
                "git_commit": "abc",
            },
        ]
        
        from scripts.matrix_orchestration.run_matrix import run_matrix
        
        summary = run_matrix(
            manifest_path=tmp_path / "fake.csv",
            output_dir=tmp_path / "output",
            dry_run=True,
            require_confirmation=False,
        )
        
        # Exactly 2 runs planned, not expanded to 96
        assert summary["planned_count"] == 2
    
    @patch("scripts.matrix_orchestration.run_matrix.validate_manifest")
    def test_only_planned_rows_selected(self, mock_validate, tmp_path):
        """Only PLANNED status rows should be selected for execution."""
        mock_validate.return_value = [
            {
                "run_id": "run1",
                "strategy": "static",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "git_commit": "abc",
            },
            {
                "run_id": "run2",
                "strategy": "proposed",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "SUCCESS",  # Already complete
                "git_commit": "abc",
            },
            {
                "run_id": "run3",
                "strategy": "naive_adaptive",
                "scenario": "gradual_drift",
                "seed": "42",
                "status": "PLANNED",
                "git_commit": "abc",
            },
        ]
        
        from scripts.matrix_orchestration.run_matrix import run_matrix
        
        summary = run_matrix(
            manifest_path=tmp_path / "fake.csv",
            output_dir=tmp_path / "output",
            dry_run=True,
            require_confirmation=False,
        )
        
        # Only 2 PLANNED runs selected (run2 has SUCCESS status)
        assert summary["planned_count"] == 2
