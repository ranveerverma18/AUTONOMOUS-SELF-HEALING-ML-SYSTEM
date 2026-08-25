"""Manifest-driven experiment matrix runner.

THE MANIFEST IS THE SOURCE OF TRUTH.

This runner reads experiment_manifest.csv and executes exactly the runs
specified in that manifest. It does NOT independently reconstruct or expand
the matrix.

Execution is sequential (--jobs=1 only). Each run executes in its own subprocess
for isolation and independent failure handling.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.config import ExperimentConfig, VALID_STRATEGIES
from scenarios.registry import SCENARIO_REGISTRY


# Valid status values
VALID_STATUSES = {"PLANNED", "RUNNING", "SUCCESS", "FAILED"}


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


def validate_manifest(manifest_path: Path, allow_git_mismatch: bool = False) -> list[dict[str, Any]]:
    """Validate manifest and return rows.
    
    Raises:
        ValueError: If validation fails
    """
    if not manifest_path.exists():
        raise ValueError(f"Manifest not found: {manifest_path}")
    
    # Read manifest
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        raise ValueError(f"Failed to read manifest: {e}")
    
    if not rows:
        raise ValueError("Manifest is empty")
    
    # Required columns
    required_columns = {
        "run_id", "strategy", "scenario", "seed", "status",
        "stream_length", "stream_mode",
        "scenario_onset_cycle_min", "scenario_onset_cycle_max",
        "git_commit",
    }
    
    fieldnames = set(rows[0].keys())
    missing = required_columns - fieldnames
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    
    # Get current config for validation
    current_config = ExperimentConfig()
    current_git = get_git_commit()
    
    # Validate rows
    run_ids = set()
    for i, row in enumerate(rows):
        # Check for missing required values
        for col in required_columns:
            if not row.get(col):
                raise ValueError(f"Row {i}: Missing required value for column '{col}'")
        
        # Unique run_id
        run_id = row["run_id"]
        if run_id in run_ids:
            raise ValueError(f"Duplicate run_id: {run_id}")
        run_ids.add(run_id)
        
        # Valid strategy
        strategy = row["strategy"]
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Row {i} ({run_id}): Invalid strategy '{strategy}'. Must be one of {VALID_STRATEGIES}")
        
        # Valid scenario
        scenario = row["scenario"]
        if scenario not in SCENARIO_REGISTRY:
            raise ValueError(f"Row {i} ({run_id}): Invalid scenario '{scenario}'. Must be one of {list(SCENARIO_REGISTRY.keys())}")
        
        # Valid seed
        try:
            int(row["seed"])
        except ValueError:
            raise ValueError(f"Row {i} ({run_id}): Invalid seed '{row['seed']}'. Must be an integer")
        
        # Valid status
        status = row["status"]
        if status not in VALID_STATUSES:
            raise ValueError(f"Row {i} ({run_id}): Invalid status '{status}'. Must be one of {VALID_STATUSES}")
        
        # Validate critical locked parameters match current config
        try:
            if int(row["stream_length"]) != current_config.stream_length:
                raise ValueError(f"Row {i} ({run_id}): stream_length mismatch. Manifest: {row['stream_length']}, Current: {current_config.stream_length}")
            if row["stream_mode"] != current_config.stream_mode:
                raise ValueError(f"Row {i} ({run_id}): stream_mode mismatch. Manifest: {row['stream_mode']}, Current: {current_config.stream_mode}")
            if int(row["scenario_onset_cycle_min"]) != current_config.scenario_onset_cycle_min:
                raise ValueError(f"Row {i} ({run_id}): scenario_onset_cycle_min mismatch. Manifest: {row['scenario_onset_cycle_min']}, Current: {current_config.scenario_onset_cycle_min}")
            if int(row["scenario_onset_cycle_max"]) != current_config.scenario_onset_cycle_max:
                raise ValueError(f"Row {i} ({run_id}): scenario_onset_cycle_max mismatch. Manifest: {row['scenario_onset_cycle_max']}, Current: {current_config.scenario_onset_cycle_max}")
        except KeyError as e:
            raise ValueError(f"Row {i} ({run_id}): Missing configuration column {e}")
    
    # Git commit check
    manifest_git = rows[0].get("git_commit", "unknown")
    if manifest_git != current_git and manifest_git != "unknown" and current_git != "unknown":
        if not allow_git_mismatch:
            raise ValueError(
                f"Git commit mismatch!\n"
                f"  Manifest was generated at: {manifest_git}\n"
                f"  Current Git commit: {current_git}\n"
                f"Use --allow-git-mismatch to execute anyway (not recommended)"
            )
        else:
            print(f"[WARNING] Git commit mismatch (--allow-git-mismatch enabled)")
            print(f"  Manifest: {manifest_git}")
            print(f"  Current: {current_git}")
    
    return rows


def load_execution_status(status_file: Path) -> dict[str, dict[str, Any]]:
    """Load execution status from file.
    
    Returns:
        Dict mapping run_id to status dict
    """
    if not status_file.exists():
        return {}
    
    try:
        with status_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row["run_id"]: dict(row) for row in reader}
    except Exception:
        return {}


def save_execution_status(status_file: Path, statuses: dict[str, dict[str, Any]]) -> None:
    """Save execution status to file."""
    if not statuses:
        return
    
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "run_id", "status", "started_at", "completed_at",
        "runtime_seconds", "exit_code",
        "raw_csv_path", "summary_json_path",
    ]
    
    with status_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_id in sorted(statuses.keys()):
            writer.writerow(statuses[run_id])


def execute_run(
    row: dict[str, Any],
    output_dir: Path,
    logs_dir: Path,
) -> dict[str, Any]:
    """Execute a single experiment run via subprocess.
    
    Returns:
        Status dict with execution details
    """
    run_id = row["run_id"]
    strategy = row["strategy"]
    scenario = row["scenario"]
    seed = int(row["seed"])
    
    # Construct command
    cmd = [
        sys.executable,
        "-m", "experiments.runner",
        "--strategy", strategy,
        "--scenario", scenario,
        "--seed", str(seed),
        "--output-dir", str(output_dir),
    ]
    
    # Prepare log files
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_dir / f"{run_id}.stdout.log"
    stderr_log = logs_dir / f"{run_id}.stderr.log"
    
    # Execute
    started_at = datetime.now().isoformat()
    start_time = datetime.now()
    
    try:
        with stdout_log.open("w", encoding="utf-8") as stdout_f, \
             stderr_log.open("w", encoding="utf-8") as stderr_f:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                stdout=stdout_f,
                stderr=stderr_f,
                timeout=3600,  # 1 hour timeout per run
            )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = -1
        with stderr_log.open("a", encoding="utf-8") as f:
            f.write("\n[TIMEOUT] Run exceeded 1 hour timeout\n")
    except Exception as e:
        exit_code = -2
        with stderr_log.open("a", encoding="utf-8") as f:
            f.write(f"\n[ERROR] Execution failed: {e}\n")
    
    completed_at = datetime.now().isoformat()
    runtime_seconds = (datetime.now() - start_time).total_seconds()
    
    # Determine actual output paths (runner creates timestamped filenames)
    # Use wildcards since we don't know the exact timestamp
    raw_csv_pattern = str(output_dir / "raw" / f"{strategy}_{scenario}_seed{seed}_*.csv")
    summary_json_pattern = str(output_dir / "aggregated" / f"{strategy}_{scenario}_seed{seed}_*.json")
    
    return {
        "run_id": run_id,
        "status": "SUCCESS" if exit_code == 0 else "FAILED",
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime_seconds": f"{runtime_seconds:.2f}",
        "exit_code": str(exit_code),
        "raw_csv_path": raw_csv_pattern,
        "summary_json_path": summary_json_pattern,
    }


def run_matrix(
    manifest_path: Path,
    output_dir: Path,
    resume: bool = False,
    dry_run: bool = False,
    allow_git_mismatch: bool = False,
    require_confirmation: bool = True,
) -> dict[str, Any]:
    """Run experiment matrix from manifest.
    
    Returns:
        Execution summary dict
    """
    # Validate manifest
    print("Validating manifest...")
    rows = validate_manifest(manifest_path, allow_git_mismatch)
    print(f"[OK] Manifest valid: {len(rows)} rows")
    
    # Load existing execution status
    status_file = output_dir / "matrix_execution_status.csv"
    execution_statuses = load_execution_status(status_file)
    
    # Determine which runs to execute
    planned_runs = []
    skipped_success = []
    
    for row in rows:
        run_id = row["run_id"]
        manifest_status = row["status"]
        
        # Check execution status
        if resume and run_id in execution_statuses:
            exec_status = execution_statuses[run_id].get("status")
            if exec_status == "SUCCESS":
                skipped_success.append(run_id)
                continue
        
        # Only execute PLANNED runs (or retry failed/running if resume)
        if manifest_status == "PLANNED":
            planned_runs.append(row)
        elif resume and manifest_status in {"FAILED", "RUNNING"}:
            planned_runs.append(row)
    
    # Print execution summary
    print()
    print("=" * 60)
    print("EXPERIMENT MATRIX EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Manifest: {manifest_path}")
    print(f"Manifest Git commit: {rows[0].get('git_commit', 'unknown')}")
    print(f"Current Git commit: {get_git_commit()}")
    print(f"Total manifest rows: {len(rows)}")
    print(f"Runs to execute: {len(planned_runs)}")
    if skipped_success:
        print(f"Runs already successful (skipped): {len(skipped_success)}")
    print(f"Jobs (parallelism): 1 (sequential only)")
    print(f"Output directory: {output_dir}")
    print(f"Logs directory: {output_dir / 'logs'}")
    print(f"Status file: {status_file}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 60)
    
    if dry_run:
        print()
        print("DRY RUN MODE - Commands that would be executed:")
        print()
        for row in planned_runs:
            cmd = f"python -m experiments.runner --strategy {row['strategy']} --scenario {row['scenario']} --seed {row['seed']} --output-dir {output_dir}"
            print(f"  {row['run_id']}")
            print(f"    {cmd}")
        print()
        print(f"[OK] Dry run complete. {len(planned_runs)} runs planned. ZERO experiments executed.")
        return {
            "dry_run": True,
            "planned_count": len(planned_runs),
        }
    
    # Confirmation prompt
    if require_confirmation and planned_runs:
        print()
        response = input(f"Execute {len(planned_runs)} experiment runs? [y/N]: ")
        if response.lower() != "y":
            print("Aborted by user.")
            sys.exit(0)
    
    # Execute runs
    print()
    print("Starting execution...")
    print()
    
    start_time = datetime.now()
    logs_dir = output_dir / "logs"
    
    for i, row in enumerate(planned_runs, 1):
        run_id = row["run_id"]
        print(f"[{i}/{len(planned_runs)}] Executing {run_id}...")
        
        status = execute_run(row, output_dir, logs_dir)
        execution_statuses[run_id] = status
        
        # Save status after each run
        save_execution_status(status_file, execution_statuses)
        
        if status["status"] == "SUCCESS":
            print(f"  [SUCCESS] (runtime: {status['runtime_seconds']}s)")
        else:
            print(f"  [FAILED] (exit code: {status['exit_code']})")
            print(f"    See logs: {logs_dir / run_id}.stderr.log")
    
    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds()
    
    # Generate execution summary
    success_count = sum(1 for s in execution_statuses.values() if s["status"] == "SUCCESS")
    failed_count = sum(1 for s in execution_statuses.values() if s["status"] == "FAILED")
    
    summary = {
        "manifest_path": str(manifest_path),
        "manifest_git_commit": rows[0].get("git_commit", "unknown"),
        "current_git_commit": get_git_commit(),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_runtime_seconds": total_runtime,
        "total_planned": len(rows),
        "executed_this_run": len(planned_runs),
        "skipped_success": len(skipped_success),
        "total_success": success_count,
        "total_failed": failed_count,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "runs": execution_statuses,
    }
    
    # Save execution log
    log_file = output_dir / "matrix_execution.log.json"
    with log_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Total runtime: {total_runtime:.1f}s ({total_runtime/60:.1f}min)")
    print(f"Executed: {len(planned_runs)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    if skipped_success:
        print(f"Skipped (already successful): {len(skipped_success)}")
    print(f"Status file: {status_file}")
    print(f"Execution log: {log_file}")
    print("=" * 60)
    
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute experiment matrix from manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute full matrix with confirmation
  python -m scripts.matrix_orchestration.run_matrix \\
    --manifest experiments/results/experiment_manifest.csv
  
  # Dry run (show commands without executing)
  python -m scripts.matrix_orchestration.run_matrix \\
    --manifest experiments/results/experiment_manifest.csv \\
    --dry-run
  
  # Execute without confirmation (automated execution)
  python -m scripts.matrix_orchestration.run_matrix \\
    --manifest experiments/results/experiment_manifest.csv \\
    --yes
  
  # Resume from previous run (skip successful runs)
  python -m scripts.matrix_orchestration.run_matrix \\
    --manifest experiments/results/experiment_manifest.csv \\
    --resume --yes
        """
    )
    
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to experiment manifest CSV (required)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/results)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallelism level (only 1 supported currently)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already succeeded",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing (zero experiments)",
    )
    parser.add_argument(
        "--allow-git-mismatch",
        action="store_true",
        help="Allow execution despite Git commit mismatch (not recommended)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (for automation)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Validate --jobs
    if args.jobs != 1:
        print(f"ERROR: --jobs {args.jobs} not supported. Only sequential execution (--jobs 1) is currently implemented.")
        sys.exit(1)
    
    manifest_path = Path(args.manifest)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Extract from manifest or use default
        output_dir = Path("experiments/results")
    
    try:
        summary = run_matrix(
            manifest_path=manifest_path,
            output_dir=output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
            allow_git_mismatch=args.allow_git_mismatch,
            require_confirmation=not args.yes,
        )
        
        if summary.get("total_failed", 0) > 0:
            sys.exit(1)
        
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
