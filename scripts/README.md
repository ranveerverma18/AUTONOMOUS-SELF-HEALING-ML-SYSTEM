# Reproducibility Scripts

This directory contains the reproducibility pipeline used to execute, verify,
aggregate, and analyze the final 96-run research matrix.

## Directory Structure

```
scripts/
├── matrix_orchestration/    # Manifest generation, matrix execution, completion verification
└── analysis/                # Result aggregation, statistical QC, statistical analysis, figures
```

## Status

✅ **Pipeline complete.** The final 96-run experiment matrix has already been executed
successfully (96/96 runs, 0 failed, 0 skipped) using this pipeline. The compact
final artifacts are committed under `experiments/results/` (see
`experiments/results/README.md`).

These scripts remain in the repository so that:
1. The statistical analysis and figures can be regenerated deterministically from
   the frozen `experiments/results/aggregated_results.csv` without rerunning any
   experiments.
2. The full experiment can be independently re-executed from scratch if desired.

## Pipeline Stages

```
generate_manifest → run_matrix → verify_completion → aggregate_results
  → statistical_qc → statistical_analysis → generate_figures
```

### Matrix Orchestration (`matrix_orchestration/`)

- `generate_manifest.py` — Generate the deterministic pre-execution experiment manifest.
  ```bash
  python -m scripts.matrix_orchestration.generate_manifest --output-dir experiments/results
  ```
- `run_matrix.py` — Execute the full factorial matrix (4 strategies × 8 scenarios × 3 seeds)
  by invoking `experiments/runner.py` once per manifest row.
  ```bash
  python -m scripts.matrix_orchestration.run_matrix --manifest experiments/results/experiment_manifest.csv --yes
  ```
- `verify_completion.py` — Independently verify that every manifest run produced valid,
  non-duplicate outputs on disk (does not trust the execution status file alone).
  ```bash
  python -m scripts.matrix_orchestration.verify_completion --manifest experiments/results/experiment_manifest.csv --results-dir experiments/results --strict
  ```

### Analysis (`analysis/`)

- `aggregate_results.py` — Combine the 96 raw event-level CSVs and summary JSONs into a
  single compact `aggregated_results.csv`.
  ```bash
  python -m scripts.analysis.aggregate_results --manifest experiments/results/experiment_manifest.csv --results-dir experiments/results --output experiments/results/aggregated_results.csv
  ```
- `statistical_qc.py` — Run structural/statistical quality checks on the aggregated table.
  ```bash
  python -m scripts.analysis.statistical_qc --aggregated experiments/results/aggregated_results.csv --manifest experiments/results/experiment_manifest.csv --output experiments/results/aggregated_results.qc.json
  ```
- `statistical_analysis.py` — Friedman tests (per metric) + Wilcoxon signed-rank pairwise
  comparisons with Holm-Bonferroni correction, blocked by `scenario_seed`.
  ```bash
  python -m scripts.analysis.statistical_analysis --aggregated experiments/results/aggregated_results.csv --manifest experiments/results/experiment_manifest.csv --qc-report experiments/results/aggregated_results.qc.json --output experiments/results
  ```
- `generate_figures.py` — Generate the 8 publication-quality PNG figures from the frozen
  aggregated results and statistical analysis.
  ```bash
  python -m scripts.analysis.generate_figures --aggregated experiments/results/aggregated_results.csv --statistical experiments/results/statistical_analysis.json --output experiments/results/figures
  ```

## Design Principles

All scripts:
1. Accept command-line arguments (no hardcoded paths)
2. Validate inputs before execution
3. Log actions to stdout/stderr
4. Return meaningful exit codes (0=success)
5. Generate structured outputs (JSON + Markdown)
6. Record complete provenance
7. Are deterministic given the same input data

## Single-Run Execution (without orchestration)

A single experimental run can still be executed directly via `experiments/runner.py`:

```bash
python -m experiments.runner \
  --strategy proposed \
  --scenario gradual_drift \
  --seed 42 \
  --stream-mode interleaved \
  --stream-length 2400 \
  --scenario-onset-cycle-min 25 \
  --scenario-onset-cycle-max 35
```

See `experiments/README.md` and the main project `README.md` for full documentation,
including the distinction between reproducing the analysis (fast, no experiments
executed) and reproducing the experiment from scratch (long-running).
