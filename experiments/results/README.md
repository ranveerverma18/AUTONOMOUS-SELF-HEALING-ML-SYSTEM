# Experiment Results

This directory contains the final compact research artifacts for the completed
96-run experiment matrix (4 strategies × 8 scenarios × 3 seeds), executed under
the locked protocol `stream_length=2400`, `stream_mode=interleaved`,
`scenario_onset_cycle=[25, 35]`.

## Status

✅ **Complete.** 96/96 runs succeeded, 0 failed, 0 skipped.

## Committed Artifacts (tracked in Git)

| File | Description |
|------|-------------|
| `experiment_manifest.csv` | Deterministic 96-run manifest (strategy × scenario × seed) |
| `aggregated_results.csv` | One row per run: MAE, RMSE, detection/adaptation/recovery metrics |
| `aggregated_results.qc.json` | Structural and statistical QC report (0 issues) |
| `statistical_analysis.json` / `.md` | 5 Friedman tests + 30 Wilcoxon pairwise comparisons (Holm-Bonferroni corrected); 27/30 significant |
| `provenance.json` | Dataset checksum, git commit lineage, software environment, full run provenance |
| `verification_report.json` / `.md` | Per-run completion verification (96/96 valid, 0 missing, 0 duplicate) |
| `figures/*.png` | 8 publication-quality figures (300 DPI) generated from the frozen results |
| `README.md` | This file |

## Artifacts Preserved Outside Git (external research archive)

The following detailed, per-run outputs are **not** committed to Git (see
`.gitignore`) to keep the repository lightweight. They remain on disk locally
and are preserved with checksums in the external research archive:

```
results/
├── raw/         # 96 event-level CSV files (one per run)
├── aggregated/  # 96 per-run summary JSON files
├── logs/        # stdout/stderr execution logs
└── mini_matrix/ # small-scale validation runs used to build this pipeline (not part of the 96-run matrix)
```

Also excluded from Git: `matrix_execution.log.json` and
`matrix_execution_status.csv` (raw per-run execution timing/status ledger —
their summary information is already captured in `verification_report.*` and
`provenance.json`).

None of this is required to inspect, verify, or cite the final results — the
compact artifacts above are self-contained. See the main project `README.md`
("Reproducing the Analysis" / "Reproducing the Experiment From Scratch") for
how to regenerate them if needed.

## Run ID Format

Run IDs are deterministic:
```
{strategy}_{scenario}_seed{seed}
```

Examples:
- `static_gradual_drift_seed42`
- `proposed_sudden_spike_seed123`
- `scheduled_high_noise_seed456`

## Reproducing the Analysis

The analysis (QC, statistics, figures) can be regenerated from the committed
`aggregated_results.csv` without re-executing any experiments — see the main
project `README.md` for exact commands.
