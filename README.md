# Autonomous Self-Healing ML System

A production-grade, real-time machine learning pipeline for predictive maintenance of turbofan aircraft engines. The system continuously monitors a live sensor stream, detects model degradation, and autonomously retrains, shadow-evaluates, and promotes improved models — all without operator intervention.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Fault Injection Scenarios](#fault-injection-scenarios)
- [Test Suite](#test-suite)
- [Research Experiment Framework](#research-experiment-framework)
- [Research Paper](#research-paper)
- [Pipeline Deep Dive](#pipeline-deep-dive)

---

## Overview

This system trains a **Random Forest regressor** on the NASA CMAPSS turbofan engine dataset (FD001) to predict **Remaining Useful Life (RUL)** in cycles. A live sensor stream is continuously fed through a multi-layer monitoring pipeline that:

- Detects **concept drift** (rising prediction error) via ADWIN
- Detects **feature drift** (distributional shift) via KS-test with Bonferroni correction
- Detects **sensor anomalies** (outlier data points) via Isolation Forest
- Autonomously triggers **background retraining** when drift is confirmed
- Validates the candidate model through a **shadow A/B evaluation** before promoting it
- Enforces an **adaptive ingestion rate controller** to protect the ML worker under load
- Exposes all runtime state through a **FastAPI dashboard API** consumed by a React frontend

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Sensor Stream (live)                         │
│   Simulated from NASA CMAPSS train_FD001.txt (unit-disjoint split)   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │   Rate Limit Controller│  ← adaptive ingestion gate
              │   (500-event queue)    │    load-shedding + throttling
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   ML Worker Thread    │
              │  (locked tick loop)   │
              └────┬──────┬──────┬───┘
                   │      │      │
        ┌──────────▼──┐ ┌─▼──────▼──────┐ ┌────────────────┐
        │  Isolation   │ │  Random Forest │ │  Shadow Model   │
        │  Forest      │ │  (production)  │ │  (candidate)    │
        │  Anomaly Det.│ │  prediction +  │ │  A/B eval       │
        └──────────────┘ │  confidence CI │ └────────────────┘
                         └───────┬────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
        ┌────────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐
        │  ADWIN Concept│ │  KS-test    │ │  Error Monitor  │
        │  Drift Detect.│ │  Feature    │ │  (rolling MAE)  │
        └────────┬──────┘ │  Drift Det. │ └─────┬──────────┘
                 │        └──────┬──────┘       │
                 └───────────────┴───────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Decision Engine      │
                    │  STABLE / WATCH /        │
                    │  MONITOR / ALERT /       │
                    │  RETRAIN / RETRAIN_URGENT│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Adaptive Cooldown      │
                    │   + Background Retrain   │
                    │   → Performance Gate     │
                    │   → Shadow Evaluation    │
                    │   → Model Promotion      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Governance / Audit Log │
                    │   (ring-buffered events) │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   FastAPI Dashboard API  │
                    │   ← React Frontend       │
                    └─────────────────────────┘
```

---

## Features

| Capability | Implementation |
|---|---|
| RUL prediction | Random Forest (200 trees, depth 12) with 90% confidence intervals via tree variance |
| Concept drift detection | ADWIN on rolling prediction error stream |
| Feature drift detection | KS-test (Bonferroni corrected) across all 24 sensor/setting columns |
| Anomaly detection | Isolation Forest on incoming sensor feature vectors |
| Autonomous retraining | Background thread triggered by cooldown + drift threshold |
| Model quality gate | New model must achieve ≥5% MAE improvement over production |
| Shadow A/B evaluation | Candidate runs in parallel for a 20-event live shadow window before promotion |
| Adaptive rate control | Smoothed EPS controller with 4 states: Nominal / Throttling / Protecting / Draining |
| Load shedding | FIFO drop with audit trail when 500-event queue is full |
| Governance audit trail | Bounded ring-buffers for audit logs, alerts, timeline, and model history |
| Fault injection | 8 parameterised sensor fault scenarios via REST API |
| Live dashboard | Polling REST API serving structured JSON snapshots every 500 ms |

---

## Technology Stack

**Backend**
- Python 3.12
- [FastAPI](https://fastapi.tiangolo.com/) — async REST API framework
- [Uvicorn](https://www.uvicorn.org/) — ASGI server
- [scikit-learn](https://scikit-learn.org/) — Random Forest, Isolation Forest, StandardScaler
- [river](https://riverml.xyz/) — ADWIN online drift detector
- [scipy](https://scipy.org/) — KS-test for feature drift
- [NumPy](https://numpy.org/) / [pandas](https://pandas.pydata.org/) — numerical processing

**Frontend**
- React 18 + Vite
- Recharts — data visualisation
- Vanilla CSS

**Testing**
- pytest with 113 tests across all modules

---

## Project Structure

```
AUTONOMOUS-SELF-HEALING-ML-SYSTEM/
│
├── api_server.py               # FastAPI app + StreamingMLRuntime (main entrypoint)
│
├── decision/
│   ├── engine.py               # Multi-state decision engine (STABLE → RETRAIN_URGENT)
│   └── adaptive_cooldown.py    # Drift-severity-aware retrain cooldown
│
├── drift/
│   ├── adwin_detector.py       # ADWIN concept drift detection
│   ├── data_drift.py           # KS-test feature drift detection
│   ├── anomaly_detector.py     # Isolation Forest anomaly detection
│   └── error_monitor.py        # Rolling MAE window + trend detection
│
├── ml/
│   ├── train.py                # train_model_with_holdout (unit-based split, configurable seed)
│   ├── predict.py              # Single-point inference helper
│   ├── evaluation.py           # split_training_and_validation (disjoint unit/temporal split)
│   ├── confidence_predictor.py # RF tree variance → confidence intervals
│   ├── performance_gate.py     # Accept/reject candidate model by MAE delta
│   └── shadow_evaluator.py     # A/B shadow evaluation before promotion
│
├── rate_limiting/
│   └── controller.py           # Adaptive ingestion rate controller + queue
│
├── metrics/
│   └── calculator.py           # Stateless dashboard metric helpers
│
├── governance/
│   └── audit_log.py            # Ring-buffered audit/alert/timeline/history store
│
├── scenarios/
│   ├── registry.py             # Central scenario registry
│   ├── gradual_drift.py        # Progressive sensor wear (+0.3σ / 10 cycles)
│   ├── sudden_spike.py         # Instant +8σ spike on all 21 sensors
│   ├── high_noise.py           # ±4.5σ zero-mean noise injection
│   ├── sensor_failure.py       # Flat-line failure on sensor_3 and sensor_9
│   ├── concept_drift.py        # RUL label shift (−60 cycles, features stable)
│   ├── correlated_drift.py     # Correlated shift across sensor pairs
│   ├── intermittent_spikes.py  # ±12σ spikes every 7th cycle
│   └── drift_recovery.py       # Drift then gradual recovery
│
├── experiments/                # Standalone research experiment framework
│   ├── config.py               # ExperimentConfig (frozen dataclass, validated)
│   ├── data_stream.py          # Deterministic interleaved fleet stream with per-engine scenarios
│   ├── baselines.py            # 4 experiment strategies (static, scheduled, naive, proposed)
│   ├── scenarios.py            # 8 parameterized degradation scenarios  
│   ├── metrics.py              # RunSummary + summarize_events()
│   ├── evaluator.py            # Per-run evaluation helpers
│   ├── aggregation.py          # Result aggregation helpers used by scripts/analysis
│   ├── statistical_tests.py    # Friedman/Wilcoxon helpers used by scripts/analysis
│   ├── runner.py               # CLI entry point (single-run execution)
│   └── results/                # Final 96-run compact artifacts (raw/aggregated/logs are gitignored)
│
├── scripts/                    # Reproducibility pipeline (see scripts/README.md)
│   ├── matrix_orchestration/   # generate_manifest.py, run_matrix.py, verify_completion.py
│   └── analysis/                # aggregate_results.py, statistical_qc.py, statistical_analysis.py, generate_figures.py
│
├── dataset/
│   ├── raw/
│   │   ├── train_FD001.txt     # NASA CMAPSS dataset (100 engine units, active)
│   │   ├── test_FD001.txt      # NASA CMAPSS test split (present, not used by the research matrix)
│   │   └── RUL_FD001.txt       # True RUL for test split (present, not used by the research matrix)
│   ├── processed/
│   │   └── preprocess_module.py
│   └── PROVENANCE.md           # Dataset source, citation, checksums
│
├── tests/                      # pytest suite (258 tests across 28 modules)
│   ├── test_decision_engine.py
│   ├── test_adaptive_cooldown.py
│   ├── test_adwin_detector.py
│   ├── test_anomaly_detector.py
│   ├── test_audit_log.py
│   ├── test_confidence_predictor.py
│   ├── test_data_drift.py
│   ├── test_error_monitor.py
│   ├── test_metrics_calculator.py
│   ├── test_model_pipeline.py
│   ├── test_predict.py
│   ├── test_preprocess.py
│   ├── test_rate_limiting.py
│   ├── test_scenarios.py
│   ├── test_train.py
│   ├── test_common_validation.py
│   ├── test_experiment_config.py
│   ├── test_experiment_metrics.py
│   ├── test_experiment_strategies.py
│   ├── test_experiment_stream.py    # Includes interleaved ordering, per-engine onset tests
│   ├── test_scenario_logic.py
│   ├── test_validation_quality.py
│   ├── test_manifest_generator.py   # scripts/matrix_orchestration/generate_manifest.py
│   ├── test_matrix_runner.py        # scripts/matrix_orchestration/run_matrix.py (mocked, no real execution)
│   ├── test_completion_verifier.py  # scripts/matrix_orchestration/verify_completion.py
│   ├── test_result_aggregation.py   # scripts/analysis/aggregate_results.py
│   ├── test_statistical_qc.py       # scripts/analysis/statistical_qc.py
│   └── test_statistical_analysis.py # scripts/analysis/statistical_analysis.py
│
├── frontend/                   # React + Vite dashboard
│   └── src/
│       ├── pages/
│       ├── charts/
│       └── components/
│
└── requirements.txt
```

---

## Setup and Installation

### Prerequisites

- Python 3.12+
- Node.js 18+ (for the frontend)

### 1. Clone the repository

```bash
git clone https://github.com/RiteshKrChauhan/AUTONOMOUS-SELF-HEALING-ML-SYSTEM.git
cd AUTONOMOUS-SELF-HEALING-ML-SYSTEM
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `numpy>=2.3.4` is pinned to prevent the numpy ABI mismatch between river 0.25 and scikit-learn wheels. Do not downgrade it.

### 3. Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Running the Application

### Start the backend API server

```bash
# From the project root
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

The server will:
1. Load and preprocess the NASA CMAPSS dataset
2. Train the initial Random Forest model
3. Start the background streaming worker
4. Begin serving the dashboard API at `http://127.0.0.1:8000`

### Start the frontend development server

```bash
# From the frontend/ directory
npm run dev
```

Open `http://localhost:5173` in your browser.

Both servers must be running simultaneously. The frontend polls `http://127.0.0.1:8000/api/dashboard` every 500 ms.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check — returns model version |
| `GET` | `/api/dashboard` | Full dashboard snapshot (metrics, series, audit trail) |
| `GET` | `/api/scenarios` | List all available fault injection scenarios |
| `POST` | `/api/anomalies` | Inject a fault scenario into the live stream |
| `POST` | `/api/controls` | Update stream rate and rate-limit settings |
| `POST` | `/api/reset` | Fully restart the streaming runtime |

### POST `/api/anomalies`

```json
{ "scenario": "sudden_spike" }
```

### POST `/api/controls`

```json
{
  "simulatedRate": 12.0,
  "rateLimit": 20.0,
  "rateLimitEnabled": true
}
```

| Field | Range | Description |
|---|---|---|
| `simulatedRate` | 0.5 – 30.0 | Incoming events per second |
| `rateLimit` | 1.0 – 40.0 | Maximum processing rate (eps) |
| `rateLimitEnabled` | bool | Toggle adaptive rate limiting |

---

## Fault Injection Scenarios

Scenarios are injected via `POST /api/anomalies` and run for a fixed number of cycles.

| Scenario ID | Name | Severity | Duration | What It Does |
|---|---|---|---|---|
| `gradual_drift` | Gradual Sensor Drift | Medium | 100 cycles | Four key sensors drift by +0.3σ every 10 cycles — simulates progressive component wear |
| `sudden_spike` | Sudden Sensor Drift | Critical | 80 cycles | All 21 sensors shift by +8σ immediately — simulates a severe environmental or mechanical fault |
| `high_noise` | High Sensor Noise | Medium | 60 cycles | All sensors receive ±4.5σ zero-mean noise — simulates electrical interference |
| `sensor_failure` | Stuck Sensor Failure | High | 80 cycles | sensor_3 and sensor_9 lock to 0.0 — simulates disconnected or grounded hardware |
| `concept_drift` | Concept Drift | High | 150 cycles | RUL labels shift −60 cycles while sensor features remain within normal ranges |
| `correlated_drift` | Correlated Sensor Drift | High | 60 cycles | Six temperature and pressure sensors drift by +3σ together — simulates a thermal event |
| `intermittent_spikes` | Intermittent Sensor Spikes | Low | 90 cycles | Two random sensors spike by ±12σ every 7th cycle — normal readings between spikes |
| `drift_recovery` | Drift to Recovery | Critical | 60 cycles | Severe +6σ drift for 30 cycles, then gradual return to baseline over 30 cycles |

### Detection expectations

| Scenario | ADWIN | KS-test | Isolation Forest |
|---|---|---|---|
| Gradual Sensor Drift | ✓ after ~40–60 cycles | ✓ | — |
| Sudden Sensor Drift | ✓ quickly | ✓ | ✓ |
| High Sensor Noise | — | ✓ (variance shift) | ✓ |
| Stuck Sensor Failure | ✓ | ✓ (zero variance) | ✓ |
| Concept Drift | ✓ (error rises) | — (features stable) | — |
| Correlated Sensor Drift | ✓ | ✓ (multi-feature) | — |
| Intermittent Sensor Spikes | — (intermittent) | — | ✓ spike cycles |
| Drift to Recovery | ✓ | ✓ | — |


---

## Test Suite

Run all 258 tests with:

```bash
python -m pytest tests/ -v
```

| Test File | Tests | Coverage |
|---|---|---|
| `test_decision_engine.py` | 6 | All 6 decision states (STABLE, WATCH, MONITOR, ALERT, RETRAIN, RETRAIN_URGENT) |
| `test_adaptive_cooldown.py` | 5 | All 3 drift-severity cooldown branches, mark_retrain, get_status |
| `test_adwin_detector.py` | 3 | Stable stream, step-change detection, return type |
| `test_anomaly_detector.py` | 5 | Fit success/fail, return types, unfitted guard, outlier scoring |
| `test_audit_log.py` | 5 | Prepend order, severity storage, model history deduplication, ring-buffer cap |
| `test_confidence_predictor.py` | 4 | 4-tuple return, interval ordering, category boundaries, missing-feature error |
| `test_data_drift.py` | 5 | Warmup phases, stable/drifted data, invalid input guard |
| `test_error_monitor.py` | 2 | Warmup + trend detection, non-increasing case |
| `test_metrics_calculator.py` | 8 | confidence_value clipping, all 3 status_from_metrics states, feature_scores, histogram |
| `test_model_pipeline.py` | 11 | PerformanceGate guards + acceptance, ShadowEvaluator lifecycle + promotion |
| `test_predict.py` | 2 | Extra-key handling, missing-feature ValueError |
| `test_preprocess.py` | 3 | RUL computation, whitespace-delimited parsing, bad-column error |
| `test_rate_limiting.py` | 6 | Enqueue, load shedding at capacity, update return, bypass mode, controls, snapshot reset |
| `test_scenarios.py` | 5 | Registry completeness, META fields, apply() mutation, zero-cycle safety |
| `test_scenario_logic.py` | 5 | Per-scenario detectability assertions (KS, ADWIN, IsolationForest) |
| `test_train.py` | 4 | Normal training, insufficient-data guard, single-unit fallback, model inference |
| `test_common_validation.py` | 5 | Disjoint train/val split, no leakage, common validation set integrity |
| `test_experiment_config.py` | 4 | Config validation, output directory creation, path types, runner CLI defaults regression |
| `test_experiment_metrics.py` | 6 | Metric definitions, validation-skip counting, gate-reject separation, recovery metrics |
| `test_experiment_strategies.py` | 1 | Static strategy never retrains |
| `test_experiment_stream.py` | 8 | Interleaved stream ordering, per-engine scenario onset determinism, cycle monotonicity |
| `test_validation_quality.py` | 11 | Buffer/validation-quality policy, configurable RF seed, config field validation |
| `test_manifest_generator.py` | 17 | Deterministic 96-run manifest generation (`scripts/matrix_orchestration/generate_manifest.py`) |
| `test_matrix_runner.py` | 16 | Matrix orchestration logic, mocked subprocess execution — no real experiments run (`scripts/matrix_orchestration/run_matrix.py`) |
| `test_completion_verifier.py` | 40 | Per-run output validation against manifest (`scripts/matrix_orchestration/verify_completion.py`) |
| `test_result_aggregation.py` | 11 | Raw/summary aggregation into compact results table (`scripts/analysis/aggregate_results.py`) |
| `test_statistical_qc.py` | 27 | Structural/statistical QC checks (`scripts/analysis/statistical_qc.py`) |
| `test_statistical_analysis.py` | 33 | Friedman/Wilcoxon/Holm-Bonferroni statistical pipeline (`scripts/analysis/statistical_analysis.py`) |

---

## Research Experiment Framework

The `experiments/` module provides a standalone framework for reproducible research experiments comparing adaptive ML strategies under controlled degradation scenarios.

### Study Design: Adaptation-Under-Degradation

This research protocol focuses on **autonomous adaptation to ongoing degradation** in fleet-wide predictive maintenance. The 2400-event stream corresponds to approximately **100 lifecycle cycles** across the 24-engine fleet, intentionally configured to observe detection, adaptation, and continued performance under progressive degradation rather than complete degradation lifecycles.

### Stream Modes

- **interleaved** (default, LOCKED for research): Fleet-wide monitoring where observations from all engines are grouped by lifecycle cycle. Simulates parallel monitoring of multiple assets. For cycle c=1,2,3,...: include all engines at cycle c, ordered by unit ID.
- **research**: Sequential single-asset monitoring preserving full per-engine lifecycle trajectories (unit, cycle order).
- **legacy**: Dashboard-compatible random permutation with configurable seed.

### Per-Engine Scenario Semantics

Scenarios are applied **per-engine** based on each engine's lifecycle cycle, not global stream position. Each engine receives a deterministic onset cycle within `[scenario_onset_cycle_min, scenario_onset_cycle_max]` using a seeded RNG (`seed + 1000`). Scenario progression is computed as `engine_cycle - engine_onset_cycle`.

**Research protocol (LOCKED):**
- `scenario_onset_cycle_min = 25`
- `scenario_onset_cycle_max = 35`
- Onset variation models realistic fleet conditions where degradation does not start synchronously

Example: With `onset_min=25, onset_max=35, seed=42`:
- Engine 1 onset = cycle 32
- Engine 2 onset = cycle 28  
- Engine 3 onset = cycle 35

### Fleet Configuration

- **Total stream units:** 24 engines (unit-disjoint split of `train_FD001.txt`, seed=42)
- **Training split (initial model):** 76% of units (76 engines from training data)
- **Stream units:** 24% of units (24 engines)
- **Candidate training units:** 18 engines (75% of stream units, unit-disjoint from validation)
- **Candidate validation units:** 6 engines (25% of stream units, unit-disjoint from training)
- **Validation enforcement:** `training_units ∩ validation_units = ∅` (required)

### Validation Logging

Every retraining event logs:
- Buffer composition (rows, units)
- Training split (rows, units)
- Validation split (rows, units)
- Unit disjointness verification
- Candidate and production MAE

All validation splits enforce unit-disjoint train/test to prevent data leakage.

### Scenario Observability

With `stream_length=2400` (100 cycles):

**Full observability:**
- **high_noise** (60 cyc): Complete scenario + post-adaptation observation
- **correlated_drift** (60 cyc): Complete scenario + post-adaptation observation
- **drift_recovery** (60 cyc): Complete scenario + **TRUE ENVIRONMENTAL RECOVERY** ← only scenario with recovery phase

**Near-complete observability:**
- **sudden_spike** (80 cyc): 88-94% of scenario visible
- **sensor_failure** (80 cyc): 88-94% of scenario visible
- **intermittent_spikes** (90 cyc): 72-83% of scenario visible

**Truncated (adaptation study):**
- **gradual_drift** (100 cyc): 65-75% of scenario visible; measures adaptation to ongoing degradation
- **concept_drift** (150 cyc): 43-50% of scenario visible; measures adaptation to observed portion

### Recovery Metrics Interpretation

**Metrics:**
- `time_to_first_error_recovery`: Samples from first MAE exceedance to first drop below threshold
- `time_to_sustained_recovery`: Samples from first exceedance to sustained MAE reduction

**IMPORTANT:** These metrics measure **error reduction after adaptation**, not environmental recovery. For scenarios without explicit recovery phases (gradual_drift, concept_drift, etc.), these metrics reflect **adaptation effectiveness** under ongoing degradation. Only **drift_recovery** measures true environmental recovery (return to baseline after degradation ends).

### Locked Research Configuration

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

**Default parameters (locked for research comparability):**
- `stream_length = 2400` (100 cycles × 24 engines)
- `scenario_onset_cycle_min = 25, scenario_onset_cycle_max = 35`
- `minimum_retraining_samples = 55`
- `minimum_validation_rows = 20, minimum_validation_units = 1`
- `shadow_window = 20, cooldown = 30`
- `performance_gate_threshold = 0.95`

Results are written to `experiments/results/raw/` (event-level CSV) and `experiments/results/aggregated/` (summary JSON).

### Final 96-Run Research Matrix: Status

The full factorial matrix — **4 strategies × 8 scenarios × 3 seeds = 96 runs** — has **already been executed successfully** (96/96 succeeded, 0 failed, 0 skipped) under the locked protocol above. The compact, final results are committed in this repository under [`experiments/results/`](experiments/results/README.md):

- `experiment_manifest.csv` — the deterministic 96-run manifest
- `aggregated_results.csv` + `aggregated_results.qc.json` — per-run metrics and QC status
- `statistical_analysis.json` / `.md` — Friedman + Wilcoxon (Holm-Bonferroni) results
- `provenance.json` — full run/software/commit provenance
- `verification_report.json` / `.md` — per-run completion verification
- `figures/*.png` — the 8 publication-quality figures

Detailed per-run outputs (96 raw event-level CSVs, 96 per-run summary JSONs, and execution logs — roughly 80MB) are **not** stored in Git. They are preserved in the external research archive (see [Where Raw Outputs Are Archived](#where-raw-outputs-are-archived) below) and are not required to inspect or reproduce the final analysis.

### Reproducing the Analysis (recommended — does not require rerunning experiments)

Because the frozen aggregated results are committed to this repository, the statistical analysis and figures can be regenerated deterministically **without re-executing any experiments**:

```bash
# Re-run QC on the committed aggregated results
python -m scripts.analysis.statistical_qc \
  --aggregated experiments/results/aggregated_results.csv \
  --manifest experiments/results/experiment_manifest.csv \
  --output experiments/results/aggregated_results.qc.json

# Re-run the Friedman + Wilcoxon (Holm-Bonferroni) statistical analysis
python -m scripts.analysis.statistical_analysis \
  --aggregated experiments/results/aggregated_results.csv \
  --manifest experiments/results/experiment_manifest.csv \
  --qc-report experiments/results/aggregated_results.qc.json \
  --output experiments/results \
  --block-def scenario_seed

# Regenerate the 8 publication figures
python -m scripts.analysis.generate_figures \
  --aggregated experiments/results/aggregated_results.csv \
  --statistical experiments/results/statistical_analysis.json \
  --output experiments/results/figures
```

These scripts are deterministic given the same input data, but exact byte-for-byte reproduction across arbitrary Python/library versions is not guaranteed — see [Environment](#locked-research-configuration) for the versions used to produce the committed results.

### Reproducing the Experiment From Scratch (optional — not required to inspect results)

The full reproducibility pipeline used to produce the committed results is:

```
generate_manifest → run_matrix → verify_completion → aggregate_results → statistical_qc → statistical_analysis → generate_figures
```

```bash
# 1. Generate the deterministic 96-run manifest
python -m scripts.matrix_orchestration.generate_manifest \
  --output-dir experiments/results

# 2. Execute the full matrix (long-running: ~4 hours; runs experiments/runner.py per manifest row)
python -m scripts.matrix_orchestration.run_matrix \
  --manifest experiments/results/experiment_manifest.csv \
  --yes

# 3. Verify all 96 runs completed with valid outputs
python -m scripts.matrix_orchestration.verify_completion \
  --manifest experiments/results/experiment_manifest.csv \
  --results-dir experiments/results \
  --strict

# 4. Aggregate raw outputs into the compact results table
python -m scripts.analysis.aggregate_results \
  --manifest experiments/results/experiment_manifest.csv \
  --results-dir experiments/results \
  --output experiments/results/aggregated_results.csv

# 5-7. Then run statistical_qc, statistical_analysis, and generate_figures as shown above.
```

**This is not necessary to inspect, cite, or verify the final results** — it is documented only for readers who want to independently re-execute the full study. Rerunning `run_matrix` will overwrite `experiments/results/raw/`, `experiments/results/aggregated/`, and `experiments/results/logs/` with new timestamped outputs; the committed compact artifacts in this repository reflect the frozen, already-completed 96-run experiment (manifest git commit `fa6cbd5571184daf2ddbebd319aaf6614c276f9b`).

### Where Raw Outputs Are Archived

The 96 raw event-level CSVs, 96 per-run summary JSONs, execution logs, and the mini-matrix validation artifacts used to bring up this pipeline are preserved outside Git in an external research archive, along with a checksum manifest. They are intentionally excluded from version control (see `.gitignore`) to keep the repository lightweight; the compact artifacts under `experiments/results/` contain everything needed to verify and cite the final results.

---

## Research Paper

**Title:** *Conservative Multi-Gated Adaptation for Physical Prognostics: Trading Prediction Accuracy for Model Stability in Non-Stationary Streams*

**Status:** Draft completed (Phase 3G, August 2026)

**Location:** `paper/RESEARCH_PAPER_DRAFT.md`

### Abstract

Adaptive model management for physical prognostics faces a fundamental trade-off: aggressive adaptation (frequent retraining) minimizes prediction error but increases model churn and operational risk, while conservative adaptation (rigorous gating) maximizes model stability but may degrade prediction accuracy. This work presents an integrated closed-loop MLOps architecture combining multi-channel statistical drift detection, autonomous background retraining, and two-stage model validation (offline performance gate + live parallel shadow evaluation) for continuous RUL regression on streaming turbofan telemetry.

A comprehensive 96-run controlled experiment (4 strategies × 8 degradation scenarios × 3 seeds, blocked factorial design) on NASA C-MAPSS FD001 demonstrates that conservative multi-gated adaptation achieves 6.4–12.6× lower model replacement frequency than aggressive baselines (1.83 vs. 11.67 vs. 23.0 promotions per 2400-cycle stream, *p* < 0.001) at the cost of 8.5–23.9% higher prediction error (MAE 10.79 vs. 9.94 vs. 8.71, *p* < 0.001) and 9.2× higher adaptation overhead than naive adaptive (46.2s vs. 5.0s, *p* < 0.001).

### Key Findings

1. **Stability–Accuracy Trade-off:** No single strategy optimizes both model stability (low promotion frequency) and prediction accuracy (low MAE/RMSE). Scheduled adaptation achieves best accuracy but maximum churn (23 promotions). Proposed conservative gating achieves minimum churn (1.83 promotions) but higher error.

2. **Domain-Specific Optimal Policies:** Safety-critical systems (aerospace, medical) may prioritize stability (proposed strategy), while offline batch prediction (maintenance scheduling) may prioritize accuracy (scheduled strategy).

3. **Two-Stage Gating Effectiveness:** The proposed strategy's offline gate rejects 83.6% of candidates (10.17 / 12.17). Shadow evaluation provides a second validation layer, though 45.5% of promotions are "degraded" (validation-production distribution mismatch).

4. **Statistical Rigor:** All five primary metrics (MAE, RMSE, detection delay, model promotions, adaptation time) show statistically significant differences (Friedman tests, *p* < 0.001). 27/30 pairwise comparisons remain significant after Holm-Bonferroni correction.

### Paper Structure

- **Section 1:** Introduction (motivation, literature gap, research questions, contributions)
- **Section 2:** Related Work (physical prognostics, drift detection, autonomic ML, shadow testing)
- **Section 3:** System Architecture (multi-channel detection, decision engine, gating, shadow evaluation)
- **Section 4:** Experimental Methodology (NASA C-MAPSS FD001, 4 strategies, 8 scenarios, 96-run factorial design)
- **Section 5:** Results (trade-off quantification, statistical significance, scenario-level analysis)
- **Section 6:** Discussion (interpretations, domain implications, degraded promotions)
- **Section 7:** Limitations (single dataset, simulated scenarios, fixed thresholds, RQ4 out of scope)
- **Section 8:** Future Work (ablation studies, catastrophic candidate injection, cross-domain generalization)
- **Section 9:** Conclusion
- **Section 10:** Acknowledgments
- **Section 11:** References
- **Appendix A:** Reproducibility Instructions
- **Appendix B:** Full Statistical Analysis Tables

### Supporting Documents

- **Results Traceability:** `paper/RESULTS_TRACEABILITY.md` — maps every numerical claim in the paper to its authoritative source artifact (aggregated_results.csv, statistical_analysis.json, provenance.json)
- **Figures (8 total):** All figures located in `experiments/results/figures/` at 300 DPI
  - mae_boxplot.png, rmse_boxplot.png
  - mae_by_scenario.png, rmse_by_scenario.png
  - strategy_comparison_barplot.png
  - adaptation_metrics_combined.png
  - effect_size_heatmap.png
  - pairwise_significance_heatmap.png

### Research Questions Addressed

| RQ | Topic | Status |
|----|-------|--------|
| RQ1 | Model adaptation strategies and RUL prediction accuracy under non-stationary degradation | ✅ Fully Answered |
| RQ2 | Candidate filtering and model-promotion behavior via offline validation + live shadow evaluation | ✅ Fully Answered |
| RQ3 | Trade-offs (accuracy vs. drift detection delay vs. promotion frequency vs. adaptation time) | ✅ Fully Answered |

**Core Contribution:** RQ3 provides the strongest empirical result—a clear, statistically validated quantification of the stability–accuracy–overhead trade-off across diverse degradation scenarios.

### Experimental Provenance

- **Original Experiment Commit:** `fa6cbd5571184daf2ddbebd319aaf6614c276f9b`
- **Dataset Checksum (MD5):** `1721c96c01e188569f0e7bb16b1ea493` (train_FD001.txt)
- **Execution Date:** August 25, 2026 (00:36–04:24 UTC+5:30)
- **Total Runs:** 96/96 successful, 0 failures, 0 skipped, 0 reruns
- **Software Versions:**
  - Python 3.12.0
  - NumPy 2.5.1
  - SciPy 1.18.0
  - pandas 3.0.3
  - scikit-learn 1.9.0
  - matplotlib 3.9.2
  - seaborn 0.13.2
  - river 0.25.0

### Reproducibility

All experiments are fully reproducible using the deterministic pipeline documented in `scripts/README.md`. The complete provenance chain (git commits, dataset checksums, software versions, random seeds) is tracked in `experiments/results/provenance.json`.

To reproduce the analysis (without rerunning experiments):
```bash
# Statistical analysis
python -m scripts.analysis.statistical_analysis \
  --aggregated experiments/results/aggregated_results.csv \
  --manifest experiments/results/experiment_manifest.csv \
  --qc experiments/results/aggregated_results.qc.json \
  --output experiments/results/statistical_analysis.json

# Figure generation
python -m scripts.analysis.generate_figures \
  --aggregated experiments/results/aggregated_results.csv \
  --statistical experiments/results/statistical_analysis.json \
  --output-dir experiments/results/figures
```

To reproduce the full 96-run experiment matrix (approximately 4 hours):
```bash
# See scripts/README.md for complete orchestration pipeline
python -m scripts.matrix_orchestration.run_matrix \
  --manifest experiments/results/experiment_manifest.csv \
  --yes
```

---

## Pipeline Deep Dive

### Drift Detection

Two independent detectors run on every processed event:

1. **ADWIN** (`drift/adwin_detector.py`) — monitors the rolling error stream. Detects **concept drift**: when the model's error distribution shifts, even if input features look normal. Uses `delta=0.002` (strict; fewer false positives).

2. **KS-test** (`drift/data_drift.py`) — runs a Kolmogorov-Smirnov test across all 24 feature columns comparing a reference window against the current window. Applies **Bonferroni correction** to control false discovery rate across multiple simultaneous tests. Drift is confirmed when the ratio of drifted features exceeds the configured `drift_feature_ratio_threshold` — **0.12** in the locked research-experiment configuration (`experiments/config.py`); the interactive dashboard (`api_server.py`) uses a separate, more sensitive default of 0.08.

### Self-Healing Pipeline

When both drift detectors agree and the rolling MAE exceeds threshold, the cooldown timer elapses, and the system:

1. **Spawns a background retrain thread** on a snapshot of the recent buffer (≥55 events).
2. Validates that the buffer meets the **candidate validation-quality policy** — the disjoint validation set must have at least `minimum_validation_rows` rows and `minimum_validation_units` distinct engine units. If not met, the trigger is recorded as skipped and monitoring continues.
3. Trains a candidate model via `train_model_with_holdout` with a unit-based train/validation split and the experiment's configured random seed.
4. Passes the candidate through the **Performance Gate** — it must achieve ≥5% lower MAE than production on held-out data.
5. Starts **Shadow Evaluation** — the candidate runs in parallel with production for a 20-event live shadow window (`shadow_window = 20`, counted in streaming events/predictions, not cycles).
6. If shadow MAE < production MAE × 0.95 after those 20 events, the candidate is **promoted** and the Isolation Forest is refitted.

### Rate Limiting

The `RateLimitController` maintains a 500-event FIFO queue and dynamically adjusts the processing rate each tick:

| State | Condition | Behaviour |
|---|---|---|
| **Nominal** | No backlog, no drift/retrain | Process at operator-configured ceiling |
| **Draining** | Backlog > 5 events | Process at full ceiling to clear queue |
| **Protecting** | Drift detected or retraining active | Reduce to 60% of ceiling |
| **Throttling** | Incoming rate > applied limit | Flag for audit; queue builds |
| **Bypassed** | Rate limiting disabled | Process at hardware capacity (~40 eps) |

Load shedding: when the queue reaches capacity (500), the oldest event is discarded. An audit entry is written on the first drop and every 100 drops thereafter.

---

## Dataset

**NASA CMAPSS FD001** — 100 turbofan engine units, single operating condition, single fault mode.

- 21 sensor readings per cycle + 3 operational settings
- RUL is computed as `max_cycle_for_unit - current_cycle` (clipped to 125)
- 76% of units used for initial training; 24% reserved for the live stream
- Baseline statistics (mean/std per feature) are computed from the training split and used for drift scoring and scenario scaling
- Both the live dashboard (`api_server.py`) and the research experiment framework (`experiments/`) load only `dataset/raw/train_FD001.txt` and partition its 100 units into initial-training and streaming/monitoring units (unit-disjoint). The NASA `test_FD001.txt` / `RUL_FD001.txt` split is **not** used by the research matrix or the live stream.

**Dataset provenance:** See [`dataset/PROVENANCE.md`](dataset/PROVENANCE.md) for complete dataset documentation including source, citation, checksums, and usage information.
