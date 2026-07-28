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
│           Simulated from NASA CMAPSS FD001 test split               │
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
| Shadow A/B evaluation | Candidate runs in parallel for 20 cycles before promotion |
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
- pytest with 74 tests across all modules

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
│   ├── train.py                # train_model_with_holdout (unit-based split)
│   ├── predict.py              # Single-point inference helper
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
│   ├── sudden_spike.py         # Instant ±8σ spike on 3 sensors
│   ├── high_noise.py           # 3× baseline noise injection
│   ├── sensor_failure.py       # Flat-line failure on sensor_7 and sensor_11
│   ├── concept_drift.py        # RUL label shift (−60 cycles, features stable)
│   ├── correlated_drift.py     # Correlated shift across sensor pairs
│   ├── intermittent_spikes.py  # ±12σ spikes every 7th cycle
│   └── drift_recovery.py       # Drift then gradual recovery
│
├── dataset/
│   ├── raw/
│   │   └── train_FD001.txt     # NASA CMAPSS dataset (100 engine units)
│   └── processed/
│       └── preprocess_module.py
│
├── tests/                      # pytest suite (74 tests)
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
│   └── test_train.py
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
| `sudden_spike` | Sudden Sensor Drift | Critical | 45 cycles | All 21 sensors shift by +8σ immediately — simulates a severe environmental or mechanical fault |
| `high_noise` | High Sensor Noise | Medium | 60 cycles | Sensor variance increases 5× with no mean shift — simulates electrical interference |
| `sensor_failure` | Stuck Sensor Failure | High | 80 cycles | Two sensors lock to 0.0 — simulates disconnected or grounded hardware |
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

Run all 74 tests with:

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
| `test_train.py` | 4 | Normal training, insufficient-data guard, single-unit fallback, model inference |

---

## Pipeline Deep Dive

### Drift Detection

Two independent detectors run on every processed event:

1. **ADWIN** (`drift/adwin_detector.py`) — monitors the rolling error stream. Detects **concept drift**: when the model's error distribution shifts, even if input features look normal. Uses `delta=0.002` (strict; fewer false positives).

2. **KS-test** (`drift/data_drift.py`) — runs a Kolmogorov-Smirnov test across all 24 feature columns comparing a reference window against the current window. Applies **Bonferroni correction** to control false discovery rate across multiple simultaneous tests. Drift is confirmed when ≥45% of tested features show significant shift.

### Self-Healing Pipeline

When both drift detectors agree and the rolling MAE exceeds threshold, the cooldown timer elapses, and the system:

1. **Spawns a background retrain thread** on a snapshot of the recent buffer (≥55 events).
2. Trains a candidate model via `train_model_with_holdout` with a unit-based train/validation split.
3. Passes the candidate through the **Performance Gate** — it must achieve ≥5% lower MAE than production on held-out data.
4. Starts **Shadow Evaluation** — the candidate runs in parallel with production for 20 live cycles.
5. If shadow MAE < production MAE × 0.95 after 20 cycles, the candidate is **promoted** and the Isolation Forest is refitted.

### Rate Limiting

The `RateLimitController` maintains a 500-event FIFO queue and dynamically adjusts the processing rate each tick:

| State | Condition | Behaviour |
|---|---|---|
| **Nominal** | No backlog, no drift/retrain | Process at operator-configured ceiling |
| **Draining** | Backlog > 5 events | Process at full ceiling to clear queue |
| **Protecting** | Drift detected or retraining active | Reduce to 60% of ceiling |
| **Throttling** | Incoming rate > applied limit | Flag for audit; queue builds |
| **Bypassed** | Rate limiting disabled | Process at hardware capacity (40 eps) |

Load shedding: when the queue reaches capacity (500), the oldest event is discarded. An audit entry is written on the first drop and every 100 drops thereafter.

---

## Dataset

**NASA CMAPSS FD001** — 100 turbofan engine units, single operating condition, single fault mode.

- 21 sensor readings per cycle + 3 operational settings
- RUL is computed as `max_cycle_for_unit - current_cycle` (clipped to 125)
- 76% of units used for initial training; 24% reserved for the live stream
- Baseline statistics (mean/std per feature) are computed from the training split and used for drift scoring and scenario scaling
