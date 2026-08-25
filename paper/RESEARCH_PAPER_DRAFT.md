# Conservative Multi-Gated Adaptation for Physical Prognostics: Trading Prediction Accuracy for Model Stability in Non-Stationary Streams

**Authors:** [To be determined]  
**Affiliation:** [To be determined]  
**Date:** August 2026

---

## Abstract

Adaptive model management for physical prognostics faces a critical but underexplored trade-off: aggressive adaptation (frequent retraining) minimizes prediction error but increases model churn and operational risk, while conservative adaptation (rigorous gating) maximizes model stability but may degrade prediction accuracy. Existing work fragments this problem across three domains—physical prognostics focuses on RUL models without adaptation strategies, drift detection research emphasizes statistical methods without safety gating, and software engineering explores canary deployment without targeting physical streams. We present an integrated closed-loop MLOps architecture combining multi-channel statistical drift detection (ADWIN concept drift, KS-test feature drift with Bonferroni correction, Isolation Forest anomaly detection), autonomous background retraining, and two-stage model validation (offline performance gate plus live parallel shadow evaluation) specifically for continuous remaining useful life (RUL) regression on streaming turbofan telemetry. A comprehensive 96-run controlled experiment (4 strategies × 8 degradation scenarios × 3 seeds, blocked factorial design) on NASA C-MAPSS FD001 reveals that conservative multi-gated adaptation achieves 6.4–12.6× lower model replacement frequency than aggressive baselines (1.83 vs. 11.67 vs. 23.0 promotions per 2400-cycle stream, *p* < 0.001, Friedman test) at the cost of 8.5–23.9% higher prediction error (MAE 10.79 vs. 9.94 vs. 8.71, *p* < 0.001) and 9.2× higher adaptation overhead than naive adaptive (46.2s vs. 5.0s, *p* < 0.001). This trade-off is statistically significant in the blocked 96-run design spanning all eight degradation scenarios (concept drift, correlated drift, sensor failures, noise spikes), suggesting the optimal adaptation policy is domain-specific rather than universally optimal. The results indicate that applications prioritizing model stability may prefer conservative adaptation, whereas applications prioritizing prediction accuracy may prefer more aggressive adaptation. We provide a deterministic reproducibility pipeline with provenance tracking, enabling exact replication of all 96 experimental runs.

**Keywords:** Physical prognostics, remaining useful life, concept drift, model adaptation, shadow evaluation, performance gating, self-healing ML, NASA C-MAPSS

---

## 1. Introduction

### 1.1 Motivation

Physical prognostics—the prediction of remaining useful life (RUL) for safety-critical components such as turbofan engines, batteries, and industrial machinery—requires machine learning models that continuously process streaming sensor telemetry under non-stationary operating conditions. Sensor drift, component degradation, environmental changes, and operational regime shifts cause the statistical properties of input features and target relationships to evolve over time, leading to concept drift that degrades model accuracy if left unaddressed.

Adaptive model management responds to drift by autonomously detecting degradation, retraining candidate models, and promoting improved models to production. However, adaptation introduces operational risk: frequent model replacement destabilizes production systems, complicates auditability, and increases deployment overhead, while infrequent replacement allows prediction accuracy to degrade under persistent drift.

This fundamental tension between **model stability** (minimizing replacement frequency) and **prediction accuracy** (minimizing error under drift) has been underexplored in the physical prognostics literature. Existing work tends to assume that "more adaptation is better," focusing on drift detection sensitivity and retraining speed while overlooking the operational costs of model churn.

### 1.2 Literature Gap

Existing research has separately investigated RUL prediction under industrial sensor streams [2, 3, 12, 17, 22], statistical drift detection with adaptive retraining [14, 18, 23], and deployment-time mechanisms such as shadow or canary evaluation [5, 7, 11, 20]. However, the reviewed literature provides limited systematic evaluation of these mechanisms as an integrated, autonomous model-lifecycle for non-stationary physical prognostics, particularly with explicit validation and live shadow gating before model promotion.

### 1.3 Research Questions

We address three research questions:

**RQ1:** How do different model adaptation strategies affect RUL prediction accuracy under diverse non-stationary degradation scenarios?

**RQ2:** How does combining offline validation and live shadow evaluation affect candidate filtering and model-promotion behavior under streaming drift compared with trigger-and-replace adaptation?

**RQ3:** What trade-offs arise between prediction accuracy, drift detection delay, model-promotion frequency, and adaptation time across different adaptation strategies?

### 1.4 Contributions

This work makes three contributions:

1. **Integrated Architecture:** A closed-loop MLOps architecture synthesizing multi-channel statistical drift detection, autonomous background retraining, and two-stage model promotion (offline performance gate + live shadow evaluation) for physical RUL regression on streaming telemetry.

2. **Two-Stage Gating Protocol:** A methodological contribution combining offline validation (candidate must outperform production on held-out data by ≥5%) with live shadow evaluation (candidate must outperform production on 20 parallel predictions) to filter candidates that fail offline or live-stream performance checks before promotion.

3. **Trade-off Quantification:** Empirical demonstration that conservative multi-gated adaptation reduces model replacement frequency by 6.4–12.6× at the cost of 8.5–23.9% higher MAE and 9.2× higher adaptation overhead than naive adaptive (and 6.9× higher than scheduled), revealing a fundamental stability–accuracy trade-off validated across 8 degradation scenarios.

### 1.5 Paper Organization

Section 2 reviews related work. Section 3 describes the system architecture. Section 4 details experimental methodology. Section 5 presents results. Section 6 discusses interpretations and implications. Section 7 acknowledges limitations. Section 8 outlines future work. Section 9 concludes.

---

## 2. Related Work

### 2.1 Physical Prognostics and RUL Prediction

Physical prognostics focuses on predicting remaining useful life (RUL) for safety-critical components using sensor telemetry [2, 3, 12, 17, 22]. The NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset has become a standard benchmark, providing simulated turbofan engine degradation data with multiple operating conditions and fault modes [2, 3, 12, 17, 22]. Traditional approaches use physics-based degradation models, while modern methods employ machine learning regressors (Random Forest, gradient boosting, neural networks) trained on historical run-to-failure trajectories [2, 3, 12, 17, 22].

However, physical prognostics literature predominantly treats models as **static artifacts** deployed once and replaced manually when accuracy degrades [2, 3, 12, 17, 22]. Adaptation strategies, when discussed, typically involve periodic retraining on fixed schedules or ad-hoc manual intervention when operators observe prediction errors. The operational challenges of autonomous model replacement—including validation under non-stationary streams, promotion safety, and the stability–accuracy trade-off—remain underexplored.

### 2.2 Statistical Drift Detection

Concept drift detection has been extensively studied in the streaming machine learning literature [14, 18, 23]. Methods fall into three categories:

1. **Error-based detection:** Monitor prediction error distributions (e.g., ADWIN for adaptive windowing over error rates) to detect shifts in model performance.

2. **Feature drift detection:** Apply statistical tests (KS-test, chi-square, Kolmogorov-Smirnov) to compare feature distributions between historical and current windows, often with Bonferroni or Benjamini-Hochberg correction for multiple testing [14, 18, 23].

3. **Model-based detection:** Track changes in model parameters, decision boundaries, or ensemble weights to infer distribution shifts.

These methods are typically evaluated on synthetic streams with known drift points (sudden, gradual, recurring) or single-domain benchmarks (spam detection, network intrusion) [14, 18, 23]. Evaluation focuses on detection latency, false positive rates, and parameter sensitivity, but rarely addresses **what to do after detection**—specifically, how to safely replace production models when both training data and validation data are non-stationary.

### 2.3 Autonomic Computing and Self-Healing ML

Autonomic computing, inspired by biological self-regulation, proposes closed-loop MAPE-K architectures (Monitor, Analyze, Plan, Execute, Knowledge) for self-managing systems [1, 4, 6, 8, 9, 11, 13, 15, 16, 19, 20, 21]. In ML contexts, this translates to:

- **Monitor:** Collect prediction errors, feature distributions, and system metrics
- **Analyze:** Apply drift detection, anomaly detection, and performance degradation analysis
- **Plan:** Decide whether to retrain, which data to use, and when to promote
- **Execute:** Trigger background retraining and model promotion
- **Knowledge:** Maintain historical performance, drift patterns, and adaptation outcomes

However, autonomic ML research often emphasizes **adaptation aggressiveness** (faster detection, immediate replacement) without rigorously analyzing the **operational costs** of model churn in safety-critical physical systems [4, 6, 7, 8, 9, 11, 13, 15, 16, 19, 20]. The implicit assumption that "more adaptation is better" overlooks scenarios where model stability (auditability, certification, trust) outweighs marginal accuracy gains.

### 2.4 Shadow Testing and Canary Deployment

Software engineering has developed mature practices for safely deploying new code versions [5, 7, 11, 20]:

- **Shadow testing:** Run new and old versions in parallel on live traffic, compare results, promote only if new version performs better
- **Canary deployment:** Gradually roll out new version to increasing percentages of traffic, monitoring for regressions
- **Blue-green deployment:** Maintain two production environments, switch atomically after validation

These practices are standard in web services (search ranking, recommendation systems, ad serving) but have seen limited application to physical prognostics [2, 3, 12, 17, 22], where:

1. **Non-stationary validation data:** Unlike web services where A/B tests assume i.i.d. traffic, physical streams undergo continuous distribution shift, invalidating offline validation metrics during shadow testing.

2. **Sample efficiency:** Physical systems may produce only 10–100 samples per hour, making statistical validation slower than web-scale A/B tests with millions of samples [2, 3, 12, 17, 22].

3. **Safety criticality:** Incorrect RUL predictions can lead to catastrophic failures (unplanned engine shutdown) or unnecessary maintenance (false alarms), requiring higher confidence thresholds than click-through rate optimization [2, 3, 12, 17, 22].

### 2.5 Gap Synthesis

The literature contains:

- Physical prognostics models **without autonomous adaptation strategies**
- Statistical drift detection **without safety-critical promotion protocols**
- Shadow testing **without application to non-stationary physical streams**

The reviewed literature does not provide a systematic evaluation of an integrated architecture combining multi-channel statistical drift detection, autonomous background retraining, offline performance gating, and live parallel shadow evaluation for continuous RUL regression on streaming physical telemetry. Furthermore, the **stability–accuracy trade-off** inherent in choosing between aggressive and conservative adaptation policies has not been empirically quantified across diverse degradation scenarios.

---

## 3. System Architecture

Our system integrates five subsystems into a closed-loop adaptive architecture for streaming RUL prediction.

### 3.1 Multi-Channel Drift Detection

We employ three complementary drift detection methods to monitor different aspects of stream non-stationarity:

**1. Concept Drift Detection (ADWIN):**  
The Adaptive Windowing (ADWIN) algorithm maintains a variable-size sliding window over prediction errors, automatically shrinking when distribution shifts are detected. ADWIN uses Hoeffding bounds to determine when two sub-windows have statistically different error means, signaling concept drift (change in the feature-target relationship).

**2. Feature Drift Detection (KS-Test with Bonferroni Correction):**  
For each of 14 sensor features, we apply the two-sample Kolmogorov-Smirnov test comparing the current window (most recent 20 samples) to a reference window (historical baseline). To control family-wise error rate across multiple features, we apply Bonferroni correction: reject null hypothesis only if *p* < α/14 where α = 0.05. Additionally, we require a minimum effect size (Cohen's *d* ≥ 0.08) and a threshold fraction of drifted features (≥12%) before triggering adaptation.

**3. Anomaly Detection (Isolation Forest):**  
An Isolation Forest trained on historical data scores each incoming sample for anomaly likelihood. Persistent anomalies (e.g., sensor failures, outlier operating conditions) trigger error monitoring escalation even if ADWIN and KS-test do not yet detect statistically significant drift.

All three detectors feed into a unified decision engine that aggregates signals and manages adaptation state transitions.

### 3.2 Decision Engine (Six-State FSM)

The decision engine implements a finite state machine with six states:

1. **STABLE:** Nominal operation, no drift detected
2. **WATCH:** Weak drift signals detected, begin monitoring
3. **MONITOR:** Multiple drift indicators active, elevated alertness
4. **ALERT:** Drift confirmed, candidate retraining recommended
5. **RETRAIN:** Active background retraining in progress
6. **RETRAIN_URGENT:** Critical drift requiring immediate adaptation

State transitions are governed by thresholds on drift score (weighted combination of ADWIN, KS-test, and anomaly signals), rolling prediction error, and error trend. The engine enforces an adaptive cooldown period (minimum 20–50 cycles between retraining events) to prevent thrashing under noisy drift signals.

### 3.3 Autonomous Background Retraining

When the decision engine triggers retraining:

1. **Validation Buffer Construction:** Extract most recent data meeting quality criteria (minimum 55 samples, minimum 1 unique engine unit, 75% train / 25% validation split within retraining buffer).

2. **Background Candidate Training:** Train a Random Forest regressor (100 trees, max depth 20) on training partition without blocking the live prediction stream.

3. **Training Time Measurement:** Record wall-clock time for reproducibility and overhead analysis.

4. **Candidate Registration:** Assign unique candidate ID for audit trail.

Retraining is **non-blocking**: live predictions continue using the current production model while the candidate trains in the background.

### 3.4 Offline Performance Gate

Before entering shadow evaluation, candidates must pass an offline validation gate:

**Gate Criterion:** Candidate MAE must be at least 5% lower than production MAE on a common held-out validation set (20 samples drawn from recent stream history).

**Rationale:** Prevents candidates with marginal or negative improvement from consuming shadow evaluation resources. The 5% threshold balances sensitivity (detecting meaningful improvements) with specificity (rejecting noise-driven candidates).

**Gate Rejection Reasons:**
- Candidate validation failed (model crashed or returned invalid predictions)
- Production validation failed (production model no longer valid on recent data)
- Insufficient improvement (<5% MAE reduction)

Candidates failing the gate are discarded; the production model remains active.

### 3.5 Live Shadow Evaluation

Candidates passing the offline gate enter parallel shadow testing:

**Shadow Protocol:**
1. For each incoming sample, run **both** production and candidate models
2. Store prediction errors in fixed-size windows (20 samples each)
3. After collecting 20 parallel predictions, compute MAE for both models
4. **Promotion Criterion:** Promote candidate if shadow MAE < production MAE × 0.95 (i.e., candidate is at least 5% better on live stream)

**Shadow Rejection:** Candidates performing worse than production during shadow testing are discarded. This is intended to filter candidates that perform worse than the current production model during the shadow evaluation window.

**Duration:** Shadow evaluation typically requires 20 samples, introducing latency of 20–100 cycles depending on stream rate and scenario dynamics.

### 3.6 Model Promotion and Audit Trail

Candidates passing both gates are promoted to production:

1. Replace production model and scaler with candidate artifacts
2. Increment model version counter
3. Log promotion event with:
   - Timestamp and sample index
   - Offline gate metrics (production MAE, candidate MAE, improvement)
   - Shadow metrics (production shadow MAE, candidate shadow MAE)
   - Training time and shadow evaluation time
   - Candidate ID for traceability

**Degraded Promotion Detection:** A promotion is flagged as "degraded" if shadow MAE > offline validation MAE, indicating the candidate performed worse on the live stream than on the validation buffer. This signals validation-production distribution mismatch but does **not** mean the candidate was worse than production (both gates still required candidate > production).

---

## 4. Experimental Methodology

### 4.1 Dataset: NASA C-MAPSS FD001

We use the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine degradation dataset, subset FD001:

- **Domain:** Simulated turbofan engine run-to-failure trajectories
- **Training Set:** 100 engines, 20,631 cycles total
- **Test Set:** 100 engines, 13,096 cycles total  
- **Features:** 21 sensor measurements (14 informative after removing constants)
- **Target:** Remaining Useful Life (RUL) in cycles
- **Operating Conditions:** Single operating condition (sea level)
- **Fault Modes:** Single fault mode (High-Pressure Compressor degradation)
- **Dataset Checksum (MD5):** `1721c96c01e188569f0e7bb16b1ea493` (train_FD001.txt)

This dataset is widely used in physical prognostics research, enabling comparison with prior work [2, 3, 12, 17, 22].

**Train/Stream Split:** The 100-engine training set is split into initial training units (76%, ~76 engines) and streaming units (24%, ~24 engines). The initial training units are used to train the baseline model deployed at stream start; the streaming units form the live evaluation stream.

### 4.2 Stream Construction: Interleaved Fleet Monitoring

Rather than sequential single-engine streams, we construct **interleaved multi-engine streams** simulating real-world fleet monitoring:

1. **Stream Length:** 2400 samples per experimental run
2. **Stream Mode:** Interleaved—round-robin sampling from multiple engines to simulate parallel fleet monitoring
3. **Engine Selection:** Randomly sample 10–15 engines from the 24% streaming unit pool, interleave their lifecycles
4. **Scenario Injection:** Apply degradation scenarios (drift, noise, sensor failures) starting at randomly selected cycle within [25, 35] to simulate non-stationary conditions

This design tests adaptation under realistic multi-asset monitoring where the model must generalize across engines at different degradation stages.

### 4.3 Degradation Scenarios (8 Scenarios)

We evaluate eight synthetic degradation scenarios representing common non-stationarity patterns in physical systems:

1. **gradual_drift:** Slow additive drift (+0.05σ per cycle, 50 cycles duration)
2. **sudden_spike:** Abrupt shift (+3σ, 30 cycles duration)
3. **high_noise:** Increased measurement noise (3× standard deviation, 50 cycles)
4. **sensor_failure:** Zeroing of 3 key sensor channels (30 cycles)
5. **concept_drift:** Multiplicative scaling of target relationship (1.4×, 40 cycles)
6. **correlated_drift:** Correlated shifts across multiple features (0.12σ, 40 cycles)
7. **intermittent_spikes:** Recurring anomaly bursts (±2.5σ, 5-cycle bursts every 15 cycles, 45 cycles total)
8. **drift_recovery:** Severe drift followed by gradual return to baseline (+6σ → 0, 60 cycles)

Each scenario tests different adaptation challenges: detection latency (gradual vs. sudden), multi-channel coordination (correlated drift), robustness to anomalies (sensor failure, spikes), and recovery dynamics (drift recovery).

### 4.4 Adaptation Strategies (4 Strategies)

We compare four adaptation strategies spanning the spectrum from non-adaptive to conservative:

**1. Static (Baseline):**  
No adaptation. The initial model trained on historical data runs for the entire stream. Represents worst-case drift degradation.

**2. Scheduled (Aggressive Periodic):**  
Retrains every 100 cycles regardless of drift signals. Immediate model replacement without gating. Represents aggressive scheduled adaptation.

**3. Naive Adaptive (Reactive Immediate):**  
Retrains on drift detection. Immediate model replacement without gating. Represents aggressive reactive adaptation.

**4. Proposed (Conservative Multi-Gated):**  
Retrains on drift detection. Requires offline performance gate (≥5% improvement) **and** live shadow evaluation (≥5% improvement over 20 parallel predictions) before promotion. Represents conservative adaptation prioritizing stability.

### 4.5 Experimental Design (4 × 8 × 3 = 96 Runs)

**Full Factorial Design:**
- 4 strategies
- 8 scenarios  
- 3 random seeds (42, 123, 456)
- **Total:** 96 experimental runs

**Blocking:** Scenario × seed combinations form 24 matched blocks. Each block contains all 4 strategies experiencing identical scenario onset timing, engine selection, and stream composition. This enables paired statistical tests controlling for confounding variables.

**Locked Protocol Parameters:**
- Stream length: 2400 samples
- Stream mode: interleaved
- Scenario onset: uniform random within cycles [25, 35]
- Train fraction: 0.76
- Validation fraction: 0.25
- Retraining interval (scheduled): 100 cycles
- Cooldown period: 30 cycles
- Performance gate threshold: 0.95 (5% improvement required)
- Shadow window: 20 samples
- Random Forest: 100 trees, max depth 20, min samples split 5

### 4.6 Metrics

**Primary Metrics:**
- **MAE (Mean Absolute Error):** Primary accuracy metric for RUL prediction
- **RMSE (Root Mean Squared Error):** Secondary accuracy metric, penalizes large errors
- **Model Promotions:** Count of model replacements (stability proxy)
- **Total Adaptation Time:** Sum of retraining and shadow evaluation time (overhead)

**Secondary Metrics:**
- Detection delay: Cycles from scenario onset to first drift detection
- Drift detections: Count of drift triggers (ADWIN + KS-test + anomaly)
- Candidates generated: Count of background retraining attempts
- Gate accepts/rejects: Offline validation outcomes
- Shadow promotions/rejections: Live shadow evaluation outcomes
- Degraded promotions: Promotions where shadow MAE > validation MAE

**Recovery Metrics:**
- Time to first error recovery: Cycles until MAE drops below pre-scenario baseline
- Time to sustained recovery: Cycles until MAE remains below baseline for 10 consecutive cycles

### 4.7 Statistical Analysis

**Omnibus Test:**  
Friedman test (non-parametric repeated-measures ANOVA) for each metric, treating scenario×seed blocks as repeated measures and strategies as conditions. Null hypothesis: All strategies have identical distributions.

**Pairwise Comparisons:**  
Wilcoxon signed-rank test for all ${4 \choose 2} = 6$ strategy pairs per metric (30 total pairwise tests across 5 primary metrics). Effect sizes computed as rank-biserial correlation *r*.

**Multiple Testing Correction:**  
Holm-Bonferroni sequential correction applied to pairwise p-values within each metric. Corrected significance threshold: reject null hypothesis if *p* < α' where α' is adjusted based on rank order of p-values.

**Significance Level:** α = 0.05

### 4.8 Reproducibility Infrastructure

All experiments are orchestrated through a deterministic pipeline:

1. **Manifest Generation:** Deterministic enumeration of 96 runs with locked random seeds
2. **Matrix Orchestration:** Sequential execution with subprocess isolation
3. **Completion Verification:** Schema validation, file existence checks, metric bounds validation
4. **Aggregation:** Parse raw CSV events into summary metrics
5. **Statistical QC:** Validate metric distributions, detect outliers
6. **Statistical Analysis:** Friedman + Wilcoxon + Holm correction
7. **Figure Generation:** Automated plotting with fixed DPI and style

**Provenance Tracking:**
- Git commit hash: `fa6cbd5` (experiment execution)
- Dataset checksum: `1721c96c01e188569f0e7bb16b1ea493`
- Software versions: Python 3.12.0, NumPy 2.5.1, SciPy 1.18.0, pandas 3.0.3, scikit-learn 1.9.0, matplotlib 3.9.2, seaborn 0.13.2, river 0.25.0
- Execution timestamps: All runs completed August 25, 2026 between 00:36 and 04:24 UTC+5:30

---

## 5. Results

All 96 experimental runs completed successfully with 0 failures, 0 skipped runs, and 0 reruns after the final execution. Statistical analysis confirms all distributions meet quality criteria (no errors, no warnings).

### 5.1 RQ3: Trade-off Between Accuracy, Stability, and Overhead (Main Result)

Table 1 summarizes strategy performance across all 96 runs (mean ± std):

| Strategy | MAE | RMSE | Promotions | Adaptation Time (s) |
|----------|-----|------|------------|---------------------|
| static | 42.814 ± 21.845 | 51.773 ± 25.532 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| scheduled | **8.708 ± 1.424** | **14.272 ± 1.953** | 23.000 ± 0.000 | 6.688 ± 2.261 |
| naive_adaptive | 9.941 ± 1.877 | 15.363 ± 2.325 | 11.667 ± 3.017 | 5.037 ± 1.341 |
| proposed | 10.794 ± 2.309 | 19.280 ± 2.670 | **1.833 ± 1.049** | 46.163 ± 26.391 |

**Table 1:** Strategy performance summary. Bold indicates best value within adaptive strategies. Static is excluded from "best" comparison as it represents non-adaptive baseline.

#### 5.1.1 Finding 1: Scheduled Achieves Lowest Prediction Error

Scheduled adaptation produces the lowest MAE (8.708) and RMSE (14.272) among all strategies, outperforming naive adaptive by 1.23 MAE points (12.4% improvement) and proposed by 2.09 MAE points (19.3% improvement).

**Statistical Significance:** Friedman test for MAE: χ²(3) = 68.55, *p* < 0.001. Pairwise Wilcoxon tests (Holm-corrected):
- scheduled vs. naive_adaptive: *p* < 0.001, *r* = 0.987 (very large effect)
- scheduled vs. proposed: *p* < 0.001, *r* = 1.000 (very large effect)
- naive_adaptive vs. proposed: *p* < 0.001, *r* = -0.973 (very large effect)

**Interpretation:** Proactive retraining every 100 cycles ensures models stay fresh under drift, minimizing error accumulation. However, this comes at the cost of maximum model churn (23 promotions per run).

#### 5.1.2 Finding 2: Proposed Minimizes Model Replacement Frequency

Proposed adaptation achieves dramatically lower promotion counts: 1.83 promotions per 2400-cycle stream, compared to 11.67 for naive adaptive (6.4× reduction) and 23.0 for scheduled (12.6× reduction).

**Statistical Significance:** Friedman test for model_promoted_events: χ²(3) = 72.00, *p* < 0.001. All 6 pairwise comparisons are statistically significant (*p* < 0.001) with effect sizes *r* = 1.0 (maximum).

**Mechanism:** The two-stage gating protocol filters candidates at two levels. The offline gate rejects 83.6% of generated candidates (10.17 out of 12.17). Across both stages (offline gate + shadow evaluation), only 1.83 of 12.17 generated candidates are ultimately promoted, meaning 84.9% are filtered before promotion:
- Candidates generated: 12.17 (mean)
- Offline gate rejects: 10.17 (83.6% rejection rate)
- Offline gate accepts: 2.00 (candidates entering shadow)
- Shadow promotions: 1.83 (mean)
- Shadow rejections: 0.17 (mean)
- Overall filtering rate: 84.9% (before promotion)

**Interpretation:** Rigorous validation prevents premature or marginal model replacements, prioritizing stability over aggressive adaptation.

#### 5.1.3 Finding 3: Proposed Incurs Highest Adaptation Overhead

Proposed adaptation requires 46.2 seconds mean adaptation time, 9.2× higher than naive adaptive (5.0s) and 6.9× higher than scheduled (6.7s).

**Statistical Significance:** Friedman test for total_adaptation_time: χ²(3) = 65.60, *p* < 0.001. Pairwise Wilcoxon (Holm-corrected):
- proposed vs. naive_adaptive: *p* < 0.001, *r* = -1.000
- proposed vs. scheduled: *p* < 0.001, *r* = 1.000

**Breakdown:**
- Mean retraining time: ~5–6 seconds (similar across all strategies)
- Mean shadow evaluation time (proposed only): 21–105 seconds (dominates overhead)

**Mechanism:** Shadow evaluation requires collecting 20 parallel predictions before making a promotion decision. In scenarios with sparse drift or slow retraining triggers, shadow windows can span 50–100 cycles, dramatically increasing total adaptation time.

**Interpretation:** Conservative gating trades computational efficiency for safety. In resource-constrained edge deployments, this overhead may be prohibitive.

#### 5.1.4 Finding 4: The Stability–Accuracy–Overhead Trade-off

Figure 1 visualizes the three-way trade-off:

```
AGGRESSIVE ADAPTATION (scheduled)
  ✓ Lowest MAE/RMSE (8.71 / 14.27)
  ✗ Highest model churn (23 promotions)
  ~ Moderate overhead (6.69s)

REACTIVE ADAPTATION (naive_adaptive)
  ~ Moderate MAE/RMSE (9.94 / 15.36)
  ✗ High model churn (11.67 promotions)
  ✓ Low overhead (5.04s)

CONSERVATIVE ADAPTATION (proposed)
  ✗ Highest MAE/RMSE among adaptive (10.79 / 19.28)
  ✓ Lowest model churn (1.83 promotions)
  ✗ Highest overhead (46.16s)
```

**Key Insight:** No strategy dominates all three dimensions. The optimal choice depends on domain priorities:
- **Applications prioritizing stability** (e.g., model audit/certification requirements): proposed (stability > accuracy)
- **Applications prioritizing accuracy** (e.g., offline batch prediction): scheduled (accuracy > churn)
- **Applications prioritizing efficiency** (e.g., resource-constrained edge devices): naive_adaptive (efficiency > both)

### 5.2 Scenario-Level Analysis

Figure 2 shows MAE distributions by scenario (boxplots). Key observations:

- **sensor_failure:** All adaptive strategies perform similarly (MAE ≈ 8.5–9.5), suggesting sensor failures are easy to detect and recover from regardless of gating rigor.
- **concept_drift:** Largest performance gap (scheduled 11.5 vs. proposed 16.7), indicating concept drift benefits from aggressive adaptation.
- **drift_recovery:** Proposed performs competitively with scheduled (9.0 vs. 8.7), suggesting recovery from transient drift does not require frequent model replacement.

Figure 3 shows promotions by scenario:
- **gradual_drift:** Proposed promotes 1–2 times, scheduled 23 times (invariant across scenarios), naive_adaptive 9–14 times
- **high_noise:** Noisy scenarios trigger more retraining in naive_adaptive (12 promotions) but gating in proposed filters most out (1–2 promotions)

### 5.3 Supporting Analysis: Drift Detection Behavior

Table 2 summarizes drift detection activity:

| Strategy | Drift Detections (mean) | Anomaly Detections (mean) | Detection Delay (cycles, mean) |
|----------|-------------------------|---------------------------|--------------------------------|
| scheduled | 14.0 | 850 | 102.0 |
| naive_adaptive | 12.3 | 850 | 272.7 |
| proposed | 13.0 | 850 | 272.7 |

**Interpretation:**
- All adaptive strategies detect drift successfully (12–14 detections per 2400-cycle stream)
- Anomaly detections are high (mean 850) across noisy scenarios (high_noise, sudden_spike, sensor_failure)
- Detection delay is lower for scheduled (102 cycles) because it retrains on fixed schedule regardless of drift; naive/proposed have variable delay (272.7 cycles mean) reflecting scenario-dependent onset timing

**Limitation:** We do not have ground-truth drift labels, so cannot compute detection precision/recall. This experiment demonstrates architectural integration and operational feasibility, not statistical detection quality.

### 5.4 RQ2: Two-Stage Gating Effectiveness (Supporting Result)

Table 3 analyzes gating behavior in proposed strategy:

| Metric | Mean ± Std |
|--------|------------|
| Candidates generated | 12.17 ± 2.68 |
| Gate accepts | 2.00 ± 1.22 |
| Gate rejects | 10.17 ± 2.16 |
| Shadow promotions | 1.83 ± 1.05 |
| Shadow rejections | 0.17 ± 0.48 |
| Degraded promotions (total across 96 runs) | 20 |
| Degraded promotion rate | 45.5% (20/44) |

**Interpretation:**

1. **Offline gate is the primary filter:** 83.6% of candidates rejected at gate (10.17 / 12.17)

2. **Shadow evaluation provides additional filtering:** Shadow rejection rate is 8.5% of gate-accepted candidates (0.17 out of 2.00), indicating that a small fraction of candidates that pass offline validation are filtered during live shadow evaluation

3. **Two-stage gating demonstrates promotion filtering:** The proposed strategy achieves lower model replacement frequency (1.83 vs. 11.67 vs. 23.0 promotions). The offline gate alone rejects 83.6% of candidates (10.17 out of 12.17); overall, 84.9% are filtered before promotion (10.34 out of 12.17). However, this experiment does not quantify safety against genuinely bad or catastrophic candidates, as no deliberately degraded models were injected. The experiment demonstrates the gating protocol's ability to reduce promotion frequency and maintain stability, not its ability to prevent unsafe model deployments under adversarial conditions. Future work should include candidate injection experiments to directly measure false negative rates (bad candidates that bypass gates).

4. **Degraded promotions indicate validation-production mismatch:** 45.5% of promotions are "degraded" (shadow MAE > validation MAE), but:
   - This indicates validation-production distribution mismatch under non-stationary streams
   - Both gates still required candidate to outperform production at evaluation time
   - Despite high degraded rate, proposed strategy achieves stable (if higher) MAE overall
   - Degraded flag is a diagnostic signal for future work on adaptive validation strategies

**Definition of "Degraded Promotion":** A promotion is flagged as degraded if `shadow_mae > validation_mae` for the promoted candidate. This detects cases where a candidate validated well offline but performed worse on the live, non-stationary stream than expected. It does **not** mean the candidate was worse than the current production model—both gates verified candidate > production before promotion.

**Statistical Validation Bounds:** We do not establish confidence intervals or error rates for shadow testing (e.g., "with 95% confidence, candidate is at least X% better than production"). Shadow window size (20 samples) is a fixed design choice, not empirically optimized. This is an acknowledged limitation.

### 5.5 Statistical Significance Summary

Table 4 summarizes Friedman omnibus tests (α = 0.05):

| Metric | χ² | df | *p*-value | Significant |
|--------|----|----|-----------|-------------|
| mae | 68.55 | 3 | 8.72e-15 | ✓ |
| rmse | 65.80 | 3 | 3.38e-14 | ✓ |
| detection_delay | 72.00 | 3 | 1.59e-15 | ✓ |
| model_promoted_events | 72.00 | 3 | 1.59e-15 | ✓ |
| total_adaptation_time | 65.60 | 3 | 3.73e-14 | ✓ |

All five primary metrics show statistically significant differences across strategies.

Figure 4 (pairwise significance heatmap) shows 27 out of 30 pairwise comparisons remain significant after Holm-Bonferroni correction. The three non-significant comparisons all involve detection_delay between strategies with identical adaptive timing (naive vs. proposed vs. static all trigger on drift detection, yielding similar delays).

Figure 5 (effect size heatmap) shows most pairwise effect sizes *r* > 0.9, indicating very large practical differences between strategies.

---

## 6. Discussion

### 6.1 The Stability–Accuracy Trade-off is Fundamental

Our results demonstrate that **model stability** (low promotion frequency) and **prediction accuracy** (low MAE/RMSE) are competing objectives in adaptive physical prognostics. No single strategy optimizes both:

- Scheduled achieves best accuracy (MAE 8.71) by replacing models every 100 cycles, but this creates maximum churn (23 promotions)
- Proposed achieves best stability (1.83 promotions) by rigorously gating candidates, but this degrades accuracy (MAE 10.79)
- Naive adaptive occupies a middle ground (MAE 9.94, 11.67 promotions, 5.04s overhead)

This trade-off arises because **drift is gradual and heterogeneous** across scenarios:
- In **sensor_failure** and **sudden_spike**, all strategies detect quickly and adapt successfully, yielding similar MAE
- In **concept_drift** and **gradual_drift**, scheduled's proactive replacement outpaces reactive detection, maintaining lower MAE
- In **drift_recovery**, aggressive adaptation may **overfit to transient drift**, while conservative gating waits for sustained evidence before replacing models

### 6.2 Why Conservative Gating Increases Overhead

The 9.2× overhead of proposed strategy compared to naive adaptive (46.2s vs. 5.0s), or 6.9× compared to scheduled (46.2s vs. 6.7s), is primarily driven by shadow evaluation latency:

- Retraining time: ~5–6 seconds (similar across strategies)
- Shadow window: 20 parallel predictions required
- In sparse scenarios (e.g., drift_recovery, intermittent_spikes), collecting 20 samples may span 50–100 cycles
- Total shadow time: 21–105 seconds (mean across scenarios)

This overhead is a **deliberate design trade-off**: shadow evaluation provides additional live-stream evidence that candidates perform well on live, non-stationary streams, not just offline validation data. In resource-constrained edge devices, this cost may be prohibitive, favoring naive adaptive (no shadow overhead) over proposed.

### 6.3 Degraded Promotions Reveal Non-Stationarity Challenges

The 45.5% degraded promotion rate (shadow MAE > validation MAE) highlights a fundamental challenge in adaptive ML under non-stationarity: **validation data and production data undergo different distribution shifts**.

When a candidate trains on historical data, validates on a recent buffer, and then enters shadow evaluation, the stream may have drifted again by the time shadow completes. This causes validation metrics (offline gate) to be imperfect predictors of live performance (shadow evaluation).

However, degraded promotions are **not catastrophic failures**:
- Both gates still required candidate to outperform production
- Proposed strategy achieves stable (if higher) overall MAE despite degraded rate
- Degraded flag is a diagnostic signal for future work (e.g., adapt validation strategies to account for drift velocity)

### 6.4 Domain-Specific Adaptation Policies

Our results suggest optimal adaptation policy is **domain-specific**, not universal:

**1. Safety-Critical Physical Systems (Aerospace, Medical Devices, Nuclear Power):**
- **Priority:** Model stability, auditability, certification traceability
- **Recommendation:** Proposed (conservative gating)
- **Justification:** 1.83 promotions per stream minimizes certification burden; 2 MAE points higher error is acceptable if certified models reduce catastrophic failure risk

**2. Offline Batch Prediction (Maintenance Scheduling, Inventory Planning):**
- **Priority:** Prediction accuracy
- **Recommendation:** Scheduled (aggressive periodic)
- **Justification:** 23 promotions are operationally feasible in offline batch contexts; lowest MAE (8.71) maximizes maintenance optimization

**3. Real-Time Edge Devices (IoT Sensors, Distributed Fleets):**
- **Priority:** Computational efficiency, low latency
- **Recommendation:** Naive adaptive (reactive immediate)
- **Justification:** 5.04s overhead is minimally disruptive; 11.67 promotions balance accuracy and churn; no shadow evaluation latency

### 6.5 Comparison to Prior Work

Prior physical prognostics work focuses on model architecture (Random Forest, LSTM, CNN) without systematic evaluation of adaptation strategies [2, 3, 12, 17, 22]. Drift detection literature evaluates detection methods in isolation, not integrated with safety-critical promotion protocols [14, 18, 23]. Software canary deployment rarely targets physical streams where validation data itself is non-stationary [5, 7, 11, 20].

This work contributes:
1. Integrates drift detection, gating, and shadow evaluation for physical RUL
2. Quantifies the stability–accuracy trade-off across diverse degradation scenarios
3. Demonstrates that conservative gating reduces churn 6–13× at the cost of 8.5–23.9% higher MAE

---

## 7. Limitations

We acknowledge eight limitations:

### 7.1 Single Dataset

We evaluate only NASA C-MAPSS FD001 (turbofan engines, single operating condition, single fault mode). Results may not generalize to:
- Other physical systems (batteries, HVAC, industrial motors)
- Multiple operating conditions or fault modes
- Real-world fleet data with unlabeled degradation

**Future Work:** Cross-domain validation on battery degradation (NASA Prognostics Center of Excellence), industrial bearing datasets (FEMTO), and real-world turbofan telemetry.

### 7.2 Simulated Degradation Scenarios

Our eight scenarios inject synthetic drift, noise, and anomalies. Real-world degradation may:
- Combine multiple scenario types simultaneously
- Exhibit temporal dependencies (e.g., drift accelerates near failure)
- Involve rare fault modes not captured by FD001

**Future Work:** Collaborate with industry partners to access field deployment data with ground-truth failure labels.

### 7.3 Fixed Thresholds

We use fixed thresholds (gate = 0.95, shadow = 0.95, cooldown = 30 cycles, shadow window = 20 samples) without sensitivity analysis. Optimal thresholds may vary by:
- Scenario type (gradual vs. sudden drift)
- Deployment context (safety-critical vs. batch prediction)
- Dataset characteristics (sample rate, signal-to-noise ratio)

**Future Work:** Multi-objective optimization over threshold space; adaptive threshold selection based on drift velocity.

### 7.4 No Ground-Truth Drift Labels

We lack ground-truth drift onset times, magnitudes, and durations for computing detection precision/recall. We demonstrate architectural integration and operational feasibility, not statistical detection quality.

**Future Work:** Synthetic streams with known drift parameters; benchmark against Massive Online Analysis (MOA) drift generators.

### 7.5 No Statistical Validation Bounds for Shadow Testing

We do not establish confidence intervals or error rates for shadow evaluation (e.g., "with 95% confidence, candidate is at least X% better"). Shadow window size (20 samples) is a design choice, not empirically validated as statistically sufficient under non-stationary streams.

**Future Work:** Derive sequential analysis bounds for shadow testing under drift; adaptive window sizing based on variance.

### 7.6 Resource Constraints Not Evaluated (RQ4 Out of Scope)

We do not experimentally vary or measure:
- CPU/memory usage
- Network bandwidth
- Adaptive rate limiting under resource constraints

Resource-aware ingestion is architecturally implemented (api_server.py, RateLimitController) but not experimentally validated. RQ4 is explicitly out of scope.

**Future Work:** Edge device deployment (Raspberry Pi, Jetson Nano); resource-constrained simulation.

### 7.7 Single ML Algorithm

We use Random Forest regressors exclusively. Results may differ for:
- Gradient boosting (XGBoost, LightGBM)
- Neural networks (LSTM, CNN, Transformer)
- Online learning algorithms (incremental updates vs. full retraining)

**Future Work:** Compare adaptation strategies across model families; investigate online learning as alternative to batch retraining.

### 7.8 No Comparison to Manual Intervention Baselines

All four strategies are autonomous. We do not evaluate:
- Human-in-the-loop decision-making
- Semi-supervised adaptation with operator approval
- Cost-benefit models balancing false alarms vs. missed degradation

**Future Work:** User studies with maintenance engineers; hybrid autonomy levels.

---

## 8. Future Work

Beyond addressing the eight limitations above, we identify three research directions:

### 8.1 Ablation Studies

**Two-Stage Gating Ablation:**
- Gate-only (no shadow evaluation)
- Shadow-only (no offline gate)
- No gating (immediate replacement)

Compare degraded promotion rates, MAE, and promotion frequency to isolate the contribution of each gate.

**Threshold Sensitivity:**
- Vary gate threshold: [0.90, 0.95, 0.98]
- Vary shadow window: [10, 20, 50, 100]
- Measure impact on trade-off surface (accuracy vs. stability vs. overhead)

### 8.2 Catastrophic Candidate Injection

Inject deliberately degraded candidates (e.g., trained on wrong data, corrupted weights) to measure:
- Gate rejection rate for truly bad candidates
- Shadow rejection rate for candidates that pass gate but fail on live stream
- False negative rate (bad candidates that promote)

This addresses the limitation that we cannot directly measure "catastrophic promotion prevention" without ground-truth bad candidates.

### 8.3 Cross-Domain Generalization

Evaluate the proposed architecture on:
- **Battery prognostics:** NASA PCoE battery datasets (capacity fade, impedance rise)
- **Industrial bearings:** FEMTO challenge datasets (vibration analysis, wear patterns)
- **HVAC systems:** Chiller fault detection and degradation (ASHRAE datasets)

Assess whether the stability–accuracy trade-off generalizes across physical domains or requires domain-specific tuning.

---

## 9. Conclusion

Adaptive model management for physical prognostics faces a fundamental trade-off between model stability (low replacement frequency) and prediction accuracy (low error under drift). We present an integrated closed-loop architecture combining multi-channel statistical drift detection (ADWIN, KS-test, anomaly detection), autonomous background retraining, and two-stage validation (offline performance gate + live shadow evaluation) for continuous RUL regression on streaming turbofan telemetry.

A comprehensive 96-run controlled experiment (4 strategies × 8 degradation scenarios × 3 seeds, blocked factorial design) on NASA C-MAPSS FD001 demonstrates that conservative multi-gated adaptation reduces model replacement frequency by 6.4–12.6× compared to aggressive baselines (1.83 vs. 11.67 vs. 23.0 promotions per 2400-cycle stream, *p* < 0.001) at the cost of 8.5–23.9% higher prediction error (MAE 10.79 vs. 9.94 vs. 8.71, *p* < 0.001) and 9.2× higher adaptation overhead than naive adaptive (46.2s vs. 5.0s, *p* < 0.001).

This trade-off is statistically significant in the blocked 96-run design spanning all eight degradation scenarios, suggesting the optimal adaptation policy is domain-specific rather than universally optimal. The results indicate that applications prioritizing model stability may prefer conservative adaptation, whereas applications prioritizing prediction accuracy may prefer more aggressive adaptation. Our deterministic reproducibility pipeline with provenance tracking enables exact replication of all 96 experimental runs, advancing reproducible research in adaptive ML.

**Key Takeaway:** More adaptation is not always better. Conservative gating dramatically reduces model churn but requires accepting modestly higher prediction error and substantially higher computational overhead. MLOps practitioners must choose adaptation policies aligned with their domain's operational priorities.

---

## 10. Acknowledgments

[To be added]

---

## 11. References

[1] Konda, S. R. (2024). Advancements in Self-Healing Technology for Software Systems. *International Journal of Engineering Research and Applications (IJERA)*.

[2] Pani, S., Pattnaik, O., & Pattanayak, B. K. (2024). Predictive Maintenance in Industrial IoT Using Machine Learning Approach. *International Journal of Intelligent Systems and Applications in Engineering (IJISAE)*.

[3] Benmansour, O., Medarhri, I., & Hosni, M. (2026). Predictive Maintenance in Industrial Systems Using Machine Learning: A Review. *Statistics, Optimization and Information Computing*.

[4] Patil, R. V., Kudande, V., Jagtap, S., Jadhav, S., & Jawalgekar, A. (2025). Self Healing Infrastructure System. *International Journal of Electrical, Electronics and Computer Systems (IJEECS)*.

[5] Khan, R. S. (2025). AI-Based Rate Limiting for Cloud Infrastructure: Implementation Guide. *Journal of Computer Science and Technology Studies (JCSTS)*.

[6] Attipalli, A., Enokkaren, S. J., Bitkuri, V., Kendyala, R., Kurma, J., & Mamidala, J. V. (2021). A Review of AI and Machine Learning Solutions for Fault Detection and Self-Healing in Cloud Services. *International Journal of AI, Big Data, Computational and Management Studies (IJAIBDCMS)*.

[7] Jangam, S. K., & Karri, N. (2022). Potential of AI and ML to Enhance Error Detection, Prediction, and Automated Remediation in Batch Processing. *International Journal of AI, Big Data, Computational and Management Studies (IJAIBDCMS)*.

[8] Jangam, S. K. (2022). Role of AI and ML in Enhancing Self-Healing Capabilities, Including Predictive Analysis and Automated Recovery. *International Journal of Artificial Intelligence, Data Science, and Machine Learning (IJAIDSML)*.

[9] Syed, A. A. M., & Anazagasty, E. (2024). AI-Driven Infrastructure Automation: Leveraging AI and ML for Self-Healing and Auto-Scaling Cloud Environments. *International Journal of Artificial Intelligence, Data Science, and Machine Learning (IJAIDSML)*.

[10] Shah, S., & Priyanshi, Er. (2019). Adaptive Rate Limiting for Microservices. *International Journal of Current Science (IJCSPUB)*.

[11] Jangam, S. K. (2022). Self-Healing Autonomous Software Code Development. *International Journal of Emerging Trends in Computer Science and Information Technology (IJETCSIT)*.

[12] Nsor, M. (2024). Predictive Maintenance Using Machine Learning for Engineering Systems Through Real-Time Sensor Data and Anomaly Detection Models. *International Journal of Research Publication and Reviews (IJRPR)*.

[13] Revanasiddappa, N. B. (2022). Machine learning for predictive maintenance in self-healing software services. *International Journal of Science and Research Archive (IJSRA)*.

[14] Patchipala, S. G. (2023). Tackling data and model drift in AI: Strategies for maintaining accuracy during ML model inference. *International Journal of Science and Research Archive (IJSRA)*.

[15] Patel, J., & Shah, H. (2021). Software Engineering Revolutionized by Machine Learning-Powered Self-Healing Systems. *International Research Journal of Engineering & Applied Sciences (IRJEAS)*.

[16] Shah, H., & Patel, J. (2023). Machine Learning and Self-Healing Capabilities Combined in Adaptive AI Architectures. *International Research Journal of Engineering & Applied Sciences (IRJEAS)*.

[17] Malaiyappan, J. N. A., Krishnamoorthy, G., & Jangoan, S. (2024). Predictive Maintenance using Machine Learning in Industrial IoT. *International Journal of Innovative Science and Research Technology (IJISRT)*.

[18] Waseem, Q., Wan Din, W. I. S., Fairooz, T., & Baharin, A. T. (2025). Drift Management in ML-Based IoT Device Classification: A Survey and Evaluation. *International Journal on Advanced Science, Engineering and Information Technology (IJASEIT)*.

[19] Shah, H., & Patel, J. (2023). Machine Learning-Driven Self-Healing Systems: Revolutionizing Software Engineering. *International Journal of Intelligent Systems and Applications in Engineering (IJISAE)*.

[20] Apuri, H., Chinthala, M. M. R., Goel, S., Aurangabadkar, M., & Yepuri, C. (2026). Self-Healing Infrastructure: Autonomous LLM Agents for Real-Time Remediation of Configuration Drift and Security Misconfigurations in IaC Deployments. *Blue Eyes Intelligence Engineering and Sciences Publication (BEIESP)*.

[21] Patel, J. (2018). Self-Healing Mechanisms in Software Development- A Machine Learning Method. *International Research Journal of Engineering & Applied Sciences (IRJEAS)*.

[22] Bhattacharyya, D., & Kundu, U. K. (2025). The Role of Industrial IoT and Machine Learning in Reshaping Predictive Maintenance Strategies. *Journal of Emerging Trends in Computer Science and Applications (JETCSA)*.

[23] Mannapur, S. B. (2025). Understanding Data Drift and Concept Drift in Machine Learning Systems. *International Journal of Scientific Research in Computer Science, Engineering and Information Technology (IJSRCSEIT)*.

---

**Note:** Paper numbering (1–23) corresponds to the filename convention paper01.pdf–paper23.pdf in the `literature/` directory. See `literature/PAPER_MAPPING.md` for the complete mapping between paper IDs, original filenames, and bibliographic metadata.
## Appendix A: Reproducibility Instructions

All experiments are reproducible using the deterministic pipeline:

```bash
# 1. Generate the 96-run manifest
python -m scripts.matrix_orchestration.generate_manifest \
  --output-dir experiments/results

# 2. Execute the full matrix (96 runs, ~4 hours)
python -m scripts.matrix_orchestration.run_matrix \
  --manifest experiments/results/experiment_manifest.csv \
  --yes

# 3. Verify completion
python -m scripts.matrix_orchestration.verify_completion \
  --manifest experiments/results/experiment_manifest.csv \
  --results-dir experiments/results

# 4. Aggregate results
python -m scripts.analysis.aggregate_results \
  --manifest experiments/results/experiment_manifest.csv \
  --results-dir experiments/results \
  --output experiments/results/aggregated_results.csv

# 5. Statistical QC
python -m scripts.analysis.statistical_qc \
  --aggregated experiments/results/aggregated_results.csv \
  --output experiments/results/aggregated_results.qc.json

# 6. Statistical analysis
python -m scripts.analysis.statistical_analysis \
  --aggregated experiments/results/aggregated_results.csv \
  --manifest experiments/results/experiment_manifest.csv \
  --qc experiments/results/aggregated_results.qc.json \
  --output experiments/results/statistical_analysis.json

# 7. Generate figures
python -m scripts.analysis.generate_figures \
  --aggregated experiments/results/aggregated_results.csv \
  --statistical experiments/results/statistical_analysis.json \
  --output-dir experiments/results/figures
```

**Dataset:** Download NASA C-MAPSS FD001 from [NASA Prognostics Data Repository]. Verify checksum: `md5sum dataset/raw/train_FD001.txt` should match `1721c96c01e188569f0e7bb16b1ea493`.

**Environment:** Python 3.12.0, dependencies listed in `requirements.txt`. Install: `pip install -r requirements.txt`.

**Execution Time:** Full 96-run matrix requires approximately 4 hours on a modern workstation (Intel i7, 16GB RAM).

**Git Provenance:** Original experiment execution at commit `fa6cbd5571184daf2ddbebd319aaf6614c276f9b`.

---

## Appendix B: Full Statistical Analysis Tables

[Full Wilcoxon pairwise comparison tables with uncorrected p-values, corrected p-values, effect sizes, and significance flags are available in `experiments/results/statistical_analysis.json` and `statistical_analysis.md`.]

---

**END OF PAPER DRAFT**

*Word Count (excluding references, acknowledgments, appendices): ~8,500 words*

*Page Count (estimated): ~18-20 pages in conference format*

*Figures: 8 (mae_boxplot, rmse_boxplot, mae_by_scenario, rmse_by_scenario, strategy_comparison_barplot, adaptation_metrics_combined, effect_size_heatmap, pairwise_significance_heatmap)*

*Tables: 4 (strategy summary, drift detection, gating analysis, Friedman tests)*
