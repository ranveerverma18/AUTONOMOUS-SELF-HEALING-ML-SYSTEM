# Statistical Analysis Report

**Aggregated File:** `experiments\results\aggregated_results.csv`
**Manifest File:** `experiments\results\experiment_manifest.csv`
**QC Report:** `experiments\results\aggregated_results.qc.json`

## Data Summary

- **Total Runs:** 96
- **Strategies (4):** naive_adaptive, proposed, scheduled, static
- **Scenarios (8):** concept_drift, correlated_drift, drift_recovery, gradual_drift, high_noise, intermittent_spikes, sensor_failure, sudden_spike
- **Seeds (3):** 42, 123, 456
- **Block Definition:** `scenario_seed`
- **Number of Blocks:** 24

## Provenance

- **Git Commit:** `6e81a330d8d29cf35bbc5d85855e56130b1b5dee`
- **Dataset Checksum:** `1721c96c01e188569f0e7bb16b1ea493`
- **Python:** 3.12.0
- **SciPy:** 1.18.0
- **NumPy:** 2.5.1

## Friedman Omnibus Tests

| Metric | χ² | df | p-value | Significant (α=0.05) |
|--------|----|----|---------|----------------------|
| mae | 68.5500 | 3 | 0.000000 | ✅ Yes |
| rmse | 65.8000 | 3 | 0.000000 | ✅ Yes |
| detection_delay | 72.0000 | 3 | 0.000000 | ✅ Yes |
| model_promoted_events | 72.0000 | 3 | 0.000000 | ✅ Yes |
| total_adaptation_time | 65.6000 | 3 | 0.000000 | ✅ Yes |

## Pairwise Wilcoxon Tests (Holm-Corrected)

### detection_delay

| Comparison | n | W | p-value | p-corrected | Effect Size (r) | Significant |
|------------|---|---|---------|-------------|-----------------|-------------|
| naive_adaptive vs proposed | 24 | 0.00 | nan | 1.000000 | 0.0000 | ❌ |
| naive_adaptive vs scheduled | 24 | 0.00 | 0.000014 | 0.000071 | 1.0000 | ✅ |
| naive_adaptive vs static | 24 | 0.00 | nan | 1.000000 | 0.0000 | ❌ |
| proposed vs scheduled | 24 | 0.00 | 0.000014 | 0.000042 | 1.0000 | ✅ |
| proposed vs static | 24 | 0.00 | nan | 1.000000 | 0.0000 | ❌ |
| scheduled vs static | 24 | 0.00 | 0.000014 | 0.000014 | -1.0000 | ✅ |

### mae

| Comparison | n | W | p-value | p-corrected | Effect Size (r) | Significant |
|------------|---|---|---------|-------------|-----------------|-------------|
| naive_adaptive vs proposed | 24 | 4.00 | 0.000001 | 0.000001 | -0.9733 | ✅ |
| naive_adaptive vs scheduled | 24 | 2.00 | 0.000000 | 0.000001 | 0.9867 | ✅ |
| naive_adaptive vs static | 24 | 0.00 | 0.000000 | 0.000001 | -1.0000 | ✅ |
| proposed vs scheduled | 24 | 0.00 | 0.000000 | 0.000001 | 1.0000 | ✅ |
| proposed vs static | 24 | 0.00 | 0.000000 | 0.000000 | -1.0000 | ✅ |
| scheduled vs static | 24 | 0.00 | 0.000000 | 0.000000 | -1.0000 | ✅ |

### model_promoted_events

| Comparison | n | W | p-value | p-corrected | Effect Size (r) | Significant |
|------------|---|---|---------|-------------|-----------------|-------------|
| naive_adaptive vs proposed | 24 | 0.00 | 0.000016 | 0.000049 | 1.0000 | ✅ |
| naive_adaptive vs scheduled | 24 | 0.00 | 0.000017 | 0.000034 | -1.0000 | ✅ |
| naive_adaptive vs static | 24 | 0.00 | 0.000017 | 0.000017 | 1.0000 | ✅ |
| proposed vs scheduled | 24 | 0.00 | 0.000013 | 0.000064 | -1.0000 | ✅ |
| proposed vs static | 24 | 0.00 | 0.000013 | 0.000051 | 1.0000 | ✅ |
| scheduled vs static | 24 | 0.00 | 0.000001 | 0.000006 | 1.0000 | ✅ |

### rmse

| Comparison | n | W | p-value | p-corrected | Effect Size (r) | Significant |
|------------|---|---|---------|-------------|-----------------|-------------|
| naive_adaptive vs proposed | 24 | 0.00 | 0.000000 | 0.000001 | -1.0000 | ✅ |
| naive_adaptive vs scheduled | 24 | 8.00 | 0.000003 | 0.000006 | 0.9467 | ✅ |
| naive_adaptive vs static | 24 | 0.00 | 0.000000 | 0.000001 | -1.0000 | ✅ |
| proposed vs scheduled | 24 | 0.00 | 0.000000 | 0.000000 | 1.0000 | ✅ |
| proposed vs static | 24 | 18.00 | 0.000030 | 0.000030 | -0.8800 | ✅ |
| scheduled vs static | 24 | 0.00 | 0.000000 | 0.000000 | -1.0000 | ✅ |

### total_adaptation_time

| Comparison | n | W | p-value | p-corrected | Effect Size (r) | Significant |
|------------|---|---|---------|-------------|-----------------|-------------|
| naive_adaptive vs proposed | 24 | 0.00 | 0.000000 | 0.000001 | -1.0000 | ✅ |
| naive_adaptive vs scheduled | 24 | 57.00 | 0.006516 | 0.006516 | -0.6200 | ✅ |
| naive_adaptive vs static | 24 | 0.00 | 0.000000 | 0.000001 | 1.0000 | ✅ |
| proposed vs scheduled | 24 | 0.00 | 0.000000 | 0.000000 | 1.0000 | ✅ |
| proposed vs static | 24 | 0.00 | 0.000000 | 0.000000 | 1.0000 | ✅ |
| scheduled vs static | 24 | 0.00 | 0.000000 | 0.000000 | 1.0000 | ✅ |
