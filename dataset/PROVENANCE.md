# Dataset Provenance

## Source

- **Organization:** NASA Prognostics Center of Excellence  
- **Dataset:** C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Turbofan Engine Degradation Simulation Data Set  
- **Subset:** FD001 (single operating condition, single fault mode)  
- **License:** Public domain (U.S. Government work)  
- **URL:** https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

## Citation

```bibtex
@inproceedings{saxena2008damage,
  title={Damage propagation modeling for aircraft engine run-to-failure simulation},
  author={Saxena, Abhinav and Goebel, Kai and Simon, Don and Eklund, Neil},
  booktitle={2008 International Conference on Prognostics and Health Management},
  pages={1--9},
  year={2008},
  organization={IEEE}
}
```

## Files

| File | Description | Size (bytes) | MD5 Checksum |
|------|-------------|--------------|--------------|
| train_FD001.txt | Training data (100 engine run-to-failure trajectories) | 3,535,987 | `1721c96c01e188569f0e7bb16b1ea493` |
| test_FD001.txt | Test data (100 partial trajectories) | 2,241,951 | `049f39b8448989ac72f375df5234d733` |
| RUL_FD001.txt | True Remaining Useful Life for test set | 529 | `e7ac1c848b4316d89fd8b4637db53004` |

## Access History

- **Original download:** 2026-03-25 (per Git initial commit e660d8f)
- **Checksum verification:** 2026-08-24

## Data Format

Each line represents one time step of engine operation:

- **Column 1:** Engine unit number (1-100)
- **Column 2:** Time cycle (sequential, starts at 1 for each unit)
- **Columns 3-5:** Operational settings (3 values)
- **Columns 6-26:** Sensor measurements (21 values)

**Training data** contains complete run-to-failure trajectories for 100 engines.  
**Test data** contains partial trajectories; true RUL for each is provided in RUL_FD001.txt.

## Usage in This Project

**Active data:** `train_FD001.txt` only  
**Test/RUL files:** Present but not used by the research experiment framework or live dashboard

### Data Split

The 100 engine units from `train_FD001.txt` are split into:

1. **Initial training:** 76 units (76%) - Used to train the initial production model
2. **Live stream / monitoring:** 24 units (24%) - Used for simulation, retraining, and validation

Split is performed with `seed=42` using `split_train_stream_units()` in `experiments/data_stream.py`.

### RUL Computation

RUL (Remaining Useful Life) is computed as:
```
RUL = max_cycle_for_unit - current_cycle
```

Clipped to a maximum of 125 cycles (per NASA C-MAPSS convention).

### Preprocessing

See `dataset/processed/preprocess_module.py` for:
- Data loading from whitespace-delimited text format
- RUL target calculation
- Standard scaling (Z-score normalization)
- Unit-based train/validation splitting logic

## Data Quality

- **Missing values:** None (confirmed during preprocessing)
- **Outliers:** Present (part of normal sensor degradation patterns)
- **Sensor columns:** 21 sensors labeled sensor_1 through sensor_21
- **Operational settings:** 3 settings labeled setting_1 through setting_3

## Validation Split Methodology

All model training in this project uses **unit-disjoint splits** to prevent data leakage:

```
training_units ∩ validation_units = ∅
```

This ensures validation sets contain completely unseen engine units, not just unseen time steps from known engines.

See `ml/evaluation.py:split_training_and_validation()` for implementation.

## References

- **NASA PCoE Data Repository:** https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
- **PHM08 Challenge:** Dataset originally released for 2008 PHM Conference Data Challenge
- **Related Work:** See citations in project README.md
