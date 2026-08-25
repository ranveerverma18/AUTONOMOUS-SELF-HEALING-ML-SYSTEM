# Experiments

This package runs reproducible research experiments without using the dashboard
or API server as the primary experiment runner.

The first supported run is:

```powershell
python -m experiments.runner --strategy proposed --scenario gradual_drift --seed 42 --stream-length 320 --output-dir experiments/results
```

Outputs are machine-readable:

- `experiments/results/raw/*_events.csv` contains one row per processed stream event.
- `experiments/results/aggregated/*_summary.json` contains run configuration and summary metrics.

The default stream mode is `research`, which sorts C-MAPSS stream records by
engine unit and cycle.  The previous dashboard ordering is available as
`legacy`, where stream rows are randomly permuted with the configured seed.
