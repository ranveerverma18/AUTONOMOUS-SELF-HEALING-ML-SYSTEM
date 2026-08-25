import pandas as pd

from experiments.baselines import make_strategy
from experiments.config import ExperimentConfig
from experiments.data_stream import StreamEvent


def _train_df(n=40):
    rows = []
    for i in range(n):
        rows.append(
            {
                "unit": i // 10 + 1,
                "cycle": i % 10 + 1,
                "RUL": float(i % 10),
                **{f"op_setting_{j}": 0.0 for j in range(1, 4)},
                **{f"sensor_{j}": float(i % 10) for j in range(1, 22)},
            }
        )
    return pd.DataFrame(rows)


def test_static_strategy_does_not_retrain():
    config = ExperimentConfig(strategy="static", stream_length=1)
    strategy = make_strategy(config, _train_df(), {f"sensor_{i}": 1.0 for i in range(1, 22)})
    event = StreamEvent(0, 1, 1, _train_df(1).iloc[0].to_dict(), False, None, False)

    row = strategy.process(event, "run")

    assert row["retraining_triggered"] is False
    assert row["model_version"] == 1
