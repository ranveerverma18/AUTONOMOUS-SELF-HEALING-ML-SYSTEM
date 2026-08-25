"""Utilities for aggregating completed experiment summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_summary_table(aggregated_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(aggregated_dir.glob("*_summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "run_id": payload["run_id"],
                **payload["config"],
                **payload["summary"],
            }
        )
    return pd.DataFrame(rows)
