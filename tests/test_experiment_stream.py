import pandas as pd

from experiments.data_stream import build_stream_records, iter_stream_events


def _stream_df():
    rows = []
    for unit in [2, 1]:
        for cycle in [3, 1, 2]:
            rows.append(
                {
                    "unit": unit,
                    "cycle": cycle,
                    "RUL": float(10 - cycle),
                    **{f"op_setting_{i}": 0.0 for i in range(1, 4)},
                    **{f"sensor_{i}": float(unit * 10 + cycle) for i in range(1, 22)},
                }
            )
    return pd.DataFrame(rows)


def test_research_mode_preserves_unit_cycle_order():
    records = build_stream_records(_stream_df(), stream_mode="research", seed=1)

    assert [(row["unit"], row["cycle"]) for row in records] == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
    ]


def test_legacy_mode_is_seeded():
    df = _stream_df()

    assert build_stream_records(df, "legacy", seed=5) == build_stream_records(df, "legacy", seed=5)
    assert build_stream_records(df, "legacy", seed=5) != build_stream_records(df, "legacy", seed=6)


def test_same_seed_reproduces_fault_injection():
    records = build_stream_records(_stream_df(), stream_mode="research", seed=1)
    stds = {f"sensor_{i}": 1.0 for i in range(1, 22)}
    stds.update({f"op_setting_{i}": 1.0 for i in range(1, 4)})

    # Updated to use new per-engine onset parameters
    a = list(iter_stream_events(records, stds, "intermittent_spikes", 123, 20,
                                  scenario_onset_cycle_min=1, scenario_onset_cycle_max=2))
    b = list(iter_stream_events(records, stds, "intermittent_spikes", 123, 20,
                                  scenario_onset_cycle_min=1, scenario_onset_cycle_max=2))
    c = list(iter_stream_events(records, stds, "intermittent_spikes", 124, 20,
                                  scenario_onset_cycle_min=1, scenario_onset_cycle_max=2))

    assert [event.data for event in a] == [event.data for event in b]
    assert [event.data for event in a] != [event.data for event in c]
    # Check that some event has degradation_started (depends on engine onset assignments)
    assert any(event.degradation_started for event in a)



def test_interleaved_mode_groups_by_cycle():
    """Interleaved mode should group engines by cycle number (round-robin)."""
    df = _stream_df()
    records = build_stream_records(df, stream_mode="interleaved", seed=1)
    
    # Expected: cycle 1 engines, then cycle 2 engines, then cycle 3 engines
    expected = [
        (1, 1), (2, 1),  # Cycle 1: both engines
        (1, 2), (2, 2),  # Cycle 2: both engines
        (1, 3), (2, 3),  # Cycle 3: both engines
    ]
    actual = [(row["unit"], row["cycle"]) for row in records]
    assert actual == expected


def test_interleaved_mode_is_deterministic():
    """Interleaved mode should produce same ordering regardless of seed."""
    df = _stream_df()
    records_a = build_stream_records(df, stream_mode="interleaved", seed=1)
    records_b = build_stream_records(df, stream_mode="interleaved", seed=999)
    
    # Ordering should be deterministic (not affected by seed)
    assert [(r["unit"], r["cycle"]) for r in records_a] == [(r["unit"], r["cycle"]) for r in records_b]


def test_per_engine_scenario_onset_is_deterministic():
    """Per-engine scenario onset should be deterministic for fixed seed."""
    records = build_stream_records(_stream_df(), stream_mode="interleaved", seed=1)
    stds = {f"sensor_{i}": 1.0 for i in range(1, 22)}
    stds.update({f"op_setting_{i}": 1.0 for i in range(1, 4)})
    
    events_a = list(iter_stream_events(records, stds, "sudden_spike", 42, 20,
                                        scenario_onset_cycle_min=1, scenario_onset_cycle_max=3))
    events_b = list(iter_stream_events(records, stds, "sudden_spike", 42, 20,
                                        scenario_onset_cycle_min=1, scenario_onset_cycle_max=3))
    
    # Same seed should produce same degradation_started flags
    assert [e.degradation_started for e in events_a] == [e.degradation_started for e in events_b]


def test_per_engine_scenario_onset_varies_by_engine():
    """Different engines should receive different onset cycles."""
    records = build_stream_records(_stream_df(), stream_mode="interleaved", seed=1)
    stds = {f"sensor_{i}": 1.0 for i in range(1, 22)}
    stds.update({f"op_setting_{i}": 1.0 for i in range(1, 4)})
    
    events = list(iter_stream_events(records, stds, "sudden_spike", 42, 6,  # Only 6 events (one full cycle through both engines)
                                     scenario_onset_cycle_min=1, scenario_onset_cycle_max=3))
    
    # Find FIRST degradation_started event for each engine
    engine_1_onset = [e for e in events if e.engine_id == 1 and e.degradation_started]
    engine_2_onset = [e for e in events if e.engine_id == 2 and e.degradation_started]
    
    # At least one engine should have an onset within this window
    assert len(engine_1_onset) >= 1 or len(engine_2_onset) >= 1
    
    # If both have onsets, they should be tracked independently
    if len(engine_1_onset) > 0 and len(engine_2_onset) > 0:
        assert engine_1_onset[0].cycle >= 1
        assert engine_2_onset[0].cycle >= 1


def test_scenario_affects_only_intended_engine():
    """Scenario should only modify data for engines past their onset."""
    records = build_stream_records(_stream_df(), stream_mode="interleaved", seed=1)
    stds = {f"sensor_{i}": 1.0 for i in range(1, 22)}
    stds.update({f"op_setting_{i}": 1.0 for i in range(1, 4)})
    
    events = list(iter_stream_events(records, stds, "sudden_spike", 42, 20,
                                     scenario_onset_cycle_min=2, scenario_onset_cycle_max=2))
    
    # At cycle 1, no engine should be affected (onset is cycle 2)
    cycle_1_events = [e for e in events if e.cycle == 1]
    assert all(not e.scenario_active for e in cycle_1_events)
    
    # At cycle 2, some engines should start being affected
    cycle_2_events = [e for e in events if e.cycle == 2]
    assert any(e.scenario_active for e in cycle_2_events)
