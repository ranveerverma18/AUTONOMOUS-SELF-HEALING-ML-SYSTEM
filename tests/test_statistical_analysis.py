"""
Tests for scripts/analysis/statistical_analysis.py

Tests statistical analysis tool with synthetic data covering:
- Complete block structure
- Missing strategy in block
- Duplicate block
- Insufficient strategies
- Insufficient blocks
- NaN/inf values
- Invalid metric
- Friedman test
- Wilcoxon pairwise tests
- Holm correction
- Effect size calculation
- Deterministic repeated execution
"""

import pytest
import json
import csv
from pathlib import Path
import numpy as np
from scipy import stats

from scripts.analysis.statistical_analysis import (
    load_aggregated_data,
    load_qc_report,
    validate_qc_passed,
    build_block_structure,
    validate_block_structure,
    extract_metric_matrix,
    friedman_test,
    wilcoxon_signed_rank,
    holm_correction,
    perform_pairwise_tests,
    analyze_metric,
    run_statistical_analysis,
    FriedmanResult,
    WilcoxonResult
)


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files."""
    return tmp_path


@pytest.fixture
def complete_aggregated_data(temp_dir):
    """
    Create complete synthetic aggregated data.
    4 strategies × 2 scenarios × 2 seeds = 16 runs
    """
    csv_path = temp_dir / "aggregated.csv"
    
    strategies = ['static', 'scheduled', 'naive_adaptive', 'proposed']
    scenarios = ['gradual_drift', 'sudden_spike']
    seeds = [42, 123]
    
    rows = []
    for strategy in strategies:
        for scenario in scenarios:
            for seed in seeds:
                # Generate synthetic metric values
                # Make proposed slightly better than others
                base_mae = 10.0
                if strategy == 'proposed':
                    mae = base_mae - 1.0 + np.random.RandomState(seed).uniform(-0.5, 0.5)
                else:
                    mae = base_mae + np.random.RandomState(seed).uniform(-0.5, 0.5)
                
                rows.append({
                    'run_id': f'{strategy}_{scenario}_seed{seed}_20260823_120000',
                    'strategy': strategy,
                    'scenario': scenario,
                    'seed': str(seed),
                    'mae': f'{mae:.4f}',
                    'rmse': f'{mae * 1.2:.4f}',
                    'detection_delay': '50',
                    'model_promoted_events': '3',
                    'total_adaptation_time': '100.5',
                    'drift_detections': '10',
                    'git_commit': 'abc1234',
                    'dataset_checksum': 'def5678',
                    'python_version': '3.12.0'
                })
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    return csv_path


@pytest.fixture
def manifest_file(temp_dir):
    """Create minimal manifest file."""
    manifest_path = temp_dir / "manifest.csv"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("strategy,scenario,seed,run_id,status\n")
        f.write("static,gradual_drift,42,run1,completed\n")
    return manifest_path


@pytest.fixture
def qc_passed(temp_dir):
    """Create QC report that passed."""
    qc_path = temp_dir / "qc_passed.json"
    with open(qc_path, 'w', encoding='utf-8') as f:
        json.dump({"passed": True, "issues": []}, f)
    return qc_path


@pytest.fixture
def qc_failed(temp_dir):
    """Create QC report that failed."""
    qc_path = temp_dir / "qc_failed.json"
    with open(qc_path, 'w', encoding='utf-8') as f:
        json.dump({"passed": False, "issues": ["Test failure"]}, f)
    return qc_path


def test_load_aggregated_data(complete_aggregated_data):
    """Test loading aggregated CSV."""
    rows, strategies, scenarios, seeds = load_aggregated_data(complete_aggregated_data)
    
    assert len(rows) == 16
    assert strategies == ['naive_adaptive', 'proposed', 'scheduled', 'static']
    assert scenarios == ['gradual_drift', 'sudden_spike']
    assert seeds == [42, 123]


def test_load_qc_report(qc_passed):
    """Test loading QC report."""
    report = load_qc_report(qc_passed)
    assert report['passed'] is True


def test_validate_qc_passed_success(qc_passed):
    """Test QC validation passes."""
    report = load_qc_report(qc_passed)
    validate_qc_passed(report)  # Should not raise


def test_validate_qc_passed_failure(qc_failed):
    """Test QC validation fails."""
    report = load_qc_report(qc_failed)
    with pytest.raises(ValueError, match="QC did not pass"):
        validate_qc_passed(report)


def test_build_block_structure_scenario_seed(complete_aggregated_data):
    """Test building blocks with scenario_seed definition."""
    rows, _, _, _ = load_aggregated_data(complete_aggregated_data)
    blocks = build_block_structure(rows, "scenario_seed")
    
    # Should have 2 scenarios × 2 seeds = 4 blocks
    assert len(blocks) == 4
    assert 'gradual_drift_seed42' in blocks
    assert 'gradual_drift_seed123' in blocks
    
    # Each block should have 4 strategies
    for block_id, strategies in blocks.items():
        assert len(strategies) == 4


def test_build_block_structure_scenario(complete_aggregated_data):
    """Test building blocks with scenario definition."""
    rows, _, _, _ = load_aggregated_data(complete_aggregated_data)
    blocks = build_block_structure(rows, "scenario")
    
    # Should have 2 scenarios
    assert len(blocks) == 2
    assert 'gradual_drift' in blocks
    assert 'sudden_spike' in blocks


def test_build_block_structure_seed(complete_aggregated_data):
    """Test building blocks with seed definition."""
    rows, _, _, _ = load_aggregated_data(complete_aggregated_data)
    blocks = build_block_structure(rows, "seed")
    
    # Should have 2 seeds
    assert len(blocks) == 2
    assert 'seed42' in blocks
    assert 'seed123' in blocks


def test_validate_block_structure_complete(complete_aggregated_data):
    """Test validation passes with complete blocks."""
    rows, strategies, _, _ = load_aggregated_data(complete_aggregated_data)
    blocks = build_block_structure(rows, "scenario_seed")
    
    issues = validate_block_structure(blocks, strategies, strict=False)
    assert issues == []


def test_validate_block_structure_missing_strategy(temp_dir):
    """Test validation detects missing strategy."""
    csv_path = temp_dir / "incomplete.csv"
    
    rows = [
        {'strategy': 'static', 'scenario': 'drift', 'seed': '42', 'mae': '10.0'},
        {'strategy': 'proposed', 'scenario': 'drift', 'seed': '42', 'mae': '9.0'},
        # Missing 'scheduled' and 'naive_adaptive' in this block
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    data_rows, _, _, _ = load_aggregated_data(csv_path)
    blocks = build_block_structure(data_rows, "scenario_seed")
    
    expected_strategies = ['static', 'scheduled', 'naive_adaptive', 'proposed']
    issues = validate_block_structure(blocks, expected_strategies, strict=False)
    
    assert len(issues) > 0
    assert any('missing' in issue.lower() for issue in issues)


def test_validate_block_structure_strict_mode(temp_dir):
    """Test strict mode raises on incomplete blocks."""
    csv_path = temp_dir / "incomplete.csv"
    
    rows = [
        {'strategy': 'static', 'scenario': 'drift', 'seed': '42', 'mae': '10.0'},
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    data_rows, _, _, _ = load_aggregated_data(csv_path)
    blocks = build_block_structure(data_rows, "scenario_seed")
    
    expected_strategies = ['static', 'proposed']
    with pytest.raises(ValueError, match="Incomplete block structure"):
        validate_block_structure(blocks, expected_strategies, strict=True)


def test_extract_metric_matrix(complete_aggregated_data):
    """Test extracting metric matrix."""
    rows, strategies, _, _ = load_aggregated_data(complete_aggregated_data)
    blocks = build_block_structure(rows, "scenario_seed")
    
    matrix = extract_metric_matrix(blocks, 'mae', strategies)
    
    # Should be (4 blocks, 4 strategies)
    assert matrix.shape == (4, 4)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix > 0)  # MAE should be positive


def test_extract_metric_matrix_missing_value(temp_dir):
    """Test extraction fails on missing metric value."""
    csv_path = temp_dir / "missing_value.csv"
    
    rows = [
        {'strategy': 'static', 'scenario': 'drift', 'seed': '42', 'mae': '10.0'},
        {'strategy': 'proposed', 'scenario': 'drift', 'seed': '42', 'mae': ''},  # Missing
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    data_rows, strategies, _, _ = load_aggregated_data(csv_path)
    blocks = build_block_structure(data_rows, "scenario_seed")
    
    with pytest.raises(ValueError, match="Null value"):
        extract_metric_matrix(blocks, 'mae', strategies)


def test_extract_metric_matrix_nan_value(temp_dir):
    """Test extraction fails on NaN value."""
    csv_path = temp_dir / "nan_value.csv"
    
    rows = [
        {'strategy': 'static', 'scenario': 'drift', 'seed': '42', 'mae': '10.0'},
        {'strategy': 'proposed', 'scenario': 'drift', 'seed': '42', 'mae': 'nan'},
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    data_rows, strategies, _, _ = load_aggregated_data(csv_path)
    blocks = build_block_structure(data_rows, "scenario_seed")
    
    # Note: 'nan' string converts to float('nan'), check happens after conversion
    with pytest.raises(ValueError, match="NaN/Inf value"):
        extract_metric_matrix(blocks, 'mae', strategies)


def test_extract_metric_matrix_inf_value(temp_dir):
    """Test extraction detects inf value."""
    csv_path = temp_dir / "inf_value.csv"
    
    rows = [
        {'strategy': 'static', 'scenario': 'drift', 'seed': '42', 'mae': '10.0'},
        {'strategy': 'proposed', 'scenario': 'drift', 'seed': '42', 'mae': 'inf'},
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    data_rows, strategies, _, _ = load_aggregated_data(csv_path)
    blocks = build_block_structure(data_rows, "scenario_seed")
    
    # Note: float('inf') converts successfully, but we check for it
    # Actually the check happens after conversion
    with pytest.raises(ValueError, match="NaN/Inf value"):
        extract_metric_matrix(blocks, 'mae', strategies)


def test_friedman_test_sufficient_data():
    """Test Friedman test with sufficient data."""
    # Create synthetic data: 5 blocks, 3 strategies
    # Strategy 3 is consistently better
    data = np.array([
        [10.0, 10.5, 9.0],   # Block 1
        [11.0, 11.5, 10.0],  # Block 2
        [9.5, 10.0, 8.5],    # Block 3
        [10.5, 11.0, 9.5],   # Block 4
        [10.2, 10.7, 9.2],   # Block 5
    ])
    
    result = friedman_test(data, 'mae', alpha=0.05)
    
    assert result.metric == 'mae'
    assert result.n_blocks == 5
    assert result.n_strategies == 3
    assert result.degrees_of_freedom == 2
    assert result.statistic > 0
    assert 0 <= result.p_value <= 1


def test_friedman_test_insufficient_blocks():
    """Test Friedman test fails with insufficient blocks."""
    data = np.array([
        [10.0, 10.5, 9.0],  # Only 1 block
    ])
    
    with pytest.raises(ValueError, match="at least 2 blocks"):
        friedman_test(data, 'mae')


def test_friedman_test_insufficient_strategies():
    """Test Friedman test fails with insufficient strategies."""
    data = np.array([
        [10.0, 10.5],  # Only 2 strategies
        [11.0, 11.5],
    ])
    
    with pytest.raises(ValueError, match="at least 3 treatments"):
        friedman_test(data, 'mae')


def test_wilcoxon_signed_rank():
    """Test Wilcoxon signed-rank test."""
    # Strategy A consistently worse than strategy B
    values_a = np.array([10.0, 11.0, 9.5, 10.5, 10.2])
    values_b = np.array([9.0, 10.0, 8.5, 9.5, 9.2])
    
    result = wilcoxon_signed_rank(values_a, values_b, 'strategy_a', 'strategy_b', 'mae')
    
    assert result.metric == 'mae'
    assert result.strategy_a == 'strategy_a'
    assert result.strategy_b == 'strategy_b'
    assert result.n_pairs == 5
    assert result.statistic >= 0
    assert 0 <= result.p_value <= 1
    assert -1 <= result.effect_size <= 1


def test_wilcoxon_insufficient_pairs():
    """Test Wilcoxon test fails with insufficient pairs."""
    values_a = np.array([10.0, 11.0])
    values_b = np.array([9.0, 10.0])
    
    with pytest.raises(ValueError, match="at least 3 pairs"):
        wilcoxon_signed_rank(values_a, values_b, 'a', 'b', 'mae')


def test_wilcoxon_mismatched_lengths():
    """Test Wilcoxon test fails with mismatched arrays."""
    values_a = np.array([10.0, 11.0, 9.5])
    values_b = np.array([9.0, 10.0])
    
    with pytest.raises(ValueError, match="Mismatched array lengths"):
        wilcoxon_signed_rank(values_a, values_b, 'a', 'b', 'mae')


def test_holm_correction():
    """Test Holm-Bonferroni correction."""
    # Test with known p-values
    p_values = [0.001, 0.01, 0.03, 0.06]
    significant = holm_correction(p_values, alpha=0.05)
    
    # With Holm correction:
    # Test 1: 0.001 < 0.05/4 = 0.0125 ✓ significant
    # Test 2: 0.01 < 0.05/3 = 0.0167 ✓ significant
    # Test 3: 0.03 < 0.05/2 = 0.025 ✗ not significant
    # Test 4: not tested (stopped at test 3)
    
    assert significant == [True, True, False, False]


def test_holm_correction_empty():
    """Test Holm correction with empty list."""
    significant = holm_correction([], alpha=0.05)
    assert significant == []


def test_holm_correction_all_significant():
    """Test Holm correction when all tests are highly significant."""
    p_values = [0.0001, 0.0002, 0.0003]
    significant = holm_correction(p_values, alpha=0.05)
    assert significant == [True, True, True]


def test_holm_correction_none_significant():
    """Test Holm correction when no tests are significant."""
    p_values = [0.1, 0.2, 0.3]
    significant = holm_correction(p_values, alpha=0.05)
    assert significant == [False, False, False]


def test_perform_pairwise_tests():
    """Test pairwise comparisons with correction."""
    # 4 blocks, 3 strategies
    data = np.array([
        [10.0, 10.5, 9.0],
        [11.0, 11.5, 10.0],
        [9.5, 10.0, 8.5],
        [10.5, 11.0, 9.5],
    ])
    strategies = ['static', 'scheduled', 'proposed']
    
    results = perform_pairwise_tests(data, strategies, 'mae')
    
    # Should have 3 comparisons: (0,1), (0,2), (1,2)
    assert len(results) == 3
    
    # Check all results have corrected p-values
    for result in results:
        assert result.corrected_p_value is not None
        assert result.significant_corrected is not None
        assert result.metric == 'mae'


def test_analyze_metric(complete_aggregated_data):
    """Test complete metric analysis."""
    rows, strategies, _, _ = load_aggregated_data(complete_aggregated_data)
    blocks = build_block_structure(rows, "scenario_seed")
    
    friedman_result, pairwise_results = analyze_metric(blocks, strategies, 'mae')
    
    # Should have Friedman result (4 strategies)
    assert friedman_result is not None
    assert friedman_result.n_strategies == 4
    
    # Should have 6 pairwise comparisons
    assert len(pairwise_results) == 6


def test_analyze_metric_skip_friedman():
    """Test analysis skips Friedman with <3 strategies."""
    # Create data with only 2 strategies
    blocks = {
        'block1': {
            'static': {'mae': '10.0'},
            'proposed': {'mae': '9.0'}
        },
        'block2': {
            'static': {'mae': '11.0'},
            'proposed': {'mae': '10.0'}
        },
        'block3': {
            'static': {'mae': '9.5'},
            'proposed': {'mae': '8.5'}
        }
    }
    strategies = ['static', 'proposed']
    
    friedman_result, pairwise_results = analyze_metric(
        blocks, strategies, 'mae', skip_friedman=True
    )
    
    assert friedman_result is None
    assert len(pairwise_results) == 1  # Only 1 comparison


def test_run_statistical_analysis_complete(complete_aggregated_data, manifest_file, qc_passed, temp_dir):
    """Test full statistical analysis pipeline."""
    report = run_statistical_analysis(
        complete_aggregated_data,
        manifest_file,
        qc_passed,
        strict=False
    )
    
    assert report.total_runs == 16
    assert report.n_strategies == 4
    assert report.n_scenarios == 2
    assert report.n_seeds == 2
    assert report.n_blocks == 4
    
    # Should have Friedman tests for primary metrics
    assert 'mae' in report.friedman_tests
    assert 'rmse' in report.friedman_tests
    
    # Should have pairwise tests
    assert 'mae' in report.pairwise_tests
    assert len(report.pairwise_tests['mae']) == 6  # 4 strategies = 6 pairs


def test_run_statistical_analysis_qc_failed(complete_aggregated_data, manifest_file, qc_failed):
    """Test analysis fails when QC did not pass."""
    with pytest.raises(ValueError, match="QC did not pass"):
        run_statistical_analysis(
            complete_aggregated_data,
            manifest_file,
            qc_failed,
            strict=False
        )


def test_run_statistical_analysis_insufficient_strategies(temp_dir, manifest_file, qc_passed):
    """Test analysis warns with insufficient strategies."""
    csv_path = temp_dir / "insufficient.csv"
    
    # Only 1 strategy
    rows = [
        {'strategy': 'static', 'scenario': 'drift', 'seed': '42', 'mae': '10.0', 'rmse': '12.0',
         'detection_delay': '50', 'model_promoted_events': '3', 'total_adaptation_time': '100.5',
         'git_commit': 'abc', 'dataset_checksum': 'def', 'python_version': '3.12'},
        {'strategy': 'static', 'scenario': 'drift', 'seed': '123', 'mae': '11.0', 'rmse': '13.0',
         'detection_delay': '55', 'model_promoted_events': '2', 'total_adaptation_time': '90.5',
         'git_commit': 'abc', 'dataset_checksum': 'def', 'python_version': '3.12'},
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    report = run_statistical_analysis(csv_path, manifest_file, qc_passed, strict=False)
    
    assert len(report.warnings) > 0
    assert any('Insufficient strategies' in w for w in report.warnings)
    assert len(report.friedman_tests) == 0
    assert len(report.pairwise_tests) == 0


def test_run_statistical_analysis_insufficient_blocks(temp_dir, manifest_file, qc_passed):
    """Test analysis warns with insufficient blocks."""
    csv_path = temp_dir / "insufficient_blocks.csv"
    
    # Only 1 block (scenario_seed combination)
    rows = [
        {'strategy': 'static', 'scenario': 'drift', 'seed': '42', 'mae': '10.0', 'rmse': '12.0',
         'detection_delay': '50', 'model_promoted_events': '3', 'total_adaptation_time': '100.5',
         'git_commit': 'abc', 'dataset_checksum': 'def', 'python_version': '3.12'},
        {'strategy': 'proposed', 'scenario': 'drift', 'seed': '42', 'mae': '9.0', 'rmse': '11.0',
         'detection_delay': '45', 'model_promoted_events': '4', 'total_adaptation_time': '105.5',
         'git_commit': 'abc', 'dataset_checksum': 'def', 'python_version': '3.12'},
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    report = run_statistical_analysis(csv_path, manifest_file, qc_passed, strict=False)
    
    assert len(report.warnings) > 0
    assert any('Insufficient blocks' in w for w in report.warnings)


def test_run_statistical_analysis_strict_mode_fails(temp_dir, manifest_file, qc_passed):
    """Test strict mode fails with insufficient data."""
    csv_path = temp_dir / "insufficient.csv"
    
    rows = [
        {'strategy': 'static', 'scenario': 'drift', 'seed': '42', 'mae': '10.0', 'rmse': '12.0',
         'detection_delay': '50', 'model_promoted_events': '3', 'total_adaptation_time': '100.5',
         'git_commit': 'abc', 'dataset_checksum': 'def', 'python_version': '3.12'},
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    with pytest.raises(ValueError, match="Insufficient"):
        run_statistical_analysis(csv_path, manifest_file, qc_passed, strict=True)


def test_deterministic_execution(complete_aggregated_data, manifest_file, qc_passed):
    """Test that repeated execution produces identical results."""
    report1 = run_statistical_analysis(
        complete_aggregated_data, manifest_file, qc_passed, strict=False
    )
    
    report2 = run_statistical_analysis(
        complete_aggregated_data, manifest_file, qc_passed, strict=False
    )
    
    # Compare Friedman results
    for metric in report1.friedman_tests:
        assert metric in report2.friedman_tests
        r1 = report1.friedman_tests[metric]
        r2 = report2.friedman_tests[metric]
        # Handle NaN comparisons (can occur with tied ranks)
        if np.isnan(r1.statistic):
            assert np.isnan(r2.statistic)
        else:
            assert r1.statistic == r2.statistic
        if np.isnan(r1.p_value):
            assert np.isnan(r2.p_value)
        else:
            assert r1.p_value == r2.p_value
        assert r1.significant == r2.significant
    
    # Compare pairwise results
    for metric in report1.pairwise_tests:
        assert metric in report2.pairwise_tests
        results1 = report1.pairwise_tests[metric]
        results2 = report2.pairwise_tests[metric]
        assert len(results1) == len(results2)
        
        for r1, r2 in zip(results1, results2):
            assert r1.strategy_a == r2.strategy_a
            assert r1.strategy_b == r2.strategy_b
            # Handle NaN comparisons
            if np.isnan(r1.statistic):
                assert np.isnan(r2.statistic)
            else:
                assert r1.statistic == r2.statistic
            if np.isnan(r1.p_value):
                assert np.isnan(r2.p_value)
            else:
                assert r1.p_value == r2.p_value
            if np.isnan(r1.effect_size):
                assert np.isnan(r2.effect_size)
            else:
                assert r1.effect_size == r2.effect_size
            # Corrected p-value might be None or NaN
            if r1.corrected_p_value is None:
                assert r2.corrected_p_value is None
            elif np.isnan(r1.corrected_p_value):
                assert np.isnan(r2.corrected_p_value)
            else:
                assert r1.corrected_p_value == r2.corrected_p_value

