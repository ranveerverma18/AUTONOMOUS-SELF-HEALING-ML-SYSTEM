#!/usr/bin/env python3
"""
Statistical Analysis Tool

Performs statistical tests on QC-approved aggregated experiment results.
Implements Friedman test, pairwise Wilcoxon tests, Holm correction, and effect sizes.

Usage:
    python -m scripts.analysis.statistical_analysis \
        --aggregated <aggregated_results.csv> \
        --manifest <experiment_manifest.csv> \
        --qc-report <qc_report.json> \
        [--output <output_dir>] \
        [--strict]

Exit codes:
    0 = analysis succeeded
    1 = analysis failed or insufficient data
    2 = invalid CLI/input/configuration
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import math

import numpy as np
from scipy import stats


# Primary metrics for statistical analysis
PRIMARY_METRICS = {
    'mae': 'Mean Absolute Error (primary performance metric)',
    'rmse': 'Root Mean Squared Error (primary performance metric)'
}

# Secondary metrics for statistical analysis
SECONDARY_METRICS = {
    'detection_delay': 'Samples from scenario onset to first detection',
    'model_promoted_events': 'Number of successful model promotions',
    'total_adaptation_time': 'Total time spent in adaptation (retraining + shadow)'
}

# Descriptive metrics (not statistically tested)
DESCRIPTIVE_METRICS = {
    'drift_detections', 'anomaly_detections', 'false_positive_triggers',
    'retraining_events', 'validation_skipped_events', 'candidates_generated',
    'gate_accepts', 'gate_rejects', 'shadow_promotions', 'shadow_rejections',
    'degraded_promotions', 'degraded_promotion_rate',
    'time_to_first_error_recovery', 'time_to_sustained_recovery',
    'total_retraining_time', 'total_shadow_evaluation_time', 'mean_inference_latency'
}


@dataclass
class FriedmanResult:
    """Friedman omnibus test result."""
    metric: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    n_blocks: int
    n_strategies: int
    significant: bool
    alpha: float = 0.05


@dataclass
class WilcoxonResult:
    """Pairwise Wilcoxon signed-rank test result."""
    metric: str
    strategy_a: str
    strategy_b: str
    statistic: float
    p_value: float
    n_pairs: int
    effect_size: float  # rank-biserial correlation
    significant_uncorrected: bool
    corrected_p_value: Optional[float] = None
    significant_corrected: Optional[bool] = None
    alpha: float = 0.05


@dataclass
class StatisticalAnalysisReport:
    """Complete statistical analysis report."""
    aggregated_file: str
    manifest_file: str
    qc_report_file: str
    timestamp: str
    
    # Data summary
    total_runs: int
    strategies: List[str]
    scenarios: List[str]
    seeds: List[int]
    n_strategies: int
    n_scenarios: int
    n_seeds: int
    
    # Block structure
    block_definition: str
    n_blocks: int
    
    # Test results
    friedman_tests: Dict[str, FriedmanResult] = field(default_factory=dict)
    pairwise_tests: Dict[str, List[WilcoxonResult]] = field(default_factory=dict)
    
    # Metadata
    git_commit: str = ""
    dataset_checksum: str = ""
    python_version: str = ""
    scipy_version: str = ""
    numpy_version: str = ""
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)


def load_aggregated_data(csv_path: Path) -> Tuple[List[Dict], List[str], List[str], List[int]]:
    """Load aggregated CSV and extract metadata."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    strategies = sorted(set(row['strategy'] for row in rows))
    scenarios = sorted(set(row['scenario'] for row in rows))
    seeds = sorted(set(int(row['seed']) for row in rows))
    
    return rows, strategies, scenarios, seeds


def load_qc_report(json_path: Path) -> Dict:
    """Load QC report JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_qc_passed(qc_report: Dict):
    """Ensure QC passed before proceeding."""
    if not qc_report.get('passed', False):
        raise ValueError("QC did not pass - cannot perform statistical analysis on invalid data")


def build_block_structure(rows: List[Dict], block_def: str = "scenario_seed") -> Dict:
    """
    Build matched blocks for statistical analysis.
    
    Block definition:
    - "scenario_seed": Each (scenario, seed) pair is a block
    - "scenario": Each scenario is a block (averaged over seeds)
    - "seed": Each seed is a block (averaged over scenarios)
    
    Returns: dict mapping block_id -> dict mapping strategy -> metric values
    """
    blocks = {}
    
    for row in rows:
        strategy = row['strategy']
        scenario = row['scenario']
        seed = row['seed']
        
        if block_def == "scenario_seed":
            block_id = f"{scenario}_seed{seed}"
        elif block_def == "scenario":
            block_id = scenario
        elif block_def == "seed":
            block_id = f"seed{seed}"
        else:
            raise ValueError(f"Unknown block definition: {block_def}")
        
        if block_id not in blocks:
            blocks[block_id] = {}
        
        blocks[block_id][strategy] = row
    
    return blocks


def validate_block_structure(blocks: Dict, expected_strategies: List[str], strict: bool = False):
    """Validate that all blocks have all strategies (complete block design)."""
    issues = []
    
    for block_id, strategies in blocks.items():
        missing = set(expected_strategies) - set(strategies.keys())
        if missing:
            issues.append(f"Block '{block_id}' missing strategies: {sorted(missing)}")
        
        extra = set(strategies.keys()) - set(expected_strategies)
        if extra:
            issues.append(f"Block '{block_id}' has unexpected strategies: {sorted(extra)}")
    
    if issues:
        message = "Incomplete block structure:\n" + "\n".join(f"  - {i}" for i in issues)
        if strict:
            raise ValueError(message)
        else:
            return issues
    
    return []


def extract_metric_matrix(blocks: Dict, metric: str, strategies: List[str]) -> np.ndarray:
    """
    Extract metric values as a matrix for statistical testing.
    
    Returns: np.ndarray of shape (n_blocks, n_strategies)
             Rows = blocks, Columns = strategies (sorted order)
    
    Raises ValueError if metric is missing or invalid.
    """
    matrix = []
    
    for block_id in sorted(blocks.keys()):
        row_values = []
        for strategy in strategies:
            if strategy not in blocks[block_id]:
                raise ValueError(f"Missing strategy '{strategy}' in block '{block_id}'")
            
            value_str = blocks[block_id][strategy].get(metric, '')
            
            # Handle null values
            if value_str == '' or value_str.lower() in ('none', 'null', 'na'):
                raise ValueError(f"Null value for metric '{metric}' in block '{block_id}', strategy '{strategy}'")
            
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric value '{value_str}' for metric '{metric}' in block '{block_id}', strategy '{strategy}'")
            
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"NaN/Inf value for metric '{metric}' in block '{block_id}', strategy '{strategy}'")
            
            row_values.append(value)
        
        matrix.append(row_values)
    
    return np.array(matrix)


def friedman_test(data_matrix: np.ndarray, metric: str, alpha: float = 0.05) -> FriedmanResult:
    """
    Perform Friedman omnibus test.
    
    H0: All strategies have the same distribution
    H1: At least one strategy differs
    
    Args:
        data_matrix: Shape (n_blocks, n_strategies)
        metric: Name of metric being tested
        alpha: Significance level
    
    Returns: FriedmanResult
    """
    n_blocks, n_strategies = data_matrix.shape
    
    if n_blocks < 2:
        raise ValueError(f"Friedman test requires at least 2 blocks, got {n_blocks}")
    
    if n_strategies < 3:
        raise ValueError(f"Friedman test requires at least 3 treatments, got {n_strategies}")
    
    # Perform test
    statistic, p_value = stats.friedmanchisquare(*data_matrix.T)
    
    return FriedmanResult(
        metric=metric,
        statistic=float(statistic),
        p_value=float(p_value),
        degrees_of_freedom=n_strategies - 1,
        n_blocks=n_blocks,
        n_strategies=n_strategies,
        significant=(p_value < alpha),
        alpha=alpha
    )


def wilcoxon_signed_rank(values_a: np.ndarray, values_b: np.ndarray,
                         strategy_a: str, strategy_b: str,
                         metric: str, alpha: float = 0.05) -> WilcoxonResult:
    """
    Perform pairwise Wilcoxon signed-rank test.
    
    H0: Paired differences have zero median
    H1: Paired differences do not have zero median
    
    Args:
        values_a: Array of values for strategy A (one per block)
        values_b: Array of values for strategy B (matched blocks)
        strategy_a: Name of strategy A
        strategy_b: Name of strategy B
        metric: Name of metric being tested
        alpha: Significance level
    
    Returns: WilcoxonResult
    """
    if len(values_a) != len(values_b):
        raise ValueError(f"Mismatched array lengths: {len(values_a)} vs {len(values_b)}")
    
    n_pairs = len(values_a)
    
    if n_pairs < 3:
        raise ValueError(f"Wilcoxon test requires at least 3 pairs, got {n_pairs}")
    
    # Perform test (two-sided by default)
    statistic, p_value = stats.wilcoxon(values_a, values_b, alternative='two-sided')
    
    # Calculate rank-biserial correlation as effect size
    # r = 1 - (2*W)/(n*(n+1)) where W is the smaller of W+ and W-
    # Simplified: r = (R+ - R-) / (n*(n+1)/2) = Z / sqrt(n)
    # Using more direct formula: r = 1 - (4*W)/(n*(n+1))
    # where W is the test statistic (smaller sum of ranks)
    
    # For matched pairs, compute effect size
    differences = values_a - values_b
    # Count ties at zero (they're dropped in Wilcoxon)
    non_zero_diffs = differences[differences != 0]
    n_effective = len(non_zero_diffs)
    
    if n_effective > 0:
        # Rank-biserial correlation approximation
        # Using formula: r = Z / sqrt(n) where Z is the z-score
        # For small samples, we use the direct rank computation
        ranks = stats.rankdata(np.abs(non_zero_diffs))
        signs = np.sign(non_zero_diffs)
        r_plus = np.sum(ranks[signs > 0])
        r_minus = np.sum(ranks[signs < 0])
        effect_size = (r_plus - r_minus) / (n_effective * (n_effective + 1) / 2)
    else:
        effect_size = 0.0
    
    return WilcoxonResult(
        metric=metric,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        statistic=float(statistic),
        p_value=float(p_value),
        n_pairs=n_pairs,
        effect_size=float(effect_size),
        significant_uncorrected=(p_value < alpha),
        alpha=alpha
    )


def holm_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of uncorrected p-values
        alpha: Family-wise error rate
    
    Returns: List of booleans indicating significance after correction
    """
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    
    # Apply Holm procedure
    significant = [False] * n
    for rank, (original_idx, p_val) in enumerate(indexed):
        # Adjusted alpha for this rank: alpha / (n - rank)
        adjusted_alpha = alpha / (n - rank)
        if p_val < adjusted_alpha:
            significant[original_idx] = True
        else:
            # Once we fail to reject, all subsequent tests also fail
            break
    
    return significant


def perform_pairwise_tests(data_matrix: np.ndarray, strategies: List[str],
                           metric: str, alpha: float = 0.05) -> List[WilcoxonResult]:
    """Perform all pairwise Wilcoxon tests with Holm correction."""
    results = []
    
    n_strategies = len(strategies)
    for i in range(n_strategies):
        for j in range(i + 1, n_strategies):
            result = wilcoxon_signed_rank(
                data_matrix[:, i],
                data_matrix[:, j],
                strategies[i],
                strategies[j],
                metric,
                alpha
            )
            results.append(result)
    
    # Apply Holm correction
    p_values = [r.p_value for r in results]
    corrected_significant = holm_correction(p_values, alpha)
    
    # Compute corrected p-values (for reporting)
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected_p = [None] * n
    for rank, (original_idx, p_val) in enumerate(indexed):
        # Corrected p-value is min(1, p * (n - rank))
        corrected_p[original_idx] = min(1.0, p_val * (n - rank))
    
    # Update results
    for i, result in enumerate(results):
        result.corrected_p_value = corrected_p[i]
        # Significance is based on corrected p-value, not Holm step-down
        if corrected_p[i] is not None and not np.isnan(corrected_p[i]):
            result.significant_corrected = corrected_p[i] < alpha
        else:
            result.significant_corrected = False
    
    return results


def analyze_metric(blocks: Dict, strategies: List[str], metric: str,
                   alpha: float = 0.05, skip_friedman: bool = False) -> Tuple[Optional[FriedmanResult], List[WilcoxonResult]]:
    """
    Perform complete statistical analysis for one metric.
    
    Returns: (friedman_result, pairwise_results)
    """
    # Extract data matrix
    data_matrix = extract_metric_matrix(blocks, metric, strategies)
    
    # Friedman omnibus test (if applicable)
    friedman_result = None
    if not skip_friedman and len(strategies) >= 3:
        friedman_result = friedman_test(data_matrix, metric, alpha)
    
    # Pairwise Wilcoxon tests
    pairwise_results = perform_pairwise_tests(data_matrix, strategies, metric, alpha)
    
    return friedman_result, pairwise_results


def run_statistical_analysis(aggregated_file: Path, manifest_file: Path,
                             qc_report_file: Path, strict: bool = False,
                             block_def: str = "scenario_seed") -> StatisticalAnalysisReport:
    """Run complete statistical analysis pipeline."""
    
    # Load data
    rows, strategies, scenarios, seeds = load_aggregated_data(aggregated_file)
    qc_report = load_qc_report(qc_report_file)
    
    # Validate QC passed
    validate_qc_passed(qc_report)
    
    # Extract provenance
    # Try to get git commit from repository (navigate to workspace root first)
    workspace_root = aggregated_file.parent.parent.parent if 'experiments' in str(aggregated_file) else aggregated_file.parent.parent
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd=workspace_root)
        git_commit = result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        git_commit = 'unknown'
    
    # Try to get dataset checksum
    # Navigate from aggregated file: experiments/results/... -> workspace root
    workspace_root = aggregated_file.parent.parent.parent if 'experiments' in str(aggregated_file) else aggregated_file.parent.parent
    dataset_file = workspace_root / 'dataset' / 'raw' / 'train_FD001.txt'
    if dataset_file.exists():
        try:
            import hashlib
            with open(dataset_file, 'rb') as f:
                dataset_checksum = hashlib.md5(f.read()).hexdigest()
        except Exception:
            dataset_checksum = 'unknown'
    else:
        dataset_checksum = 'unknown'
    
    # Get Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Get library versions
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = 'unknown'
    numpy_version = np.__version__
    
    # Build report
    report = StatisticalAnalysisReport(
        aggregated_file=str(aggregated_file),
        manifest_file=str(manifest_file),
        qc_report_file=str(qc_report_file),
        timestamp="",  # Will be filled when writing (deterministic in data)
        total_runs=len(rows),
        strategies=strategies,
        scenarios=scenarios,
        seeds=seeds,
        n_strategies=len(strategies),
        n_scenarios=len(scenarios),
        n_seeds=len(seeds),
        block_definition=block_def,
        n_blocks=0,  # Will be computed
        git_commit=git_commit,
        dataset_checksum=dataset_checksum,
        python_version=python_version,
        scipy_version=scipy_version,
        numpy_version=numpy_version
    )
    
    # Build blocks
    blocks = build_block_structure(rows, block_def)
    report.n_blocks = len(blocks)
    
    # Validate block structure
    block_issues = validate_block_structure(blocks, strategies, strict=strict)
    if block_issues:
        for issue in block_issues:
            report.add_warning(issue)
        if strict:
            raise ValueError("Incomplete block structure in strict mode")
    
    # Check if we have enough data
    if report.n_strategies < 2:
        report.add_warning(f"Insufficient strategies for statistical analysis: {report.n_strategies} < 2")
        if strict:
            raise ValueError("Insufficient strategies for pairwise comparisons")
        return report
    
    if report.n_blocks < 2:
        report.add_warning(f"Insufficient blocks for statistical analysis: {report.n_blocks} < 2")
        if strict:
            raise ValueError("Insufficient blocks for paired tests")
        return report
    
    # Perform analysis for each metric
    all_metrics = {**PRIMARY_METRICS, **SECONDARY_METRICS}
    
    for metric in all_metrics.keys():
        try:
            skip_friedman = report.n_strategies < 3
            friedman_result, pairwise_results = analyze_metric(
                blocks, strategies, metric, skip_friedman=skip_friedman
            )
            
            if friedman_result:
                report.friedman_tests[metric] = friedman_result
            
            report.pairwise_tests[metric] = pairwise_results
        
        except ValueError as e:
            report.add_warning(f"Skipping metric '{metric}': {str(e)}")
            continue
        except Exception as e:
            report.add_warning(f"Error analyzing metric '{metric}': {str(e)}")
            continue
    
    return report


def write_report_json(report: StatisticalAnalysisReport, output_path: Path):
    """Write machine-readable JSON report."""
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    data = {
        "aggregated_file": report.aggregated_file,
        "manifest_file": report.manifest_file,
        "qc_report_file": report.qc_report_file,
        "total_runs": report.total_runs,
        "strategies": report.strategies,
        "scenarios": report.scenarios,
        "seeds": report.seeds,
        "n_strategies": report.n_strategies,
        "n_scenarios": report.n_scenarios,
        "n_seeds": report.n_seeds,
        "block_definition": report.block_definition,
        "n_blocks": report.n_blocks,
        "git_commit": report.git_commit,
        "dataset_checksum": report.dataset_checksum,
        "python_version": report.python_version,
        "scipy_version": report.scipy_version,
        "numpy_version": report.numpy_version,
        "warnings": report.warnings,
        "friedman_tests": {
            metric: convert_numpy_types(asdict(result))
            for metric, result in report.friedman_tests.items()
        },
        "pairwise_tests": {
            metric: [convert_numpy_types(asdict(r)) for r in results]
            for metric, results in report.pairwise_tests.items()
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def write_report_text(report: StatisticalAnalysisReport, output_path: Path):
    """Write human-readable text report."""
    lines = [
        "# Statistical Analysis Report",
        "",
        f"**Aggregated File:** `{report.aggregated_file}`",
        f"**Manifest File:** `{report.manifest_file}`",
        f"**QC Report:** `{report.qc_report_file}`",
        "",
        "## Data Summary",
        "",
        f"- **Total Runs:** {report.total_runs}",
        f"- **Strategies ({report.n_strategies}):** {', '.join(report.strategies)}",
        f"- **Scenarios ({report.n_scenarios}):** {', '.join(report.scenarios)}",
        f"- **Seeds ({report.n_seeds}):** {', '.join(map(str, report.seeds))}",
        f"- **Block Definition:** `{report.block_definition}`",
        f"- **Number of Blocks:** {report.n_blocks}",
        "",
        "## Provenance",
        "",
        f"- **Git Commit:** `{report.git_commit}`",
        f"- **Dataset Checksum:** `{report.dataset_checksum}`",
        f"- **Python:** {report.python_version}",
        f"- **SciPy:** {report.scipy_version}",
        f"- **NumPy:** {report.numpy_version}",
        ""
    ]
    
    if report.warnings:
        lines.extend([
            "## Warnings",
            ""
        ])
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")
    
    # Friedman tests
    if report.friedman_tests:
        lines.extend([
            "## Friedman Omnibus Tests",
            "",
            "| Metric | χ² | df | p-value | Significant (α=0.05) |",
            "|--------|----|----|---------|----------------------|"
        ])
        for metric, result in report.friedman_tests.items():
            sig = "✅ Yes" if result.significant else "❌ No"
            lines.append(f"| {metric} | {result.statistic:.4f} | {result.degrees_of_freedom} | {result.p_value:.6f} | {sig} |")
        lines.append("")
    
    # Pairwise tests
    if report.pairwise_tests:
        lines.extend([
            "## Pairwise Wilcoxon Tests (Holm-Corrected)",
            ""
        ])
        for metric in sorted(report.pairwise_tests.keys()):
            results = report.pairwise_tests[metric]
            lines.append(f"### {metric}")
            lines.append("")
            lines.append("| Comparison | n | W | p-value | p-corrected | Effect Size (r) | Significant |")
            lines.append("|------------|---|---|---------|-------------|-----------------|-------------|")
            for r in results:
                comp = f"{r.strategy_a} vs {r.strategy_b}"
                sig = "✅" if r.significant_corrected else "❌"
                p_corr = f"{r.corrected_p_value:.6f}" if r.corrected_p_value is not None else "N/A"
                lines.append(f"| {comp} | {r.n_pairs} | {r.statistic:.2f} | {r.p_value:.6f} | {p_corr} | {r.effect_size:.4f} | {sig} |")
            lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of aggregated experiment results"
    )
    parser.add_argument(
        "--aggregated",
        type=Path,
        required=True,
        help="Path to aggregated results CSV"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to experiment manifest CSV"
    )
    parser.add_argument(
        "--qc-report",
        type=Path,
        required=True,
        help="Path to QC report JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as aggregated file)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on insufficient data or incomplete blocks"
    )
    parser.add_argument(
        "--block-def",
        choices=["scenario_seed", "scenario", "seed"],
        default="scenario_seed",
        help="Block definition for matched pairs (default: scenario_seed)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.aggregated.exists():
        print(f"ERROR: Aggregated file not found: {args.aggregated}", file=sys.stderr)
        return 2
    
    if not args.manifest.exists():
        print(f"ERROR: Manifest file not found: {args.manifest}", file=sys.stderr)
        return 2
    
    if not args.qc_report.exists():
        print(f"ERROR: QC report not found: {args.qc_report}", file=sys.stderr)
        return 2
    
    # Set default output
    if args.output is None:
        args.output = args.aggregated.parent
    else:
        args.output.mkdir(parents=True, exist_ok=True)
    
    json_output = args.output / "statistical_analysis.json"
    text_output = args.output / "statistical_analysis.md"
    
    try:
        # Run analysis
        print(f"Running statistical analysis...")
        print(f"  Aggregated: {args.aggregated}")
        print(f"  Manifest: {args.manifest}")
        print(f"  QC Report: {args.qc_report}")
        print(f"  Block Definition: {args.block_def}")
        print(f"  Strict Mode: {args.strict}")
        
        report = run_statistical_analysis(
            args.aggregated,
            args.manifest,
            args.qc_report,
            strict=args.strict,
            block_def=args.block_def
        )
        
        # Write reports
        write_report_json(report, json_output)
        write_report_text(report, text_output)
        
        print(f"\n{'='*60}")
        print("STATISTICAL ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total Runs:      {report.total_runs}")
        print(f"Strategies:      {report.n_strategies} ({', '.join(report.strategies)})")
        print(f"Scenarios:       {report.n_scenarios}")
        print(f"Seeds:           {report.n_seeds}")
        print(f"Blocks:          {report.n_blocks}")
        print(f"Friedman Tests:  {len(report.friedman_tests)}")
        print(f"Pairwise Tests:  {sum(len(tests) for tests in report.pairwise_tests.values())}")
        print(f"Warnings:        {len(report.warnings)}")
        
        if report.warnings:
            print(f"\nWarnings:")
            for warning in report.warnings:
                print(f"  - {warning}")
        
        print(f"\nReports saved:")
        print(f"  - {json_output}")
        print(f"  - {text_output}")
        
        # Exit code
        if report.warnings and args.strict:
            return 1
        
        return 0
    
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nERROR: Analysis failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
