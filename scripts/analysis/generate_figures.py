"""
Generate publication-quality figures from frozen experimental results.

This script generates figures from statistical analysis outputs WITHOUT
executing any experiments or modifying experimental data.

Usage:
    python -m scripts.analysis.generate_figures \\
        --aggregated experiments/results/aggregated_results.csv \\
        --statistical experiments/results/statistical_analysis.json \\
        --output experiments/results/figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import warnings

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for publication quality
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Strategy display names and colors
STRATEGY_NAMES = {
    'static': 'Static',
    'scheduled': 'Scheduled',
    'naive_adaptive': 'Naive Adaptive',
    'proposed': 'Proposed'
}

STRATEGY_COLORS = {
    'static': '#e74c3c',
    'scheduled': '#f39c12',
    'naive_adaptive': '#3498db',
    'proposed': '#2ecc71'
}

STRATEGY_ORDER = ['static', 'scheduled', 'naive_adaptive', 'proposed']


def load_data(aggregated_path: Path, statistical_path: Path) -> tuple[pd.DataFrame, Dict]:
    """Load aggregated results and statistical analysis."""
    print(f"Loading aggregated results from {aggregated_path}")
    df = pd.read_csv(aggregated_path)
    
    print(f"Loading statistical analysis from {statistical_path}")
    with open(statistical_path, 'r') as f:
        stats = json.load(f)
    
    return df, stats


def generate_performance_boxplots(df: pd.DataFrame, output_dir: Path):
    """Generate boxplots for MAE and RMSE."""
    metrics = ['mae', 'rmse']
    metric_labels = {'mae': 'Mean Absolute Error (MAE)', 'rmse': 'Root Mean Squared Error (RMSE)'}
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Prepare data
        plot_data = []
        for strategy in STRATEGY_ORDER:
            strategy_data = df[df['strategy'] == strategy][metric].values
            plot_data.append(strategy_data)
        
        # Create boxplot
        bp = ax.boxplot(plot_data, labels=[STRATEGY_NAMES[s] for s in STRATEGY_ORDER],
                        patch_artist=True, widths=0.6)
        
        # Color boxes
        for patch, strategy in zip(bp['boxes'], STRATEGY_ORDER):
            patch.set_facecolor(STRATEGY_COLORS[strategy])
            patch.set_alpha(0.7)
        
        # Style
        ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_labels[metric]} by Strategy', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f'{metric}_boxplot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: {output_file.name}")


def generate_adaptation_metrics_plot(df: pd.DataFrame, output_dir: Path):
    """Generate combined adaptation metrics plot."""
    metrics = ['detection_delay', 'model_promoted_events', 'total_adaptation_time']
    metric_labels = {
        'detection_delay': 'Detection Delay (cycles)',
        'model_promoted_events': 'Model Promotions',
        'total_adaptation_time': 'Total Adaptation Time (cycles)'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data
        plot_data = []
        for strategy in STRATEGY_ORDER:
            strategy_data = df[df['strategy'] == strategy][metric].values
            plot_data.append(strategy_data)
        
        # Create boxplot
        bp = ax.boxplot(plot_data, labels=[STRATEGY_NAMES[s] for s in STRATEGY_ORDER],
                        patch_artist=True, widths=0.6)
        
        # Color boxes
        for patch, strategy in zip(bp['boxes'], STRATEGY_ORDER):
            patch.set_facecolor(STRATEGY_COLORS[strategy])
            patch.set_alpha(0.7)
        
        # Style
        ax.set_ylabel(metric_labels[metric], fontsize=10, fontweight='bold')
        ax.set_xlabel('Strategy', fontsize=10, fontweight='bold')
        ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_file = output_dir / 'adaptation_metrics_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Generated: {output_file.name}")


def generate_strategy_comparison_barplot(df: pd.DataFrame, output_dir: Path):
    """Generate mean performance comparison bar plot."""
    metrics = ['mae', 'rmse']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Calculate means and std errors
        means = []
        errors = []
        for strategy in STRATEGY_ORDER:
            strategy_data = df[df['strategy'] == strategy][metric].values
            means.append(np.mean(strategy_data))
            errors.append(np.std(strategy_data) / np.sqrt(len(strategy_data)))
        
        # Create bar plot
        x_pos = np.arange(len(STRATEGY_ORDER))
        bars = ax.bar(x_pos, means, yerr=errors, capsize=5, 
                      color=[STRATEGY_COLORS[s] for s in STRATEGY_ORDER],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Style
        ax.set_xticks(x_pos)
        ax.set_xticklabels([STRATEGY_NAMES[s] for s in STRATEGY_ORDER], rotation=45, ha='right')
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_xlabel('Strategy', fontsize=11, fontweight='bold')
        ax.set_title(f'Mean {metric.upper()} ± SEM', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'strategy_comparison_barplot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Generated: {output_file.name}")


def generate_pairwise_significance_heatmap(stats: Dict, output_dir: Path):
    """Generate heatmap of pairwise significance (Holm-corrected p-values)."""
    metrics = ['mae', 'rmse', 'detection_delay', 'model_promoted_events', 'total_adaptation_time']
    metric_labels = {
        'mae': 'MAE',
        'rmse': 'RMSE',
        'detection_delay': 'Detection Delay',
        'model_promoted_events': 'Model Promotions',
        'total_adaptation_time': 'Adaptation Time'
    }
    
    # Build significance matrix
    strategy_pairs = []
    sig_matrix = []
    
    for metric in metrics:
        if metric not in stats['pairwise_tests']:
            continue
        
        pairwise = stats['pairwise_tests'][metric]
        if not pairwise:
            continue
        
        # Extract pairs for first metric (same for all)
        if not strategy_pairs:
            for test in pairwise:
                comp = f"{test['strategy_a']} vs {test['strategy_b']}"
                strategy_pairs.append(comp.replace(' vs ', '\nvs\n'))
        
        # Extract corrected p-values
        row = []
        for test in pairwise:
            p_corrected = test['corrected_p_value']
            if p_corrected is None or np.isnan(p_corrected):
                row.append(1.0)  # Non-significant
            else:
                row.append(p_corrected)
        sig_matrix.append(row)
    
    if not sig_matrix:
        print("  WARNING: No pairwise test data available for heatmap")
        return
    
    # Convert to -log10(p) for better visualization
    sig_matrix_log = []
    for row in sig_matrix:
        log_row = []
        for p in row:
            if p == 0 or p < 1e-10:
                log_row.append(10.0)  # Cap at 10
            else:
                log_row.append(min(-np.log10(p), 10.0))
        sig_matrix_log.append(log_row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(sig_matrix_log, 
                xticklabels=strategy_pairs,
                yticklabels=[metric_labels.get(m, m) for m in metrics if m in stats['pairwise_tests']],
                cmap='RdYlGn_r', vmin=0, vmax=10,
                annot=False, fmt='.2f', linewidths=0.5,
                cbar_kws={'label': '-log₁₀(p-value, Holm-corrected)'},
                ax=ax)
    
    ax.set_title('Pairwise Significance (Holm-Corrected)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Strategy Comparison', fontsize=11, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=11, fontweight='bold')
    
    # Add significance threshold line
    ax.axhline(y=0, color='black', linewidth=2)
    
    plt.tight_layout()
    output_file = output_dir / 'pairwise_significance_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Generated: {output_file.name}")


def generate_effect_size_heatmap(stats: Dict, output_dir: Path):
    """Generate heatmap of effect sizes (rank-biserial correlation)."""
    metrics = ['mae', 'rmse', 'detection_delay', 'model_promoted_events', 'total_adaptation_time']
    metric_labels = {
        'mae': 'MAE',
        'rmse': 'RMSE',
        'detection_delay': 'Detection Delay',
        'model_promoted_events': 'Model Promotions',
        'total_adaptation_time': 'Adaptation Time'
    }
    
    # Build effect size matrix
    strategy_pairs = []
    effect_matrix = []
    
    for metric in metrics:
        if metric not in stats['pairwise_tests']:
            continue
        
        pairwise = stats['pairwise_tests'][metric]
        if not pairwise:
            continue
        
        # Extract pairs
        if not strategy_pairs:
            for test in pairwise:
                comp = f"{test['strategy_a']} vs {test['strategy_b']}"
                strategy_pairs.append(comp.replace(' vs ', '\nvs\n'))
        
        # Extract effect sizes
        row = []
        for test in pairwise:
            effect_size = test.get('effect_size', 0.0)
            if effect_size is None or np.isnan(effect_size):
                row.append(0.0)
            else:
                row.append(effect_size)
        effect_matrix.append(row)
    
    if not effect_matrix:
        print("  WARNING: No effect size data available for heatmap")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(effect_matrix,
                xticklabels=strategy_pairs,
                yticklabels=[metric_labels.get(m, m) for m in metrics if m in stats['pairwise_tests']],
                cmap='RdBu_r', vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', linewidths=0.5,
                cbar_kws={'label': 'Rank-Biserial Correlation (r)'},
                ax=ax)
    
    ax.set_title('Effect Sizes (Rank-Biserial Correlation)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Strategy Comparison', fontsize=11, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'effect_size_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Generated: {output_file.name}")


def generate_scenario_performance_plot(df: pd.DataFrame, output_dir: Path):
    """Generate performance by scenario for primary metrics."""
    metrics = ['mae', 'rmse']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data by scenario
        scenarios = sorted(df['scenario'].unique())
        x = np.arange(len(scenarios))
        width = 0.2
        
        for idx, strategy in enumerate(STRATEGY_ORDER):
            strategy_means = []
            strategy_errors = []
            
            for scenario in scenarios:
                data = df[(df['strategy'] == strategy) & (df['scenario'] == scenario)][metric].values
                strategy_means.append(np.mean(data))
                strategy_errors.append(np.std(data) / np.sqrt(len(data)))
            
            ax.bar(x + idx * width, strategy_means, width, 
                   yerr=strategy_errors, capsize=3,
                   label=STRATEGY_NAMES[strategy],
                   color=STRATEGY_COLORS[strategy],
                   alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Scenario', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} by Scenario and Strategy', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], 
                           rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f'{metric}_by_scenario.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from frozen experimental results"
    )
    parser.add_argument(
        '--aggregated',
        type=Path,
        required=True,
        help='Path to aggregated results CSV'
    )
    parser.add_argument(
        '--statistical',
        type=Path,
        required=True,
        help='Path to statistical analysis JSON'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for figures'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.aggregated.exists():
        raise FileNotFoundError(f"Aggregated results not found: {args.aggregated}")
    if not args.statistical.exists():
        raise FileNotFoundError(f"Statistical analysis not found: {args.statistical}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FIGURE GENERATION FROM FROZEN RESULTS")
    print("="*60)
    print(f"Aggregated: {args.aggregated}")
    print(f"Statistical: {args.statistical}")
    print(f"Output: {args.output}")
    print()
    
    # Load data
    df, stats = load_data(args.aggregated, args.statistical)
    print(f"Loaded {len(df)} observations")
    print(f"Strategies: {sorted(df['strategy'].unique())}")
    print(f"Scenarios: {sorted(df['scenario'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print()
    
    # Generate figures
    print("Generating figures...")
    
    generate_performance_boxplots(df, args.output)
    generate_adaptation_metrics_plot(df, args.output)
    generate_strategy_comparison_barplot(df, args.output)
    generate_pairwise_significance_heatmap(stats, args.output)
    generate_effect_size_heatmap(stats, args.output)
    generate_scenario_performance_plot(df, args.output)
    
    # Summary
    figures = list(args.output.glob('*.png'))
    print()
    print("="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"Total Figures: {len(figures)}")
    for fig in sorted(figures):
        print(f"  - {fig.name}")
    print()


if __name__ == '__main__':
    main()
