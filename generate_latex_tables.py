"""
Generate LaTeX tables from statistical test results.

This script reads the CSV files containing statistical test results and
generates properly formatted LaTeX tables for publication.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def format_pvalue(p):
    """Format p-value for display."""
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


def format_number(x, decimals=2):
    """Format number with specified decimals."""
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def generate_summary_table(csv_file, output_file, metric_name):
    """
    Generate LaTeX table from summary statistics CSV.
    
    Parameters:
    -----------
    csv_file : str
        Path to the summary CSV file
    output_file : str
        Path to save the LaTeX table
    metric_name : str
        Name of the metric being summarized
    """
    df = pd.read_csv(csv_file)
    
    # Create LaTeX table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Summary Statistics: " + metric_name + "}")
    latex.append("\\label{tab:" + metric_name.lower().replace(" ", "_") + "_summary}")
    
    # Determine column format
    n_cols = 5  # environment, setup, mean, std, n
    latex.append("\\begin{tabular}{ll" + "r" * (n_cols - 2) + "}")
    latex.append("\\toprule")
    
    # Header
    latex.append("Environment & Setup & Mean & Std & Median & IQR & N \\\\")
    latex.append("\\midrule")
    
    # Group by environment
    for env in sorted(df['environment'].unique()):
        env_data = df[df['environment'] == env].sort_values('training_setup')
        n_rows = len(env_data)
        
        for idx, (_, row) in enumerate(env_data.iterrows()):
            if idx == 0:
                env_label = env.replace("minatar-", "").replace("_", " ").replace("-", " ").title()
                env_col = f"\\multirow{{{n_rows}}}{{*}}{{{env_label}}}"
            else:
                env_col = ""
            
            setup = row['training_setup']
            mean_val = format_number(row['mean'], 3)
            std_val = format_number(row['std'], 3)
            median_val = format_number(row['median'], 3)
            iqr_val = format_number(row['iqr'], 3)
            n = int(row['n'])
            
            latex.append(f"{env_col} & {setup} & {mean_val} & {std_val} & {median_val} & {iqr_val} & {n} \\\\")
        
        latex.append("\\midrule")
    
    latex[-1] = latex[-1].replace("\\midrule", "")  # Remove last midrule
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"Generated LaTeX table: {output_file}")
    return '\n'.join(latex)


def generate_pairwise_compact_table(csv_file, output_file, metric_name):
    """
    Generate compact LaTeX table from pairwise comparison CSV.
    Shows only t-test results with effect sizes.
    
    Parameters:
    -----------
    csv_file : str
        Path to the pairwise CSV file
    output_file : str
        Path to save the LaTeX table
    metric_name : str
        Name of the metric being compared
    """
    df = pd.read_csv(csv_file)
    
    # Filter to only t-test results
    df = df[df['test'] == 'ttest'].copy()
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Pairwise Comparisons: " + metric_name + " (t-test)}")
    latex.append("\\label{tab:" + metric_name.lower().replace(" ", "_") + "_pairwise}")
    latex.append("\\begin{tabular}{llrrrrr}")
    latex.append("\\toprule")
    latex.append("Environment & Comparison & $\\mu_1$ & $\\mu_2$ & $t$ & $p$ & Cohen's $d$ \\\\")
    latex.append("\\midrule")
    
    # Group by environment
    for env in sorted(df['environment'].unique()):
        env_data = df[df['environment'] == env].sort_values(['setup1', 'setup2'])
        n_rows = len(env_data)
        
        for idx, (_, row) in enumerate(env_data.iterrows()):
            if idx == 0:
                env_label = env.replace("minatar-", "").replace("_", " ").replace("-", " ").title()
                env_col = f"\\multirow{{{n_rows}}}{{*}}{{{env_label}}}"
            else:
                env_col = ""
            
            comparison = f"{row['setup1']} vs {row['setup2']}"
            mean1 = format_number(row['mean1'], 2)
            mean2 = format_number(row['mean2'], 2)
            t_stat = format_number(row['statistic'], 2)
            p_val = format_pvalue(row['pvalue'])
            cohens_d = format_number(row['effect_size'], 2)
            
            # Add significance marker
            sig_marker = ""
            if row['pvalue'] < 0.001:
                sig_marker = "***"
            elif row['pvalue'] < 0.01:
                sig_marker = "**"
            elif row['pvalue'] < 0.05:
                sig_marker = "*"
            
            latex.append(f"{env_col} & {comparison} & {mean1} & {mean2} & {t_stat} & {p_val}{sig_marker} & {cohens_d} \\\\")
        
        latex.append("\\midrule")
    
    latex[-1] = latex[-1].replace("\\midrule", "")  # Remove last midrule
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\\\[0.5em]")
    latex.append("\\small{$^*p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$}")
    latex.append("\\end{table}")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"Generated LaTeX table: {output_file}")
    return '\n'.join(latex)


def generate_pairwise_full_table(csv_file, output_file, metric_name):
    """
    Generate comprehensive LaTeX table from pairwise comparison CSV.
    Shows multiple test types.
    
    Parameters:
    -----------
    csv_file : str
        Path to the pairwise CSV file
    output_file : str
        Path to save the LaTeX table
    metric_name : str
        Name of the metric being compared
    """
    df = pd.read_csv(csv_file)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Comprehensive Pairwise Comparisons: " + metric_name + "}")
    latex.append("\\label{tab:" + metric_name.lower().replace(" ", "_") + "_pairwise_full}")
    latex.append("\\begin{tabular}{lllrrr}")
    latex.append("\\toprule")
    latex.append("Environment & Comparison & Test & Statistic & $p$-value & Significant \\\\")
    latex.append("\\midrule")
    
    # Group by environment and comparison
    for env in sorted(df['environment'].unique()):
        env_data = df[df['environment'] == env]
        
        comparisons = list(env_data.groupby(['setup1', 'setup2']))
        n_comparisons = len(comparisons)
        
        # Count rows per environment for multirow
        n_tests_per_comparison = len(comparisons[0][1]) if comparisons else 0
        n_rows_env = n_comparisons * n_tests_per_comparison
        
        first_env = True
        for comp_idx, ((setup1, setup2), group) in enumerate(comparisons):
            comparison = f"{setup1} vs {setup2}"
            n_tests = len(group)
            
            for test_idx, (_, row) in enumerate(group.iterrows()):
                # Environment column with multirow
                if first_env:
                    env_label = env.replace("minatar-", "").replace("_", " ").replace("-", " ").title()
                    env_col = f"\\multirow{{{n_rows_env}}}{{*}}{{{env_label}}}"
                    first_env = False
                else:
                    env_col = ""
                
                # Comparison column with multirow
                if test_idx == 0:
                    comp_col = f"\\multirow{{{n_tests}}}{{*}}{{{comparison}}}"
                else:
                    comp_col = ""
                
                test_name = row['test'].replace('_', ' ').title()
                if test_name == 'Mannwhitneyu':
                    test_name = 'Mann-Whitney U'
                elif test_name == 'Kolmogorov Smirnov':
                    test_name = 'K-S'
                elif test_name == 'Welch Ttest':
                    test_name = "Welch's t"
                
                stat = format_number(row['statistic'], 2)
                p_val = format_pvalue(row['pvalue'])
                sig = "Yes" if row['significant'] else "No"
                
                latex.append(f"{env_col} & {comp_col} & {test_name} & {stat} & {p_val} & {sig} \\\\")
        
        latex.append("\\midrule")
    
    latex[-1] = latex[-1].replace("\\midrule", "")  # Remove last midrule
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"Generated LaTeX table: {output_file}")
    return '\n'.join(latex)


def generate_effect_sizes_table(pairwise_files, output_file):
    """
    Generate a table comparing effect sizes across all metrics.
    
    Parameters:
    -----------
    pairwise_files : dict
        Dictionary mapping metric names to pairwise CSV file paths
    output_file : str
        Path to save the LaTeX table
    """
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Effect Sizes (Cohen's $d$) Across Metrics}")
    latex.append("\\label{tab:effect_sizes}")
    latex.append("\\begin{tabular}{llrrr}")
    latex.append("\\toprule")
    latex.append("Environment & Comparison & Importance & Similarity & Performance \\\\")
    latex.append("\\midrule")
    
    # Load all dataframes
    dfs = {}
    for name, file in pairwise_files.items():
        df = pd.read_csv(file)
        df = df[df['test'] == 'ttest']
        dfs[name] = df
    
    # Get unique environments and comparisons
    envs = sorted(dfs[list(dfs.keys())[0]]['environment'].unique())
    
    for env in envs:
        env_label = env.replace("minatar-", "").replace("_", " ").replace("-", " ").title()
        
        # Get all comparisons for this environment
        comparisons = dfs[list(dfs.keys())[0]][
            dfs[list(dfs.keys())[0]]['environment'] == env
        ][['setup1', 'setup2']].values
        
        n_comparisons = len(comparisons)
        
        for idx, (setup1, setup2) in enumerate(comparisons):
            if idx == 0:
                env_col = f"\\multirow{{{n_comparisons}}}{{*}}{{{env_label}}}"
            else:
                env_col = ""
            
            comparison = f"{setup1} vs {setup2}"
            
            # Get effect sizes for each metric
            effect_sizes = []
            for name in ['importance', 'similarity', 'performance']:
                if name in dfs:
                    row_data = dfs[name][
                        (dfs[name]['environment'] == env) &
                        (dfs[name]['setup1'] == setup1) &
                        (dfs[name]['setup2'] == setup2)
                    ]
                    
                    if len(row_data) > 0:
                        d = row_data.iloc[0]['effect_size']
                        effect_sizes.append(format_number(d, 2))
                    else:
                        effect_sizes.append("—")
                else:
                    effect_sizes.append("—")
            
            latex.append(f"{env_col} & {comparison} & {' & '.join(effect_sizes)} \\\\")
        
        latex.append("\\midrule")
    
    latex[-1] = latex[-1].replace("\\midrule", "")  # Remove last midrule
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"Generated LaTeX table: {output_file}")
    return '\n'.join(latex)


def generate_extended_effect_sizes_table(pairwise_files, output_file):
    """
    Generate an extended table comparing effect sizes across all metrics including action deviance.
    
    Parameters:
    -----------
    pairwise_files : dict
        Dictionary mapping metric names to pairwise CSV file paths
    output_file : str
        Path to save the LaTeX table
    """
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Effect Sizes (Cohen's $d$) Across All Metrics}")
    latex.append("\\label{tab:effect_sizes_extended}")
    latex.append("\\begin{tabular}{llrrrrrr}")
    latex.append("\\toprule")
    latex.append("Environment & Comparison & Importance & Similarity & Performance & Overlap & Action Dev. & Action Dev. (Imp.) \\\\")
    latex.append("\\midrule")
    
    # Load all dataframes
    dfs = {}
    for name, file in pairwise_files.items():
        if Path(file).exists():
            df = pd.read_csv(file)
            df = df[df['test'] == 'ttest']
            dfs[name] = df
    
    if not dfs:
        print("Warning: No data files found for extended effect sizes table")
        return
    
    # Get unique environments and comparisons
    envs = sorted(dfs[list(dfs.keys())[0]]['environment'].unique())
    
    for env in envs:
        env_label = env.replace("minatar-", "").replace("_", " ").replace("-", " ").title()
        
        # Get all comparisons for this environment
        comparisons = dfs[list(dfs.keys())[0]][
            dfs[list(dfs.keys())[0]]['environment'] == env
        ][['setup1', 'setup2']].values
        
        n_comparisons = len(comparisons)
        
        for idx, (setup1, setup2) in enumerate(comparisons):
            if idx == 0:
                env_col = f"\\multirow{{{n_comparisons}}}{{*}}{{{env_label}}}"
            else:
                env_col = ""
            
            comparison = f"{setup1} vs {setup2}"
            
            # Get effect sizes for each metric
            effect_sizes = []
            for name in ['importance', 'similarity', 'performance', 'overlap', 'action_dev', 'action_dev_imp']:
                if name in dfs:
                    row_data = dfs[name][
                        (dfs[name]['environment'] == env) &
                        (dfs[name]['setup1'] == setup1) &
                        (dfs[name]['setup2'] == setup2)
                    ]
                    
                    if len(row_data) > 0:
                        d = row_data.iloc[0]['effect_size']
                        effect_sizes.append(format_number(d, 2))
                    else:
                        effect_sizes.append("—")
                else:
                    effect_sizes.append("—")
            
            latex.append(f"{env_col} & {comparison} & {' & '.join(effect_sizes)} \\\\")
        
        latex.append("\\midrule")
    
    latex[-1] = latex[-1].replace("\\midrule", "")  # Remove last midrule
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"Generated LaTeX table: {output_file}")
    return '\n'.join(latex)


def main():
    """Generate all LaTeX tables from statistical results."""
    results_dir = Path('statistical_results')
    output_dir = Path('latex_tables')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("GENERATING LATEX TABLES")
    print("=" * 80)
    
    # Generate summary tables
    print("\n--- Summary Tables ---")
    generate_summary_table(
        results_dir / 'importance_agreement_summary.csv',
        output_dir / 'importance_agreement_summary.tex',
        'Importance Agreement'
    )
    
    generate_summary_table(
        results_dir / 'similarity_summary.csv',
        output_dir / 'similarity_summary.tex',
        'Feature Similarity'
    )
    
    generate_summary_table(
        results_dir / 'policy_performance_summary.csv',
        output_dir / 'policy_performance_summary.tex',
        'Policy Performance'
    )
    
    # Generate compact pairwise tables (t-test only)
    print("\n--- Compact Pairwise Comparison Tables ---")
    generate_pairwise_compact_table(
        results_dir / 'importance_agreement_pairwise.csv',
        output_dir / 'importance_agreement_pairwise.tex',
        'Importance Agreement'
    )
    
    generate_pairwise_compact_table(
        results_dir / 'similarity_pairwise.csv',
        output_dir / 'similarity_pairwise.tex',
        'Feature Similarity'
    )
    
    generate_pairwise_compact_table(
        results_dir / 'policy_performance_pairwise.csv',
        output_dir / 'policy_performance_pairwise.tex',
        'Policy Performance'
    )
    
    # Generate full pairwise tables (all tests)
    print("\n--- Full Pairwise Comparison Tables ---")
    generate_pairwise_full_table(
        results_dir / 'importance_agreement_pairwise.csv',
        output_dir / 'importance_agreement_pairwise_full.tex',
        'Importance Agreement'
    )
    
    generate_pairwise_full_table(
        results_dir / 'similarity_pairwise.csv',
        output_dir / 'similarity_pairwise_full.tex',
        'Feature Similarity'
    )
    
    generate_pairwise_full_table(
        results_dir / 'policy_performance_pairwise.csv',
        output_dir / 'policy_performance_pairwise_full.tex',
        'Policy Performance'
    )
    
    # Generate feature overlap tables
    print("\n--- Feature Overlap Tables ---")
    
    if (results_dir / 'feature_overlap_summary.csv').exists():
        generate_summary_table(
            results_dir / 'feature_overlap_summary.csv',
            output_dir / 'feature_overlap_summary.tex',
            'Feature Overlap'
        )
        
        generate_pairwise_compact_table(
            results_dir / 'feature_overlap_pairwise.csv',
            output_dir / 'feature_overlap_pairwise.tex',
            'Feature Overlap'
        )
        
        generate_pairwise_full_table(
            results_dir / 'feature_overlap_pairwise.csv',
            output_dir / 'feature_overlap_pairwise_full.tex',
            'Feature Overlap'
        )
    
    # Generate action deviance tables
    print("\n--- Action Deviance Tables ---")
    
    if (results_dir / 'action_deviance_summary.csv').exists():
        generate_summary_table(
            results_dir / 'action_deviance_summary.csv',
            output_dir / 'action_deviance_summary.tex',
            'Action Deviance (Overall)'
        )
        
        generate_pairwise_compact_table(
            results_dir / 'action_deviance_pairwise.csv',
            output_dir / 'action_deviance_pairwise.tex',
            'Action Deviance (Overall)'
        )
        
        generate_pairwise_full_table(
            results_dir / 'action_deviance_pairwise.csv',
            output_dir / 'action_deviance_pairwise_full.tex',
            'Action Deviance (Overall)'
        )
    
    if (results_dir / 'action_deviance_in_most_important_states_summary.csv').exists():
        generate_summary_table(
            results_dir / 'action_deviance_in_most_important_states_summary.csv',
            output_dir / 'action_deviance_important_states_summary.tex',
            'Action Deviance (Important States)'
        )
        
        generate_pairwise_compact_table(
            results_dir / 'action_deviance_in_most_important_states_pairwise.csv',
            output_dir / 'action_deviance_important_states_pairwise.tex',
            'Action Deviance (Important States)'
        )
        
        generate_pairwise_full_table(
            results_dir / 'action_deviance_in_most_important_states_pairwise.csv',
            output_dir / 'action_deviance_important_states_pairwise_full.tex',
            'Action Deviance (Important States)'
        )
    
    # Generate effect sizes comparison table
    print("\n--- Effect Sizes Comparison Table ---")
    generate_effect_sizes_table(
        {
            'importance': results_dir / 'importance_agreement_pairwise.csv',
            'similarity': results_dir / 'similarity_pairwise.csv',
            'performance': results_dir / 'policy_performance_pairwise.csv'
        },
        output_dir / 'effect_sizes_comparison.tex'
    )
    
    # Generate extended effect sizes table with action deviance
    if (results_dir / 'action_deviance_pairwise.csv').exists():
        print("\n--- Extended Effect Sizes Comparison Table ---")
        generate_extended_effect_sizes_table(
            {
                'importance': results_dir / 'importance_agreement_pairwise.csv',
                'similarity': results_dir / 'similarity_pairwise.csv',
                'performance': results_dir / 'policy_performance_pairwise.csv',
                'overlap': results_dir / 'feature_overlap_pairwise.csv',
                'action_dev': results_dir / 'action_deviance_pairwise.csv',
                'action_dev_imp': results_dir / 'action_deviance_in_most_important_states_pairwise.csv'
            },
            output_dir / 'effect_sizes_comparison_extended.tex'
        )
    
    print("\n" + "=" * 80)
    print("LATEX TABLE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTables saved to: {output_dir}/")
    print("\nGenerated files:")
    for tex_file in sorted(output_dir.glob('*.tex')):
        print(f"  - {tex_file.name}")


if __name__ == "__main__":
    main()

