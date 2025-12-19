"""
Statistical analysis of ensemble differences.

This script performs statistical tests comparing different ensemble training setups
on importance agreement and feature similarity metrics.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from statistical_tests import (
    compare_distributions,
    summary_statistics,
    levene_test,
    anova_test,
    kruskal_wallis_test
)
from itertools import combinations
import os


def load_pickle_data(filepath):
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def analyze_importance_agreement(data, output_dir='results'):
    """
    Analyze importance agreement data across training setups.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing 'environment', 'training_setup', and 'importance_agreement'
    output_dir : str
        Directory to save results
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    environments = data['environment']
    training_setups = data['training_setup']
    importance_agreement = data['importance_agreement']
    
    # Get unique values
    unique_envs = sorted(set(environments))
    unique_setups = sorted(set(training_setups))
    
    print("=" * 80)
    print("IMPORTANCE AGREEMENT ANALYSIS")
    print("=" * 80)
    print(f"\nEnvironments: {unique_envs}")
    print(f"Training Setups: {unique_setups}")
    print(f"Total comparisons: {len(environments)}")
    
    # Summary statistics for each environment and setup
    summary_results = []
    
    for env in unique_envs:
        for setup in unique_setups:
            # Find matching indices
            indices = [i for i, (e, s) in enumerate(zip(environments, training_setups))
                      if e == env and s == setup]
            
            if indices:
                # Get all values for this env-setup combination
                values = np.concatenate([importance_agreement[i].flatten() 
                                       for i in indices])
                
                stats = summary_statistics(values)
                stats['environment'] = env
                stats['training_setup'] = setup
                summary_results.append(stats)
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_results)
    summary_file = Path(output_dir) / 'importance_agreement_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary statistics to: {summary_file}")
    
    # Pairwise comparisons between training setups for each environment
    pairwise_results = []
    
    for env in unique_envs:
        print(f"\n{'-' * 80}")
        print(f"Environment: {env}")
        print(f"{'-' * 80}")
        
        # Collect data for each training setup
        setup_data = {}
        for setup in unique_setups:
            indices = [i for i, (e, s) in enumerate(zip(environments, training_setups))
                      if e == env and s == setup]
            if indices:
                setup_data[setup] = np.concatenate([importance_agreement[i].flatten() 
                                                   for i in indices])
        
        # Perform ANOVA/Kruskal-Wallis if we have multiple setups
        if len(setup_data) > 2:
            distributions = list(setup_data.values())
            anova_result = anova_test(*distributions)
            kw_result = kruskal_wallis_test(*distributions)
            
            print(f"\nANOVA F-statistic: {anova_result['statistic']:.4f}, "
                  f"p-value: {anova_result['pvalue']:.6f}, "
                  f"significant: {anova_result['significant']}")
            print(f"Kruskal-Wallis H: {kw_result['statistic']:.4f}, "
                  f"p-value: {kw_result['pvalue']:.6f}, "
                  f"significant: {kw_result['significant']}")
        
        # Pairwise comparisons
        for setup1, setup2 in combinations(unique_setups, 2):
            if setup1 in setup_data and setup2 in setup_data:
                print(f"\n  Comparing {setup1} vs {setup2}:")
                
                dist1 = setup_data[setup1]
                dist2 = setup_data[setup2]
                
                # Perform all statistical tests
                results = compare_distributions(dist1, dist2)
                
                # Add metadata
                for test_name, test_result in results.items():
                    result_row = {
                        'environment': env,
                        'setup1': setup1,
                        'setup2': setup2,
                        'test': test_name,
                        'n1': len(dist1),
                        'n2': len(dist2),
                        'mean1': np.mean(dist1),
                        'mean2': np.mean(dist2),
                        'median1': np.median(dist1),
                        'median2': np.median(dist2),
                        'std1': np.std(dist1, ddof=1),
                        'std2': np.std(dist2, ddof=1),
                    }
                    result_row.update(test_result)
                    pairwise_results.append(result_row)
                
                # Print key results
                ttest = results['ttest']
                print(f"    t-test: t={ttest['statistic']:.4f}, p={ttest['pvalue']:.6f}, "
                      f"Cohen's d={ttest['effect_size']:.4f} ({ttest['interpretation']})")
                
                mw = results['mannwhitneyu']
                print(f"    Mann-Whitney U: U={mw['statistic']:.4f}, p={mw['pvalue']:.6f}")
    
    # Save pairwise comparison results
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_file = Path(output_dir) / 'importance_agreement_pairwise.csv'
    pairwise_df.to_csv(pairwise_file, index=False)
    print(f"\n\nSaved pairwise comparisons to: {pairwise_file}")
    
    return summary_df, pairwise_df


def analyze_similarities(data, output_dir='results'):
    """
    Analyze feature similarity data across training setups.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing similarity metrics
    output_dir : str
        Directory to save results
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    environments = data['environment']
    training_setups = data['training_setup']
    mean_similarity = data['mean_similarity']
    
    # Get unique values
    unique_envs = sorted(set(environments))
    unique_setups = sorted(set(training_setups))
    
    print("\n\n" + "=" * 80)
    print("FEATURE SIMILARITY ANALYSIS")
    print("=" * 80)
    print(f"\nEnvironments: {unique_envs}")
    print(f"Training Setups: {unique_setups}")
    
    # Summary statistics for each environment and setup
    summary_results = []
    
    for env in unique_envs:
        for setup in unique_setups:
            # Find matching indices
            indices = [i for i, (e, s) in enumerate(zip(environments, training_setups))
                      if e == env and s == setup]
            
            if indices:
                # Get all values for this env-setup combination
                # Extract upper triangle of similarity matrices (excluding diagonal)
                values = []
                for i in indices:
                    sim_matrix = mean_similarity[i]
                    # Get upper triangle excluding diagonal
                    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                    values.extend(sim_matrix[mask].flatten())
                
                values = np.array(values)
                
                stats = summary_statistics(values)
                stats['environment'] = env
                stats['training_setup'] = setup
                summary_results.append(stats)
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_results)
    summary_file = Path(output_dir) / 'similarity_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary statistics to: {summary_file}")
    
    # Pairwise comparisons between training setups for each environment
    pairwise_results = []
    
    for env in unique_envs:
        print(f"\n{'-' * 80}")
        print(f"Environment: {env}")
        print(f"{'-' * 80}")
        
        # Collect data for each training setup
        setup_data = {}
        for setup in unique_setups:
            indices = [i for i, (e, s) in enumerate(zip(environments, training_setups))
                      if e == env and s == setup]
            if indices:
                values = []
                for i in indices:
                    sim_matrix = mean_similarity[i]
                    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                    values.extend(sim_matrix[mask].flatten())
                setup_data[setup] = np.array(values)
        
        # Perform ANOVA/Kruskal-Wallis if we have multiple setups
        if len(setup_data) > 2:
            distributions = list(setup_data.values())
            anova_result = anova_test(*distributions)
            kw_result = kruskal_wallis_test(*distributions)
            
            print(f"\nANOVA F-statistic: {anova_result['statistic']:.4f}, "
                  f"p-value: {anova_result['pvalue']:.6f}, "
                  f"significant: {anova_result['significant']}")
            print(f"Kruskal-Wallis H: {kw_result['statistic']:.4f}, "
                  f"p-value: {kw_result['pvalue']:.6f}, "
                  f"significant: {kw_result['significant']}")
        
        # Pairwise comparisons
        for setup1, setup2 in combinations(unique_setups, 2):
            if setup1 in setup_data and setup2 in setup_data:
                print(f"\n  Comparing {setup1} vs {setup2}:")
                
                dist1 = setup_data[setup1]
                dist2 = setup_data[setup2]
                
                # Perform all statistical tests
                results = compare_distributions(dist1, dist2)
                
                # Add metadata
                for test_name, test_result in results.items():
                    result_row = {
                        'environment': env,
                        'setup1': setup1,
                        'setup2': setup2,
                        'test': test_name,
                        'n1': len(dist1),
                        'n2': len(dist2),
                        'mean1': np.mean(dist1),
                        'mean2': np.mean(dist2),
                        'median1': np.median(dist1),
                        'median2': np.median(dist2),
                        'std1': np.std(dist1, ddof=1),
                        'std2': np.std(dist2, ddof=1),
                    }
                    result_row.update(test_result)
                    pairwise_results.append(result_row)
                
                # Print key results
                ttest = results['ttest']
                print(f"    t-test: t={ttest['statistic']:.4f}, p={ttest['pvalue']:.6f}, "
                      f"Cohen's d={ttest['effect_size']:.4f} ({ttest['interpretation']})")
                
                mw = results['mannwhitneyu']
                print(f"    Mann-Whitney U: U={mw['statistic']:.4f}, p={mw['pvalue']:.6f}")
    
    # Save pairwise comparison results
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_file = Path(output_dir) / 'similarity_pairwise.csv'
    pairwise_df.to_csv(pairwise_file, index=False)
    print(f"\n\nSaved pairwise comparisons to: {pairwise_file}")
    
    return summary_df, pairwise_df


def analyze_policy_performance(eval_dir='evaluation_of_policies', output_dir='results'):
    """
    Analyze policy performance data across training setups.
    
    Parameters:
    -----------
    eval_dir : str
        Directory containing evaluation data
    output_dir : str
        Directory to save results
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Mapping from file names to training setup labels
    file_to_setup = {
        'different-world-policy-performances.csv': 'DWDI',
        'different-world-same-init-policy-performances.csv': 'DWSI',
        'same-world-policy-performances.csv': 'SWDI'
    }
    
    # Get all environment directories
    eval_path = Path(eval_dir)
    env_dirs = [d for d in eval_path.iterdir() if d.is_dir()]
    
    print("\n\n" + "=" * 80)
    print("POLICY PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Collect all data
    all_data = []
    
    for env_dir in sorted(env_dirs):
        env_name = env_dir.name
        
        for filename, setup in file_to_setup.items():
            filepath = env_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                
                # Filter out invalid performance values (-1.0)
                df = df[df['mean_rewards'] != -1.0]
                
                # For same-world policies, only keep first 200
                if setup == 'SWDI':
                    df = df[df['policy_index'] <= 200]
                
                for _, row in df.iterrows():
                    all_data.append({
                        'environment': env_name,
                        'training_setup': setup,
                        'mean_reward': row['mean_rewards'],
                        'std_reward': row['std_rewards'],
                        'policy_index': row['policy_index']
                    })
    
    # Create dataframe from all data
    full_df = pd.DataFrame(all_data)
    
    unique_envs = sorted(full_df['environment'].unique())
    unique_setups = sorted(full_df['training_setup'].unique())
    
    print(f"\nEnvironments: {unique_envs}")
    print(f"Training Setups: {unique_setups}")
    print(f"Total policies evaluated: {len(full_df)}")
    
    # Summary statistics for each environment and setup
    summary_results = []
    
    for env in unique_envs:
        for setup in unique_setups:
            # Get rewards for this env-setup combination
            mask = (full_df['environment'] == env) & (full_df['training_setup'] == setup)
            rewards = full_df.loc[mask, 'mean_reward'].values
            
            if len(rewards) > 0:
                stats = summary_statistics(rewards)
                stats['environment'] = env
                stats['training_setup'] = setup
                summary_results.append(stats)
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_results)
    summary_file = Path(output_dir) / 'policy_performance_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary statistics to: {summary_file}")
    
    # Pairwise comparisons between training setups for each environment
    pairwise_results = []
    
    for env in unique_envs:
        print(f"\n{'-' * 80}")
        print(f"Environment: {env}")
        print(f"{'-' * 80}")
        
        # Collect data for each training setup
        setup_data = {}
        for setup in unique_setups:
            mask = (full_df['environment'] == env) & (full_df['training_setup'] == setup)
            rewards = full_df.loc[mask, 'mean_reward'].values
            if len(rewards) > 0:
                setup_data[setup] = rewards
        
        # Perform ANOVA/Kruskal-Wallis if we have multiple setups
        if len(setup_data) > 2:
            distributions = list(setup_data.values())
            anova_result = anova_test(*distributions)
            kw_result = kruskal_wallis_test(*distributions)
            
            print(f"\nANOVA F-statistic: {anova_result['statistic']:.4f}, "
                  f"p-value: {anova_result['pvalue']:.6f}, "
                  f"significant: {anova_result['significant']}")
            print(f"Kruskal-Wallis H: {kw_result['statistic']:.4f}, "
                  f"p-value: {kw_result['pvalue']:.6f}, "
                  f"significant: {kw_result['significant']}")
        
        # Pairwise comparisons
        for setup1, setup2 in combinations(unique_setups, 2):
            if setup1 in setup_data and setup2 in setup_data:
                print(f"\n  Comparing {setup1} vs {setup2}:")
                
                dist1 = setup_data[setup1]
                dist2 = setup_data[setup2]
                
                # Perform all statistical tests
                results = compare_distributions(dist1, dist2)
                
                # Add metadata
                for test_name, test_result in results.items():
                    result_row = {
                        'environment': env,
                        'setup1': setup1,
                        'setup2': setup2,
                        'test': test_name,
                        'n1': len(dist1),
                        'n2': len(dist2),
                        'mean1': np.mean(dist1),
                        'mean2': np.mean(dist2),
                        'median1': np.median(dist1),
                        'median2': np.median(dist2),
                        'std1': np.std(dist1, ddof=1),
                        'std2': np.std(dist2, ddof=1),
                    }
                    result_row.update(test_result)
                    pairwise_results.append(result_row)
                
                # Print key results
                ttest = results['ttest']
                print(f"    t-test: t={ttest['statistic']:.4f}, p={ttest['pvalue']:.6f}, "
                      f"Cohen's d={ttest['effect_size']:.4f} ({ttest['interpretation']})")
                
                mw = results['mannwhitneyu']
                print(f"    Mann-Whitney U: U={mw['statistic']:.4f}, p={mw['pvalue']:.6f}")
    
    # Save pairwise comparison results
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_file = Path(output_dir) / 'policy_performance_pairwise.csv'
    pairwise_df.to_csv(pairwise_file, index=False)
    print(f"\n\nSaved pairwise comparisons to: {pairwise_file}")
    
    return summary_df, pairwise_df


def analyze_action_deviance(csv_file, output_dir='results'):
    """
    Analyze action deviance data across training setups.
    
    Parameters:
    -----------
    csv_file : str
        Path to the action deviance CSV file
    output_dir : str
        Directory to save results
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Get unique values
    unique_envs = sorted(df['environment'].unique())
    unique_setups = sorted(df['training_setup'].unique())
    
    print("\n\n" + "=" * 80)
    print("ACTION DEVIANCE ANALYSIS")
    print("=" * 80)
    print(f"\nEnvironments: {unique_envs}")
    print(f"Training Setups: {unique_setups}")
    print(f"Total policies: {len(df)}")
    
    # Analyze both metrics
    metrics = [
        ('fraction_action_deviance', 'Overall Action Deviance'),
        ('fraction_action_deviance_in_most_important_states', 'Action Deviance in Important States')
    ]
    
    for metric_col, metric_name in metrics:
        print("\n" + "=" * 80)
        print(f"Metric: {metric_name}")
        print("=" * 80)
        
        # Summary statistics for each environment and setup
        summary_results = []
        
        for env in unique_envs:
            for setup in unique_setups:
                # Get values for this env-setup combination
                mask = (df['environment'] == env) & (df['training_setup'] == setup)
                values = df.loc[mask, metric_col].values
                
                if len(values) > 0:
                    stats = summary_statistics(values)
                    stats['environment'] = env
                    stats['training_setup'] = setup
                    stats['metric'] = metric_name
                    summary_results.append(stats)
        
        # Save summary statistics
        summary_df = pd.DataFrame(summary_results)
        metric_filename = metric_col.replace('fraction_', '')
        summary_file = Path(output_dir) / f'{metric_filename}_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary statistics to: {summary_file}")
        
        # Pairwise comparisons between training setups for each environment
        pairwise_results = []
        
        for env in unique_envs:
            print(f"\n{'-' * 80}")
            print(f"Environment: {env}")
            print(f"{'-' * 80}")
            
            # Collect data for each training setup
            setup_data = {}
            for setup in unique_setups:
                mask = (df['environment'] == env) & (df['training_setup'] == setup)
                values = df.loc[mask, metric_col].values
                if len(values) > 0:
                    setup_data[setup] = values
            
            # Perform ANOVA/Kruskal-Wallis if we have multiple setups
            if len(setup_data) > 2:
                distributions = list(setup_data.values())
                anova_result = anova_test(*distributions)
                kw_result = kruskal_wallis_test(*distributions)
                
                print(f"\nANOVA F-statistic: {anova_result['statistic']:.4f}, "
                      f"p-value: {anova_result['pvalue']:.6f}, "
                      f"significant: {anova_result['significant']}")
                print(f"Kruskal-Wallis H: {kw_result['statistic']:.4f}, "
                      f"p-value: {kw_result['pvalue']:.6f}, "
                      f"significant: {kw_result['significant']}")
            
            # Pairwise comparisons
            for setup1, setup2 in combinations(unique_setups, 2):
                if setup1 in setup_data and setup2 in setup_data:
                    print(f"\n  Comparing {setup1} vs {setup2}:")
                    
                    dist1 = setup_data[setup1]
                    dist2 = setup_data[setup2]
                    
                    # Perform all statistical tests
                    results = compare_distributions(dist1, dist2)
                    
                    # Add metadata
                    for test_name, test_result in results.items():
                        result_row = {
                            'environment': env,
                            'setup1': setup1,
                            'setup2': setup2,
                            'test': test_name,
                            'metric': metric_name,
                            'n1': len(dist1),
                            'n2': len(dist2),
                            'mean1': np.mean(dist1),
                            'mean2': np.mean(dist2),
                            'median1': np.median(dist1),
                            'median2': np.median(dist2),
                            'std1': np.std(dist1, ddof=1),
                            'std2': np.std(dist2, ddof=1),
                        }
                        result_row.update(test_result)
                        pairwise_results.append(result_row)
                    
                    # Print key results
                    ttest = results['ttest']
                    print(f"    t-test: t={ttest['statistic']:.4f}, p={ttest['pvalue']:.6f}, "
                          f"Cohen's d={ttest['effect_size']:.4f} ({ttest['interpretation']})")
                    
                    mw = results['mannwhitneyu']
                    print(f"    Mann-Whitney U: U={mw['statistic']:.4f}, p={mw['pvalue']:.6f}")
        
        # Save pairwise comparison results
        pairwise_df = pd.DataFrame(pairwise_results)
        pairwise_file = Path(output_dir) / f'{metric_filename}_pairwise.csv'
        pairwise_df.to_csv(pairwise_file, index=False)
        print(f"\n\nSaved pairwise comparisons to: {pairwise_file}")
    
    return summary_results, pairwise_results


def analyze_feature_overlap(pkl_file, output_dir='results'):
    """
    Analyze feature overlap data across training setups.
    
    Parameters:
    -----------
    pkl_file : str
        Path to the feature overlap pickle file
    output_dir : str
        Directory to save results
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    data = load_pickle_data(pkl_file)
    
    environments = data['environment']
    training_setups = data['training_setup']
    overlaps = data['overlaps']
    
    # Get unique values
    unique_envs = sorted(set(environments))
    unique_setups = sorted(set(training_setups))
    
    print("\n\n" + "=" * 80)
    print("FEATURE OVERLAP ANALYSIS")
    print("=" * 80)
    print(f"\nEnvironments: {unique_envs}")
    print(f"Training Setups: {unique_setups}")
    print(f"Total overlap matrices: {len(environments)}")
    
    # Summary statistics for each environment and setup
    summary_results = []
    
    for env in unique_envs:
        for setup in unique_setups:
            # Get overlap matrices for this env-setup combination
            indices = [i for i, (e, s) in enumerate(zip(environments, training_setups))
                      if e == env and s == setup]
            
            if indices:
                # Extract values from overlap arrays
                # The overlaps are already flattened upper triangle values
                values = []
                for i in indices:
                    overlap_array = np.array(overlaps[i])
                    values.extend(overlap_array.flatten())
                
                values = np.array(values)
                
                stats = summary_statistics(values)
                stats['environment'] = env
                stats['training_setup'] = setup
                summary_results.append(stats)
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_results)
    summary_file = Path(output_dir) / 'feature_overlap_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary statistics to: {summary_file}")
    
    # Pairwise comparisons between training setups for each environment
    pairwise_results = []
    
    for env in unique_envs:
        print(f"\n{'-' * 80}")
        print(f"Environment: {env}")
        print(f"{'-' * 80}")
        
        # Collect data for each training setup
        setup_data = {}
        for setup in unique_setups:
            indices = [i for i, (e, s) in enumerate(zip(environments, training_setups))
                      if e == env and s == setup]
            if indices:
                values = []
                for i in indices:
                    overlap_array = np.array(overlaps[i])
                    values.extend(overlap_array.flatten())
                setup_data[setup] = np.array(values)
        
        # Perform ANOVA/Kruskal-Wallis if we have multiple setups
        if len(setup_data) > 2:
            distributions = list(setup_data.values())
            anova_result = anova_test(*distributions)
            kw_result = kruskal_wallis_test(*distributions)
            
            print(f"\nANOVA F-statistic: {anova_result['statistic']:.4f}, "
                  f"p-value: {anova_result['pvalue']:.6f}, "
                  f"significant: {anova_result['significant']}")
            print(f"Kruskal-Wallis H: {kw_result['statistic']:.4f}, "
                  f"p-value: {kw_result['pvalue']:.6f}, "
                  f"significant: {kw_result['significant']}")
        
        # Pairwise comparisons
        for setup1, setup2 in combinations(unique_setups, 2):
            if setup1 in setup_data and setup2 in setup_data:
                print(f"\n  Comparing {setup1} vs {setup2}:")
                
                dist1 = setup_data[setup1]
                dist2 = setup_data[setup2]
                
                # Perform all statistical tests
                results = compare_distributions(dist1, dist2)
                
                # Add metadata
                for test_name, test_result in results.items():
                    result_row = {
                        'environment': env,
                        'setup1': setup1,
                        'setup2': setup2,
                        'test': test_name,
                        'n1': len(dist1),
                        'n2': len(dist2),
                        'mean1': np.mean(dist1),
                        'mean2': np.mean(dist2),
                        'median1': np.median(dist1),
                        'median2': np.median(dist2),
                        'std1': np.std(dist1, ddof=1),
                        'std2': np.std(dist2, ddof=1),
                    }
                    result_row.update(test_result)
                    pairwise_results.append(result_row)
                
                # Print key results
                ttest = results['ttest']
                print(f"    t-test: t={ttest['statistic']:.4f}, p={ttest['pvalue']:.6f}, "
                      f"Cohen's d={ttest['effect_size']:.4f} ({ttest['interpretation']})")
                
                mw = results['mannwhitneyu']
                print(f"    Mann-Whitney U: U={mw['statistic']:.4f}, p={mw['pvalue']:.6f}")
    
    # Save pairwise comparison results
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_file = Path(output_dir) / 'feature_overlap_pairwise.csv'
    pairwise_df.to_csv(pairwise_file, index=False)
    print(f"\n\nSaved pairwise comparisons to: {pairwise_file}")
    
    return summary_df, pairwise_df


def main():
    """Main analysis function."""
    # Load data
    print("Loading data...")
    importance_data = load_pickle_data('interpretation_data/importance_agreement_data.pkl')
    similarity_data = load_pickle_data('feature_analysis/similarities.pkl')
    
    # Create output directory
    output_dir = 'statistical_results'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Analyze importance agreement
    importance_summary, importance_pairwise = analyze_importance_agreement(
        importance_data, output_dir
    )
    
    # Analyze similarities
    similarity_summary, similarity_pairwise = analyze_similarities(
        similarity_data, output_dir
    )
    
    # Analyze policy performance
    performance_summary, performance_pairwise = analyze_policy_performance(
        'evaluation_of_policies', output_dir
    )
    
    # Analyze action deviance
    action_deviance_summary, action_deviance_pairwise = analyze_action_deviance(
        'interpretation_data/action_deviance_data.csv', output_dir
    )
    
    # Analyze feature overlap
    feature_overlap_summary, feature_overlap_pairwise = analyze_feature_overlap(
        'feature_analysis/feature_overlap_data.pkl', output_dir
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("Files created:")
    print("  - importance_agreement_summary.csv")
    print("  - importance_agreement_pairwise.csv")
    print("  - similarity_summary.csv")
    print("  - similarity_pairwise.csv")
    print("  - policy_performance_summary.csv")
    print("  - policy_performance_pairwise.csv")
    print("  - action_deviance_summary.csv")
    print("  - action_deviance_pairwise.csv")
    print("  - action_deviance_in_most_important_states_summary.csv")
    print("  - action_deviance_in_most_important_states_pairwise.csv")
    print("  - feature_overlap_summary.csv")
    print("  - feature_overlap_pairwise.csv")


if __name__ == "__main__":
    main()

