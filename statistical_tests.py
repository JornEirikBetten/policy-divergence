"""
Statistical tests for comparing distributions.

This module provides functions for performing various statistical tests
to compare distributions, useful for analyzing RL performance data.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Optional


def compare_distributions(
    dist1: np.ndarray,
    dist2: np.ndarray,
    tests: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    Perform multiple statistical tests to compare two distributions.
    
    Parameters:
    -----------
    dist1 : np.ndarray
        First distribution (1D array of samples)
    dist2 : np.ndarray
        Second distribution (1D array of samples)
    tests : List[str], optional
        List of tests to perform. Options: 'ttest', 'mannwhitneyu', 'ks', 'welch'
        If None, performs all tests.
    alpha : float, default=0.05
        Significance level for hypothesis testing
        
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary with test names as keys and results as values.
        Each result contains 'statistic', 'pvalue', and 'significant' fields.
    """
    if tests is None:
        tests = ['ttest', 'mannwhitneyu', 'ks', 'welch']
    
    results = {}
    
    if 'ttest' in tests:
        results['ttest'] = t_test(dist1, dist2, alpha)
    
    if 'mannwhitneyu' in tests:
        results['mannwhitneyu'] = mann_whitney_u_test(dist1, dist2, alpha)
    
    if 'ks' in tests:
        results['kolmogorov_smirnov'] = kolmogorov_smirnov_test(dist1, dist2, alpha)
    
    if 'welch' in tests:
        results['welch_ttest'] = welch_t_test(dist1, dist2, alpha)
    
    return results


def t_test(
    dist1: np.ndarray,
    dist2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform independent samples t-test.
    
    Tests the null hypothesis that two independent samples have identical
    average (expected) values. Assumes equal variances.
    
    Parameters:
    -----------
    dist1 : np.ndarray
        First distribution
    dist2 : np.ndarray
        Second distribution
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing 'statistic', 'pvalue', 'significant', and 'effect_size'
    """
    statistic, pvalue = stats.ttest_ind(dist1, dist2)
    
    # Calculate Cohen's d for effect size
    pooled_std = np.sqrt(((len(dist1) - 1) * np.var(dist1, ddof=1) + 
                          (len(dist2) - 1) * np.var(dist2, ddof=1)) / 
                         (len(dist1) + len(dist2) - 2))
    cohens_d = (np.mean(dist1) - np.mean(dist2)) / pooled_std if pooled_std > 0 else 0.0
    
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < alpha,
        'effect_size': float(cohens_d),
        'interpretation': interpret_cohens_d(cohens_d)
    }


def welch_t_test(
    dist1: np.ndarray,
    dist2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform Welch's t-test (does not assume equal variances).
    
    Parameters:
    -----------
    dist1 : np.ndarray
        First distribution
    dist2 : np.ndarray
        Second distribution
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing 'statistic', 'pvalue', and 'significant'
    """
    statistic, pvalue = stats.ttest_ind(dist1, dist2, equal_var=False)
    
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < alpha
    }


def mann_whitney_u_test(
    dist1: np.ndarray,
    dist2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Perform Mann-Whitney U test (non-parametric test).
    
    Tests whether the distribution underlying sample dist1 is the same as
    the distribution underlying sample dist2. Does not assume normality.
    
    Parameters:
    -----------
    dist1 : np.ndarray
        First distribution
    dist2 : np.ndarray
        Second distribution
    alpha : float, default=0.05
        Significance level
    alternative : str, default='two-sided'
        Defines the alternative hypothesis. Options: 'two-sided', 'less', 'greater'
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing 'statistic', 'pvalue', and 'significant'
    """
    statistic, pvalue = stats.mannwhitneyu(dist1, dist2, alternative=alternative)
    
    # Calculate rank-biserial correlation as effect size
    n1, n2 = len(dist1), len(dist2)
    rank_biserial = 1 - (2 * statistic) / (n1 * n2)
    
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < alpha,
        'effect_size': float(rank_biserial)
    }


def kolmogorov_smirnov_test(
    dist1: np.ndarray,
    dist2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform two-sample Kolmogorov-Smirnov test.
    
    Tests whether two independent samples are drawn from the same
    continuous distribution.
    
    Parameters:
    -----------
    dist1 : np.ndarray
        First distribution
    dist2 : np.ndarray
        Second distribution
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing 'statistic', 'pvalue', and 'significant'
    """
    statistic, pvalue = stats.ks_2samp(dist1, dist2)
    
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < alpha
    }


def levene_test(
    *distributions: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform Levene's test for equality of variances.
    
    Tests the null hypothesis that all input samples are from populations
    with equal variances.
    
    Parameters:
    -----------
    *distributions : np.ndarray
        Variable number of distributions to test
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing 'statistic', 'pvalue', and 'significant'
    """
    statistic, pvalue = stats.levene(*distributions)
    
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < alpha,
        'interpretation': 'Variances are significantly different' if pvalue < alpha 
                         else 'No significant difference in variances'
    }


def anova_test(
    *distributions: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform one-way ANOVA test.
    
    Tests the null hypothesis that all groups have the same population mean.
    
    Parameters:
    -----------
    *distributions : np.ndarray
        Variable number of distributions to test
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing 'statistic', 'pvalue', and 'significant'
    """
    statistic, pvalue = stats.f_oneway(*distributions)
    
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < alpha
    }


def kruskal_wallis_test(
    *distributions: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform Kruskal-Wallis H test (non-parametric version of ANOVA).
    
    Tests whether the population median of all groups are equal.
    
    Parameters:
    -----------
    *distributions : np.ndarray
        Variable number of distributions to test
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing 'statistic', 'pvalue', and 'significant'
    """
    statistic, pvalue = stats.kruskal(*distributions)
    
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < alpha
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    statistic_func : callable, default=np.mean
        Function to compute the statistic (e.g., np.mean, np.median)
    n_bootstrap : int, default=10000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
    --------
    Tuple[float, float, float]
        (lower_bound, statistic, upper_bound)
    """
    bootstrap_statistics = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_statistics.append(statistic_func(sample))
    
    bootstrap_statistics = np.array(bootstrap_statistics)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
    upper_bound = np.percentile(bootstrap_statistics, upper_percentile)
    statistic = statistic_func(data)
    
    return float(lower_bound), float(statistic), float(upper_bound)


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Parameters:
    -----------
    d : float
        Cohen's d value
        
    Returns:
    --------
    str
        Interpretation of the effect size
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def summary_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate summary statistics for a distribution.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing various summary statistics
    """
    return {
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data, ddof=1)),
        'var': float(np.var(data, ddof=1)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),
        'n': len(data)
    }


def print_comparison_report(
    dist1: np.ndarray,
    dist2: np.ndarray,
    label1: str = "Distribution 1",
    label2: str = "Distribution 2",
    alpha: float = 0.05
) -> None:
    """
    Print a comprehensive comparison report for two distributions.
    
    Parameters:
    -----------
    dist1 : np.ndarray
        First distribution
    dist2 : np.ndarray
        Second distribution
    label1 : str, default="Distribution 1"
        Label for first distribution
    label2 : str, default="Distribution 2"
        Label for second distribution
    alpha : float, default=0.05
        Significance level
    """
    print("=" * 80)
    print(f"Statistical Comparison: {label1} vs {label2}")
    print("=" * 80)
    
    print(f"\n{label1} Summary:")
    stats1 = summary_statistics(dist1)
    for key, value in stats1.items():
        print(f"  {key:12s}: {value:.4f}")
    
    print(f"\n{label2} Summary:")
    stats2 = summary_statistics(dist2)
    for key, value in stats2.items():
        print(f"  {key:12s}: {value:.4f}")
    
    print("\n" + "-" * 80)
    print("Statistical Tests")
    print("-" * 80)
    
    # Check for equal variances
    levene_result = levene_test(dist1, dist2, alpha=alpha)
    print(f"\nLevene's Test (Equal Variances):")
    print(f"  Statistic: {levene_result['statistic']:.4f}")
    print(f"  P-value:   {levene_result['pvalue']:.4f}")
    print(f"  Result:    {levene_result['interpretation']}")
    
    # Perform comparison tests
    results = compare_distributions(dist1, dist2, alpha=alpha)
    
    print(f"\nStudent's t-test:")
    ttest = results['ttest']
    print(f"  Statistic:   {ttest['statistic']:.4f}")
    print(f"  P-value:     {ttest['pvalue']:.4f}")
    print(f"  Significant: {ttest['significant']} (α={alpha})")
    print(f"  Cohen's d:   {ttest['effect_size']:.4f} ({ttest['interpretation']})")
    
    print(f"\nWelch's t-test (unequal variances):")
    welch = results['welch_ttest']
    print(f"  Statistic:   {welch['statistic']:.4f}")
    print(f"  P-value:     {welch['pvalue']:.4f}")
    print(f"  Significant: {welch['significant']} (α={alpha})")
    
    print(f"\nMann-Whitney U test:")
    mw = results['mannwhitneyu']
    print(f"  Statistic:   {mw['statistic']:.4f}")
    print(f"  P-value:     {mw['pvalue']:.4f}")
    print(f"  Significant: {mw['significant']} (α={alpha})")
    print(f"  Effect size: {mw['effect_size']:.4f}")
    
    print(f"\nKolmogorov-Smirnov test:")
    ks = results['kolmogorov_smirnov']
    print(f"  Statistic:   {ks['statistic']:.4f}")
    print(f"  P-value:     {ks['pvalue']:.4f}")
    print(f"  Significant: {ks['significant']} (α={alpha})")
    
    # Bootstrap confidence intervals
    print("\n" + "-" * 80)
    print("Bootstrap Confidence Intervals (95%)")
    print("-" * 80)
    
    ci1 = bootstrap_confidence_interval(dist1)
    print(f"\n{label1} Mean:")
    print(f"  95% CI: [{ci1[0]:.4f}, {ci1[2]:.4f}]")
    print(f"  Mean:   {ci1[1]:.4f}")
    
    ci2 = bootstrap_confidence_interval(dist2)
    print(f"\n{label2} Mean:")
    print(f"  95% CI: [{ci2[0]:.4f}, {ci2[2]:.4f}]")
    print(f"  Mean:   {ci2[1]:.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    print("Example: Comparing two normal distributions\n")
    
    # Generate sample data
    np.random.seed(42)
    dist1 = np.random.normal(100, 15, 100)
    dist2 = np.random.normal(110, 15, 100)
    
    # Print comprehensive report
    print_comparison_report(dist1, dist2, "Group A", "Group B")
    
    print("\n\nExample: Comparing three distributions with ANOVA\n")
    dist3 = np.random.normal(105, 15, 100)
    
    anova_result = anova_test(dist1, dist2, dist3)
    print(f"One-Way ANOVA:")
    print(f"  F-statistic: {anova_result['statistic']:.4f}")
    print(f"  P-value:     {anova_result['pvalue']:.4f}")
    print(f"  Significant: {anova_result['significant']}")
    
    kw_result = kruskal_wallis_test(dist1, dist2, dist3)
    print(f"\nKruskal-Wallis H test:")
    print(f"  H-statistic: {kw_result['statistic']:.4f}")
    print(f"  P-value:     {kw_result['pvalue']:.4f}")
    print(f"  Significant: {kw_result['significant']}")

