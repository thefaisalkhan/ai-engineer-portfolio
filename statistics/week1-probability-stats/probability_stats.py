"""
Phase 1 — Week 1: Probability & Descriptive Statistics
=======================================================
Covers: distributions, central limit theorem, Bayes' theorem, sampling
Job relevance: Statistical modeling (94% of AI engineer job postings)
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributionSummary:
    name: str
    mean: float
    std: float
    skewness: float
    kurtosis: float


def descriptive_stats(data: np.ndarray) -> DistributionSummary:
    """Compute full descriptive statistics for a dataset."""
    return DistributionSummary(
        name="sample",
        mean=float(np.mean(data)),
        std=float(np.std(data, ddof=1)),
        skewness=float(stats.skew(data)),
        kurtosis=float(stats.kurtosis(data)),
    )


def demonstrate_central_limit_theorem(
    population_size: int = 100_000,
    sample_size: int = 50,
    n_samples: int = 1000,
) -> dict:
    """
    CLT: sampling distribution of the mean approaches normal
    regardless of the underlying population distribution.
    """
    # Skewed population (exponential)
    population = np.random.exponential(scale=2.0, size=population_size)

    sample_means = [
        np.mean(np.random.choice(population, size=sample_size, replace=False))
        for _ in range(n_samples)
    ]

    _, p_value = stats.shapiro(sample_means[:50])  # Shapiro-Wilk on subset

    return {
        "population_mean": float(np.mean(population)),
        "population_std": float(np.std(population)),
        "sample_means_mean": float(np.mean(sample_means)),
        "sample_means_std": float(np.std(sample_means)),
        "expected_se": float(np.std(population) / np.sqrt(sample_size)),
        "shapiro_p_value": float(p_value),
        "is_approximately_normal": p_value > 0.05,
    }


def bayes_theorem(
    prior: float,
    likelihood: float,
    marginal_likelihood: float,
) -> float:
    """
    P(A|B) = P(B|A) * P(A) / P(B)

    Example: disease detection
    - prior = P(disease) base rate
    - likelihood = P(positive test | disease) sensitivity
    - marginal_likelihood = P(positive test) overall
    """
    posterior = (likelihood * prior) / marginal_likelihood
    return posterior


def medical_test_example() -> dict:
    """
    Classic Bayes problem: disease prevalence + test accuracy → posterior.
    Demonstrates why low-prevalence screening yields many false positives.
    """
    prevalence = 0.001          # P(disease) = 0.1%
    sensitivity = 0.99          # P(+test | disease)
    specificity = 0.95          # P(-test | no disease)

    false_positive_rate = 1 - specificity
    # P(+test) = P(+test|disease)*P(disease) + P(+test|no disease)*P(no disease)
    marginal = (sensitivity * prevalence) + (false_positive_rate * (1 - prevalence))

    posterior = bayes_theorem(prevalence, sensitivity, marginal)

    return {
        "P(disease)": prevalence,
        "P(positive | disease)": sensitivity,
        "P(positive)": marginal,
        "P(disease | positive)": posterior,
        "insight": f"Even with 99% sensitivity, only {posterior:.1%} of positive tests indicate disease",
    }


def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval using t-distribution."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    margin = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(mean - margin), float(mean + margin)


def common_distributions() -> dict:
    """
    Generate samples from distributions used in ML:
    Normal, Binomial, Poisson, Exponential, Beta.
    """
    rng = np.random.default_rng(seed=42)
    return {
        "normal": rng.normal(loc=0, scale=1, size=1000),
        "binomial": rng.binomial(n=10, p=0.3, size=1000),
        "poisson": rng.poisson(lam=3, size=1000),
        "exponential": rng.exponential(scale=1.5, size=1000),
        "beta": rng.beta(a=2, b=5, size=1000),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    data = rng.normal(loc=10, scale=2.5, size=500)

    summary = descriptive_stats(data)
    print("=== Descriptive Statistics ===")
    print(f"Mean:     {summary.mean:.4f}")
    print(f"Std:      {summary.std:.4f}")
    print(f"Skewness: {summary.skewness:.4f}")
    print(f"Kurtosis: {summary.kurtosis:.4f}")

    ci_low, ci_high = confidence_interval(data)
    print(f"\n95% CI: ({ci_low:.4f}, {ci_high:.4f})")

    print("\n=== Central Limit Theorem Demo ===")
    clt = demonstrate_central_limit_theorem()
    for k, v in clt.items():
        print(f"  {k}: {v}")

    print("\n=== Bayes' Theorem: Medical Test ===")
    result = medical_test_example()
    for k, v in result.items():
        print(f"  {k}: {v}")
