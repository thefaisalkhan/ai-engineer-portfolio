"""
Phase 1 — Week 2: Hypothesis Testing
=====================================
Covers: t-tests, ANOVA, chi-square, p-values, Type I/II errors
Job relevance: A/B testing (82%), statistical modeling (94%) in job postings
"""

import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
from typing import Literal


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    reject_null: bool
    effect_size: Optional[float]
    interpretation: str

    def __post_init__(self):
        pass


# ── One-sample t-test ──────────────────────────────────────────────────────────
def one_sample_t_test(
    data: np.ndarray,
    population_mean: float,
    alpha: float = 0.05,
) -> TestResult:
    """Test if sample mean differs from a known population mean."""
    t_stat, p_val = stats.ttest_1samp(data, popmean=population_mean)
    # Cohen's d effect size
    d = (np.mean(data) - population_mean) / np.std(data, ddof=1)
    return TestResult(
        test_name="One-sample t-test",
        statistic=float(t_stat),
        p_value=float(p_val),
        reject_null=p_val < alpha,
        effect_size=float(d),
        interpretation=(
            f"Sample mean ({np.mean(data):.3f}) significantly differs from "
            f"population mean ({population_mean}): {'YES' if p_val < alpha else 'NO'} "
            f"(p={p_val:.4f}, d={d:.3f})"
        ),
    )


# ── Two-sample t-test ─────────────────────────────────────────────────────────
def two_sample_t_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    equal_var: bool = False,  # Welch's by default (more robust)
    alpha: float = 0.05,
) -> TestResult:
    """Compare means of two independent groups (e.g., A/B test variants)."""
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=equal_var)
    # Pooled Cohen's d
    pooled_std = np.sqrt(
        (np.std(group_a, ddof=1) ** 2 + np.std(group_b, ddof=1) ** 2) / 2
    )
    d = (np.mean(group_a) - np.mean(group_b)) / pooled_std
    return TestResult(
        test_name="Two-sample t-test (Welch's)" if not equal_var else "Two-sample t-test",
        statistic=float(t_stat),
        p_value=float(p_val),
        reject_null=p_val < alpha,
        effect_size=float(d),
        interpretation=(
            f"Groups differ significantly: {'YES' if p_val < alpha else 'NO'} "
            f"(p={p_val:.4f}). Effect size d={d:.3f} "
            f"({'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'})"
        ),
    )


# ── One-way ANOVA ──────────────────────────────────────────────────────────────
def one_way_anova(*groups: np.ndarray, alpha: float = 0.05) -> TestResult:
    """
    Test if 3+ group means are equal.
    Null: all group means are equal.
    Alternative: at least one group mean differs.
    """
    f_stat, p_val = stats.f_oneway(*groups)
    # Eta-squared effect size
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
    )
    ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    return TestResult(
        test_name="One-way ANOVA",
        statistic=float(f_stat),
        p_value=float(p_val),
        reject_null=p_val < alpha,
        effect_size=float(eta_sq),
        interpretation=(
            f"At least one group mean differs: {'YES' if p_val < alpha else 'NO'} "
            f"(F={f_stat:.3f}, p={p_val:.4f}, η²={eta_sq:.3f})"
        ),
    )


# ── Chi-square test ────────────────────────────────────────────────────────────
def chi_square_test(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> TestResult:
    """
    Goodness-of-fit: test if observed frequencies match expected.
    Independence: test if two categorical variables are independent.
    """
    chi2_stat, p_val = stats.chisquare(observed, f_exp=expected)
    return TestResult(
        test_name="Chi-square test",
        statistic=float(chi2_stat),
        p_value=float(p_val),
        reject_null=p_val < alpha,
        effect_size=None,
        interpretation=(
            f"Observed distribution matches expected: {'NO' if p_val < alpha else 'YES'} "
            f"(χ²={chi2_stat:.3f}, p={p_val:.4f})"
        ),
    )


# ── Type I & II errors ─────────────────────────────────────────────────────────
def type_errors_demo():
    """
    Illustrate Type I (false positive) and Type II (false negative) errors
    through Monte Carlo simulation.
    """
    rng = np.random.default_rng(seed=42)
    alpha = 0.05
    n_simulations = 10_000
    n = 30

    # Type I error: null is TRUE, measure false rejection rate
    null_true_rejections = sum(
        stats.ttest_ind(
            rng.normal(0, 1, n), rng.normal(0, 1, n)
        ).pvalue < alpha
        for _ in range(n_simulations)
    )
    type_i_rate = null_true_rejections / n_simulations

    # Type II error: null is FALSE (true diff = 0.5 SD), measure miss rate
    null_false_rejections = sum(
        stats.ttest_ind(
            rng.normal(0, 1, n), rng.normal(0.5, 1, n)
        ).pvalue < alpha
        for _ in range(n_simulations)
    )
    power = null_false_rejections / n_simulations
    type_ii_rate = 1 - power

    return {
        "Type I error rate (α)": type_i_rate,
        "Expected α": alpha,
        "Type II error rate (β)": type_ii_rate,
        "Power (1-β)": power,
        "Note": "Power < 0.80 → increase sample size",
    }


if __name__ == "__main__":
    from typing import Optional

    rng = np.random.default_rng(seed=42)

    print("=== One-Sample t-test ===")
    data = rng.normal(loc=10.5, scale=2.0, size=50)
    result = one_sample_t_test(data, population_mean=10.0)
    print(result.interpretation)

    print("\n=== Two-Sample t-test (A/B test scenario) ===")
    control = rng.normal(loc=5.0, scale=1.5, size=200)
    treatment = rng.normal(loc=5.4, scale=1.5, size=200)
    result = two_sample_t_test(control, treatment)
    print(result.interpretation)

    print("\n=== One-Way ANOVA (3 model variants) ===")
    model_a = rng.normal(0.82, 0.05, 100)
    model_b = rng.normal(0.85, 0.05, 100)
    model_c = rng.normal(0.79, 0.05, 100)
    result = one_way_anova(model_a, model_b, model_c)
    print(result.interpretation)

    print("\n=== Type I & II Error Rates ===")
    errors = type_errors_demo()
    for k, v in errors.items():
        print(f"  {k}: {v}")
