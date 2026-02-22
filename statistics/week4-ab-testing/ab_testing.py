"""
Phase 1 — Week 4: A/B Testing & Experiment Design
==================================================
Covers: power analysis, sample size, sequential testing, multiple comparisons
Job relevance: A/B testing (82%), experiment design — standard ML interview topic
"""

import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
from typing import Literal


@dataclass
class SampleSizeResult:
    n_per_group: int
    total_n: int
    power: float
    alpha: float
    effect_size: float
    minimum_detectable_effect: float


@dataclass
class ABTestResult:
    control_mean: float
    treatment_mean: float
    relative_lift: float
    p_value: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    power_achieved: float
    recommendation: str


def compute_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,  # relative, e.g. 0.05 = 5% lift
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
) -> SampleSizeResult:
    """
    Calculate minimum sample size for an A/B test.
    This is a standard ML/data science interview question.
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)

    # Cohen's h for proportions
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_tailed else 1))
    z_beta = stats.norm.ppf(power)

    n = ((z_alpha + z_beta) / h) ** 2
    n_per_group = int(np.ceil(n))

    return SampleSizeResult(
        n_per_group=n_per_group,
        total_n=2 * n_per_group,
        power=power,
        alpha=alpha,
        effect_size=float(abs(h)),
        minimum_detectable_effect=minimum_detectable_effect,
    )


def run_ab_test(
    control: np.ndarray,
    treatment: np.ndarray,
    metric_type: Literal["proportion", "continuous"] = "continuous",
    alpha: float = 0.05,
) -> ABTestResult:
    """Run a two-sample A/B test and return a complete result."""
    ctrl_mean = float(np.mean(control))
    trt_mean = float(np.mean(treatment))
    relative_lift = (trt_mean - ctrl_mean) / ctrl_mean if ctrl_mean != 0 else 0.0

    if metric_type == "proportion":
        n1, n2 = len(control), len(treatment)
        p1, p2 = ctrl_mean, trt_mean
        p_pool = (np.sum(control) + np.sum(treatment)) / (n1 + n2)
        z_stat = (p2 - p1) / np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    else:
        _, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    # 95% CI for the difference
    diff = trt_mean - ctrl_mean
    se_diff = np.sqrt(
        np.var(control, ddof=1) / len(control)
        + np.var(treatment, ddof=1) / len(treatment)
    )
    margin = stats.norm.ppf(1 - alpha / 2) * se_diff
    ci = (float(diff - margin), float(diff + margin))

    # Post-hoc power
    pooled_std = np.sqrt(
        (np.std(control, ddof=1) ** 2 + np.std(treatment, ddof=1) ** 2) / 2
    )
    effect_size = abs(diff) / pooled_std if pooled_std > 0 else 0.0
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    power = float(1 - stats.norm.cdf(z_alpha - effect_size * np.sqrt(len(control) / 2)))

    is_significant = p_value < alpha
    if is_significant and relative_lift > 0:
        rec = f"Launch treatment — {relative_lift:.1%} lift is statistically significant."
    elif is_significant and relative_lift <= 0:
        rec = "Do NOT launch — treatment performs worse (statistically significant)."
    else:
        rec = f"Inconclusive — p={p_value:.4f} > α={alpha}. Collect more data."

    return ABTestResult(
        control_mean=ctrl_mean,
        treatment_mean=trt_mean,
        relative_lift=relative_lift,
        p_value=float(p_value),
        confidence_interval=ci,
        is_significant=is_significant,
        power_achieved=power,
        recommendation=rec,
    )


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Correct for multiple comparisons (testing multiple variants simultaneously)."""
    adjusted_alpha = alpha / len(p_values)
    return [p < adjusted_alpha for p in p_values]


def sequential_probability_ratio_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
    beta: float = 0.20,
) -> dict:
    """
    SPRT allows you to stop an A/B test early with controlled error rates.
    Used in 'always-valid inference' / peeking-safe testing.
    """
    p0 = control_conversions / control_n
    p1 = treatment_conversions / treatment_n

    # Log likelihood ratio
    if p0 == 0 or p1 == 0 or p0 == 1 or p1 == 1:
        return {"status": "insufficient_data", "action": "continue"}

    llr = (
        treatment_conversions * np.log(p1 / p0)
        + (treatment_n - treatment_conversions) * np.log((1 - p1) / (1 - p0))
    )

    upper_bound = np.log((1 - beta) / alpha)
    lower_bound = np.log(beta / (1 - alpha))

    if llr >= upper_bound:
        action = "STOP — reject null (treatment wins)"
    elif llr <= lower_bound:
        action = "STOP — accept null (no difference)"
    else:
        action = "CONTINUE collecting data"

    return {
        "log_likelihood_ratio": float(llr),
        "upper_bound": float(upper_bound),
        "lower_bound": float(lower_bound),
        "action": action,
        "control_rate": float(p0),
        "treatment_rate": float(p1),
        "observed_lift": float((p1 - p0) / p0) if p0 > 0 else 0.0,
    }


if __name__ == "__main__":
    print("=== Sample Size Calculation ===")
    ss = compute_sample_size(
        baseline_rate=0.10,         # 10% conversion rate
        minimum_detectable_effect=0.10,  # detect 10% relative lift (0.10 → 0.11)
        alpha=0.05,
        power=0.80,
    )
    print(f"n per group: {ss.n_per_group:,}")
    print(f"Total n:     {ss.total_n:,}")
    print(f"Effect size (Cohen's h): {ss.effect_size:.4f}")

    print("\n=== A/B Test Results ===")
    rng = np.random.default_rng(seed=42)
    control = rng.normal(loc=50.0, scale=10.0, size=ss.n_per_group)
    treatment = rng.normal(loc=53.0, scale=10.0, size=ss.n_per_group)  # 6% lift
    result = run_ab_test(control, treatment)
    print(f"Control mean:   {result.control_mean:.4f}")
    print(f"Treatment mean: {result.treatment_mean:.4f}")
    print(f"Relative lift:  {result.relative_lift:.2%}")
    print(f"p-value:        {result.p_value:.4f}")
    print(f"95% CI diff:    ({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})")
    print(f"Power achieved: {result.power_achieved:.2%}")
    print(f"Recommendation: {result.recommendation}")

    print("\n=== Multiple Comparisons (Bonferroni) ===")
    p_values = [0.03, 0.04, 0.001, 0.08, 0.02]
    significant = bonferroni_correction(p_values)
    for p, sig in zip(p_values, significant):
        print(f"  p={p:.3f} → {'SIGNIFICANT' if sig else 'not significant'} after correction")

    print("\n=== Sequential Probability Ratio Test (SPRT) ===")
    sprt = sequential_probability_ratio_test(
        control_conversions=480, control_n=5000,
        treatment_conversions=530, treatment_n=5000,
    )
    for k, v in sprt.items():
        print(f"  {k}: {v}")
