"""
Phase 1 — Week 3: Statistical Modeling
========================================
Covers: OLS regression, logistic regression, model diagnostics, multicollinearity
Job relevance: "Statistical models and predictive analytics" — 94% of job postings
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class OLSReport:
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    aic: float
    bic: float
    coefficients: dict
    residuals_normal: bool  # Jarque-Bera test


def ols_regression(X: pd.DataFrame, y: pd.Series) -> OLSReport:
    """
    Ordinary Least Squares regression with full statistical diagnostics.
    This is what 'statistical modeling' means in job descriptions.
    """
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # Jarque-Bera test for residual normality
    jb_stat, jb_p, _, _ = sm.stats.stattools.jarque_bera(model.resid)

    coefficients = {
        name: {
            "coef": float(model.params[name]),
            "std_err": float(model.bse[name]),
            "t_stat": float(model.tvalues[name]),
            "p_value": float(model.pvalues[name]),
            "significant": model.pvalues[name] < 0.05,
        }
        for name in model.params.index
    }

    return OLSReport(
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=float(model.fvalue),
        f_pvalue=float(model.f_pvalue),
        aic=float(model.aic),
        bic=float(model.bic),
        coefficients=coefficients,
        residuals_normal=jb_p > 0.05,
    )


def check_multicollinearity(X: pd.DataFrame) -> pd.DataFrame:
    """
    Variance Inflation Factor (VIF) — detect multicollinearity.
    VIF > 5: concern; VIF > 10: serious problem.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame({
        "feature": X_with_const.columns,
        "VIF": [
            variance_inflation_factor(X_with_const.values, i)
            for i in range(X_with_const.shape[1])
        ],
    })
    vif_data["concern"] = vif_data["VIF"].apply(
        lambda v: "none" if v < 5 else "moderate" if v < 10 else "high"
    )
    return vif_data.sort_values("VIF", ascending=False)


def logistic_regression_statsmodels(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Logistic regression with odds ratios, confidence intervals.
    Statsmodels gives the full statistical picture (not just accuracy).
    """
    X_with_const = sm.add_constant(X)
    model = sm.Logit(y, X_with_const).fit(disp=0)

    # Odds ratios + 95% CI
    odds_ratios = np.exp(model.params)
    conf = np.exp(model.conf_int())
    conf.columns = ["OR_lower_95", "OR_upper_95"]

    summary = pd.concat([odds_ratios.rename("odds_ratio"), conf], axis=1)
    summary["p_value"] = model.pvalues
    summary["significant"] = model.pvalues < 0.05

    return {
        "pseudo_r_squared": float(model.prsquared),
        "log_likelihood": float(model.llf),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "n_obs": int(model.nobs),
        "odds_ratios": summary,
    }


def residual_diagnostics(model_resid: np.ndarray) -> dict:
    """
    Four key residual checks for OLS assumptions:
    1. Normality (Jarque-Bera)
    2. Homoscedasticity (Breusch-Pagan)
    3. Autocorrelation (Durbin-Watson)
    4. Outliers (standardized residuals)
    """
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan

    jb_stat, jb_p, _, _ = sm.stats.stattools.jarque_bera(model_resid)
    dw = durbin_watson(model_resid)

    standardized = (model_resid - model_resid.mean()) / model_resid.std()
    n_outliers = int(np.sum(np.abs(standardized) > 3))

    return {
        "normality_jarque_bera_p": float(jb_p),
        "normality_ok": jb_p > 0.05,
        "durbin_watson": float(dw),
        "autocorrelation_ok": 1.5 < dw < 2.5,
        "n_outliers_3sigma": n_outliers,
    }


if __name__ == "__main__":
    # ── OLS on California Housing ──────────────────────────────────────────────
    print("=== OLS Regression: California Housing ===")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    features = ["MedInc", "HouseAge", "AveRooms", "AveOccup", "Latitude"]
    X = df[features]
    y = df["MedHouseVal"]

    report = ols_regression(X, y)
    print(f"R²: {report.r_squared:.4f}")
    print(f"Adj R²: {report.adj_r_squared:.4f}")
    print(f"F-stat: {report.f_statistic:.2f} (p={report.f_pvalue:.2e})")
    print(f"Residuals normal: {report.residuals_normal}")
    print("\nCoefficients:")
    for feat, vals in report.coefficients.items():
        sig = "**" if vals["significant"] else "  "
        print(f"  {sig} {feat:15s} coef={vals['coef']:8.4f}  p={vals['p_value']:.4f}")

    print("\n=== Multicollinearity Check (VIF) ===")
    vif = check_multicollinearity(X)
    print(vif.to_string(index=False))

    # ── Logistic regression on breast cancer ──────────────────────────────────
    print("\n=== Logistic Regression: Breast Cancer ===")
    cancer = load_breast_cancer(as_frame=True)
    Xc = cancer.frame[["mean radius", "mean texture", "mean perimeter"]].copy()
    Xc.columns = ["radius", "texture", "perimeter"]
    scaler = StandardScaler()
    Xc_scaled = pd.DataFrame(scaler.fit_transform(Xc), columns=Xc.columns)
    yc = cancer.target

    lr = logistic_regression_statsmodels(Xc_scaled, pd.Series(yc))
    print(f"Pseudo R²: {lr['pseudo_r_squared']:.4f}")
    print(f"AIC: {lr['aic']:.2f}")
    print("\nOdds Ratios:")
    print(lr["odds_ratios"].to_string())
