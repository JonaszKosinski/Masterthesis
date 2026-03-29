import warnings
import os

import matplotlib
matplotlib.use("Agg")   # avoids VS Code/backend plotting issues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot

# ============================================================
# SUPPRESS WARNINGS
# ============================================================
warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================
MASTER_PATH = "DATA/Processed/armax_master_dataframe.xlsx"
OUT_PATH_SUMMARY = "DATA/Processed/armax_selected_results.xlsx"
OUT_PATH_COEFS = "DATA/Processed/armax_selected_coefficients.xlsx"

COUNTRIES = ["Thailand", "Philippines", "Korea", "Indonesia"]

# ----------------------------------------------------------------
# FINAL ARMAX STRUCTURE
# ----------------------------------------------------------------
MODEL_SPEC = {
    "Thailand": (1, 0, 1),
    "Philippines": (1, 0, 1),
    "Korea": (1, 0, 0),
    "Indonesia": (2, 0, 1)
}

# ----------------------------------------------------------------
# Exogenous variable suffixes
# ----------------------------------------------------------------
EXOG_SUFFIXES = [
    "dlog_m2",
    "dlog_exr",
    "dlog_res",
    "d_ltr",
    "d_str"
]

MAX_LAG = 6
SIGNIFICANCE_LEVEL = 0.10
REMOVE_ONLY_EXOG = True
MIN_EXOG_TERMS = 0

# ----------------------------------------------------------------
# AUTOMATIC OUTLIER DUMMIES
# Thailand only
# ----------------------------------------------------------------
AUTO_OUTLIER_DUMMIES = True
OUTLIER_COUNTRIES = ["Thailand"]
OUTLIER_Z_THRESHOLD = 3.0
MAX_OUTLIER_DUMMIES = 3

# ============================================================
# LOAD DATA
# ============================================================
def load_master(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.isna()]
    return df

# ============================================================
# SERIES HELPERS
# ============================================================
def get_series(master: pd.DataFrame, column: str) -> pd.Series:
    if column not in master.columns:
        raise ValueError(f"Column not found: {column}")

    s = pd.to_numeric(master[column], errors="coerce").copy()
    s.index = pd.DatetimeIndex(master.index)
    s = s.asfreq("MS")
    return s

def get_inflation(master: pd.DataFrame, country: str) -> pd.Series:
    return get_series(master, f"{country}_infl")

# ============================================================
# BUILD EXOG MATRIX WITH LAGS 1..MAX_LAG
# ============================================================
def build_exog_lags(master: pd.DataFrame, country: str, suffixes: list, max_lag: int) -> pd.DataFrame:
    exog_parts = []

    for suffix in suffixes:
        col = f"{country}_{suffix}"
        s = get_series(master, col)

        for lag in range(1, max_lag + 1):
            exog_parts.append(s.shift(lag).rename(f"{suffix}_L{lag}"))

    exog = pd.concat(exog_parts, axis=1)
    return exog

# ============================================================
# PREPARE COUNTRY DATASET
# ============================================================
def build_country_dataset(master: pd.DataFrame, country: str, suffixes: list, max_lag: int):
    y = get_inflation(master, country)
    X = build_exog_lags(master, country, suffixes, max_lag)

    data = pd.concat([y.rename("infl"), X], axis=1).dropna()

    if data.empty:
        raise ValueError(
            f"{country}: dataset is empty after aligning inflation and lagged exogenous variables."
        )

    y_final = data["infl"].copy()
    X_final = data.drop(columns=["infl"]).copy()

    return y_final, X_final

# ============================================================
# FIT MODEL
# ============================================================
def fit_armax(y: pd.Series, X: pd.DataFrame | None, order: tuple, country: str, verbose: bool = True):
    p, d, q = order

    exog_input = None
    n_exog = 0

    if X is not None and X.shape[1] > 0:
        exog_input = X
        n_exog = X.shape[1]

    model = ARIMA(
        endog=y,
        exog=exog_input,
        order=(p, d, q),
        trend="c"
    )

    result = model.fit(method_kwargs={"maxiter": 1000})

    if verbose:
        print("\n" + "=" * 70)
        print(f"{country} | ARIMA({p},{d},{q}) + exog | n_exog = {n_exog}")
        print("=" * 70)
        print(result.summary())

    return result

# ============================================================
# STEPWISE BACKWARD ELIMINATION OF EXOG TERMS
# ============================================================
def prune_insignificant_exog(
    y: pd.Series,
    X: pd.DataFrame,
    order: tuple,
    country: str,
    alpha: float = 0.10,
    remove_only_exog: bool = True,
    min_exog_terms: int = 0
):
    current_X = X.copy()
    removed_terms = []
    step = 0

    while True:
        step += 1
        print(f"\n[{country}] Step {step}: fitting with {current_X.shape[1]} exogenous terms...")

        res = fit_armax(y, current_X, order, country, verbose=True)
        pvalues = res.pvalues.copy()

        protected_terms = {"const", "sigma2"}

        if remove_only_exog:
            for name in pvalues.index:
                if name.startswith("ar.") or name.startswith("ma."):
                    protected_terms.add(name)

        candidate_pvals = pvalues.drop(
            labels=[term for term in protected_terms if term in pvalues.index],
            errors="ignore"
        )

        if current_X.shape[1] > 0:
            candidate_pvals = candidate_pvals[candidate_pvals.index.isin(current_X.columns)]
        else:
            candidate_pvals = pd.Series(dtype=float)

        if candidate_pvals.empty:
            return res, current_X, removed_terms

        worst_term = candidate_pvals.idxmax()
        worst_p = candidate_pvals.max()

        if worst_p <= alpha:
            return res, current_X, removed_terms

        if current_X.shape[1] <= min_exog_terms:
            return res, current_X, removed_terms

        current_X = current_X.drop(columns=[worst_term])
        removed_terms.append((worst_term, worst_p))

        print(f"[{country}] Removing: {worst_term} (p-value = {worst_p:.4f})")

        if current_X.shape[1] == 0:
            print(f"[{country}] All exogenous terms removed. Refitting pure ARIMA({order[0]},{order[1]},{order[2]}).")
            res = fit_armax(y, None, order, country, verbose=True)
            return res, pd.DataFrame(index=y.index), removed_terms

# ============================================================
# OUTLIER DETECTION AND DUMMY CREATION
# ============================================================
def detect_outlier_dates_from_residuals(
    res,
    threshold: float = 3.0,
    max_dummies: int = 3
):
    resid = pd.Series(res.resid).dropna()

    if len(resid) < 10:
        return []

    std_resid = (resid - resid.mean()) / resid.std(ddof=1)
    candidates = std_resid[std_resid.abs() > threshold]

    if candidates.empty:
        return []

    # strongest outliers first
    candidates = candidates.reindex(candidates.abs().sort_values(ascending=False).index)
    selected_dates = list(candidates.index[:max_dummies])

    print("\nDetected outlier dates based on standardized residuals:")
    for dt in selected_dates:
        print(f"  {dt.strftime('%Y-%m-%d')} | z = {std_resid.loc[dt]:.3f} | resid = {resid.loc[dt]:.6f}")

    return selected_dates

def build_outlier_dummies(index: pd.DatetimeIndex, outlier_dates: list, prefix: str = "outlier") -> pd.DataFrame:
    dummies = []

    for i, dt in enumerate(outlier_dates, start=1):
        s = pd.Series(0.0, index=index)
        if dt in s.index:
            s.loc[dt] = 1.0
        s.name = f"{prefix}_{i}"
        dummies.append(s)

    if not dummies:
        return pd.DataFrame(index=index)

    return pd.concat(dummies, axis=1)

def refit_with_outlier_dummies(
    y: pd.Series,
    X: pd.DataFrame,
    order: tuple,
    country: str,
    alpha: float,
    remove_only_exog: bool,
    min_exog_terms: int,
    threshold: float = 3.0,
    max_dummies: int = 3
):
    # Initial fit
    initial_res, initial_X, initial_removed = prune_insignificant_exog(
        y=y,
        X=X,
        order=order,
        country=country,
        alpha=alpha,
        remove_only_exog=remove_only_exog,
        min_exog_terms=min_exog_terms
    )

    # Detect outliers from initial residuals
    outlier_dates = detect_outlier_dates_from_residuals(
        initial_res,
        threshold=threshold,
        max_dummies=max_dummies
    )

    if not outlier_dates:
        print(f"\n[{country}] No outliers detected above |z| > {threshold}.")
        return initial_res, initial_X, initial_removed, []

    # Build dummies and append them to original X
    dummy_df = build_outlier_dummies(
        index=y.index,
        outlier_dates=outlier_dates,
        prefix=f"{country.lower()}_outlier"
    )

    X_augmented = pd.concat([X, dummy_df], axis=1)

    print(f"\n[{country}] Refitting with {dummy_df.shape[1]} outlier dummy variable(s)...")

    final_res, final_X, final_removed = prune_insignificant_exog(
        y=y,
        X=X_augmented,
        order=order,
        country=country,
        alpha=alpha,
        remove_only_exog=remove_only_exog,
        min_exog_terms=min_exog_terms
    )

    return final_res, final_X, final_removed, outlier_dates

# ============================================================
# DIAGNOSTICS
# ============================================================
def compute_diagnostics(res):
    resid = pd.Series(res.resid).dropna()

    lb_12 = np.nan
    lb_24 = np.nan

    try:
        if len(resid) > 12:
            lb_12 = acorr_ljungbox(resid, lags=[12], return_df=True)["lb_pvalue"].iloc[0]
        if len(resid) > 24:
            lb_24 = acorr_ljungbox(resid, lags=[24], return_df=True)["lb_pvalue"].iloc[0]
    except Exception:
        pass

    return {
        "resid_mean": resid.mean() if len(resid) > 0 else np.nan,
        "resid_std": resid.std() if len(resid) > 1 else np.nan,
        "ljungbox_p_12": lb_12,
        "ljungbox_p_24": lb_24,
        "converged": getattr(res, "mle_retvals", {}).get("converged", np.nan)
    }

# ============================================================
# EXTRACT RESULTS
# ============================================================
def extract_model_summary(
    country: str,
    order: tuple,
    res,
    selected_X: pd.DataFrame,
    removed_terms: list,
    outlier_dates_used: list | None = None
):
    p, d, q = order
    diag = compute_diagnostics(res)

    return {
        "country": country,
        "model": f"ARIMA({p},{d},{q}) + exog",
        "n_obs": int(res.nobs),
        "n_selected_exog": int(selected_X.shape[1]),
        "selected_exog_terms": ", ".join(selected_X.columns.tolist()) if selected_X.shape[1] > 0 else "",
        "removed_terms": ", ".join([f"{name} (p={pv:.3f})" for name, pv in removed_terms]) if removed_terms else "",
        "outlier_dates": ", ".join([dt.strftime("%Y-%m-%d") for dt in outlier_dates_used]) if outlier_dates_used else "",
        "const": res.params.get("const", np.nan),
        "aic": res.aic,
        "bic": res.bic,
        "hqic": getattr(res, "hqic", np.nan),
        "loglik": res.llf,
        "resid_mean": diag["resid_mean"],
        "resid_std": diag["resid_std"],
        "ljungbox_p_12": diag["ljungbox_p_12"],
        "ljungbox_p_24": diag["ljungbox_p_24"],
        "converged": diag["converged"]
    }

def extract_coefficients(country: str, res):
    rows = []

    for param in res.params.index:
        if param == "sigma2":
            continue

        rows.append({
            "country": country,
            "term": param,
            "coef": res.params.get(param, np.nan),
            "std_err": res.bse.get(param, np.nan),
            "p_value": res.pvalues.get(param, np.nan),
            "t_or_z": res.tvalues.get(param, np.nan)
        })

    return rows

# ============================================================
# SAVE RESIDUALS TO EXCEL
# ============================================================
def save_residuals_to_excel(res, country: str, out_dir: str = "DATA/Processed/residual_diagnostics"):
    os.makedirs(out_dir, exist_ok=True)

    resid = pd.Series(res.resid).dropna()
    out = pd.DataFrame({
        "date": resid.index,
        "residual": resid.values,
        "squared_residual": resid.values ** 2
    })

    save_path = os.path.join(out_dir, f"{country}_residuals.xlsx")
    out.to_excel(save_path, index=False)
    print(f"Saved residuals to: {save_path}")

# ============================================================
# PLOT RESIDUAL DIAGNOSTICS
# ============================================================
def plot_residual_diagnostics(res, country: str, out_dir: str = "DATA/Processed/residual_diagnostics"):
    os.makedirs(out_dir, exist_ok=True)

    resid = pd.Series(res.resid).dropna()

    if len(resid) < 5:
        print(f"Not enough residual observations to plot diagnostics for {country}.")
        return

    # -------- Combined 2x2 diagnostics figure --------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Residuals over time
    axes[0, 0].plot(resid.index, resid.values)
    axes[0, 0].axhline(0, linestyle="--")
    axes[0, 0].set_title(f"{country}: residuals over time")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Residual")

    # Residual ACF
    max_acf_lag = min(24, max(1, len(resid) // 2 - 1))
    plot_acf(resid, lags=max_acf_lag, ax=axes[0, 1])
    axes[0, 1].set_title(f"{country}: residual ACF")

    # Histogram
    axes[1, 0].hist(resid, bins=20)
    axes[1, 0].set_title(f"{country}: residual histogram")
    axes[1, 0].set_xlabel("Residual")
    axes[1, 0].set_ylabel("Frequency")

    # Squared residuals
    axes[1, 1].plot(resid.index, resid.values ** 2)
    axes[1, 1].set_title(f"{country}: squared residuals over time")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Residual squared")

    plt.tight_layout()
    combined_path = os.path.join(out_dir, f"{country}_residual_diagnostics.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined residual diagnostics plot to: {combined_path}")

    # -------- Separate residual line plot --------
    plt.figure(figsize=(12, 5))
    plt.plot(resid.index, resid.values)
    plt.axhline(0, linestyle="--")
    plt.title(f"{country}: residuals over time")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.tight_layout()
    resid_line_path = os.path.join(out_dir, f"{country}_residuals_over_time.png")
    plt.savefig(resid_line_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved residual time plot to: {resid_line_path}")

    # -------- Separate residual ACF --------
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    plot_acf(resid, lags=max_acf_lag, ax=ax)
    plt.title(f"{country}: residual ACF")
    plt.tight_layout()
    acf_path = os.path.join(out_dir, f"{country}_residual_acf.png")
    plt.savefig(acf_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved residual ACF plot to: {acf_path}")

    # -------- Separate Q-Q plot --------
    std_resid = (resid - resid.mean()) / resid.std(ddof=1)

    plt.figure(figsize=(6, 6))
    qqplot(std_resid, line="s", ax=plt.gca())
    plt.title(f"{country}: residual Q-Q plot")
    plt.tight_layout()
    qq_path = os.path.join(out_dir, f"{country}_residual_qqplot.png")
    plt.savefig(qq_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved Q-Q plot to: {qq_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    master = load_master(MASTER_PATH)

    summary_rows = []
    coef_rows = []

    for country in COUNTRIES:
        print("\n" + "#" * 90)
        print(f"PROCESSING: {country}")
        print("#" * 90)

        try:
            y, X = build_country_dataset(master, country, EXOG_SUFFIXES, MAX_LAG)
            order = MODEL_SPEC[country]
            outlier_dates_used = []

            if AUTO_OUTLIER_DUMMIES and country in OUTLIER_COUNTRIES:
                final_res, selected_X, removed_terms, outlier_dates_used = refit_with_outlier_dummies(
                    y=y,
                    X=X,
                    order=order,
                    country=country,
                    alpha=SIGNIFICANCE_LEVEL,
                    remove_only_exog=REMOVE_ONLY_EXOG,
                    min_exog_terms=MIN_EXOG_TERMS,
                    threshold=OUTLIER_Z_THRESHOLD,
                    max_dummies=MAX_OUTLIER_DUMMIES
                )
            else:
                final_res, selected_X, removed_terms = prune_insignificant_exog(
                    y=y,
                    X=X,
                    order=order,
                    country=country,
                    alpha=SIGNIFICANCE_LEVEL,
                    remove_only_exog=REMOVE_ONLY_EXOG,
                    min_exog_terms=MIN_EXOG_TERMS
                )

            summary_rows.append(
                extract_model_summary(
                    country=country,
                    order=order,
                    res=final_res,
                    selected_X=selected_X,
                    removed_terms=removed_terms,
                    outlier_dates_used=outlier_dates_used
                )
            )

            coef_rows.extend(
                extract_coefficients(country, final_res)
            )

            # Export residual diagnostics for Thailand and Indonesia
            if country in ["Thailand", "Indonesia"]:
                save_residuals_to_excel(final_res, country)
                plot_residual_diagnostics(final_res, country)

        except Exception as e:
            print(f"[ERROR] {country}: {e}")
            summary_rows.append({
                "country": country,
                "model": None,
                "n_obs": None,
                "n_selected_exog": None,
                "selected_exog_terms": None,
                "removed_terms": None,
                "outlier_dates": None,
                "const": None,
                "aic": None,
                "bic": None,
                "hqic": None,
                "loglik": None,
                "resid_mean": None,
                "resid_std": None,
                "ljungbox_p_12": None,
                "ljungbox_p_24": None,
                "converged": None
            })

    summary_df = pd.DataFrame(summary_rows)
    coef_df = pd.DataFrame(coef_rows)

    summary_df.to_excel(OUT_PATH_SUMMARY, index=False)
    coef_df.to_excel(OUT_PATH_COEFS, index=False)

    print(f"\nSaved summary results to: {OUT_PATH_SUMMARY}")
    print(f"Saved coefficient results to: {OUT_PATH_COEFS}")

if __name__ == "__main__":
    main()