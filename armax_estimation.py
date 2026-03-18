import warnings
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

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
# BASELINE ARMA STRUCTURE
# These should come from your inflation ACF/PACF inspection,
# and/or low-order candidate testing.
# ----------------------------------------------------------------
MODEL_SPEC = {
    "Thailand": (1, 0, 1),      # ARIMA(p,d,q) with d=0 after transformation
    "Philippines": (1, 0, 1),
    "Korea": (1, 0, 0),
    "Indonesia": (1, 0, 1)
}

# ----------------------------------------------------------------
# Exogenous variable suffixes
# If columns are Thailand_dlog_m2, Thailand_dlog_exr, ...
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

# Keep ARMA structure fixed while pruning exogenous regressors
REMOVE_ONLY_EXOG = True

# Optional minimum number of exogenous terms to keep
MIN_EXOG_TERMS = 0

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

    result = model.fit(method_kwargs={"maxiter": 500})

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

        # Terms we protect from removal
        protected_terms = {"const", "sigma2"}

        if remove_only_exog:
            for name in pvalues.index:
                if name.startswith("ar.") or name.startswith("ma."):
                    protected_terms.add(name)

        candidate_pvals = pvalues.drop(
            labels=[term for term in protected_terms if term in pvalues.index],
            errors="ignore"
        )

        # Keep only terms that are in the current exog matrix
        if current_X.shape[1] > 0:
            candidate_pvals = candidate_pvals[candidate_pvals.index.isin(current_X.columns)]
        else:
            candidate_pvals = pd.Series(dtype=float)

        # No removable exogenous terms left
        if candidate_pvals.empty:
            return res, current_X, removed_terms

        worst_term = candidate_pvals.idxmax()
        worst_p = candidate_pvals.max()

        # Stop if all remaining exogenous terms are significant enough
        if worst_p <= alpha:
            return res, current_X, removed_terms

        # Stop if removing one more term would violate minimum exog count
        if current_X.shape[1] <= min_exog_terms:
            return res, current_X, removed_terms

        current_X = current_X.drop(columns=[worst_term])
        removed_terms.append((worst_term, worst_p))

        print(f"[{country}] Removing: {worst_term} (p-value = {worst_p:.4f})")

        # If everything gets removed, fit pure ARMA
        if current_X.shape[1] == 0:
            print(f"[{country}] All exogenous terms removed. Refitting pure ARIMA({order[0]},{order[1]},{order[2]}).")
            res = fit_armax(y, None, order, country, verbose=True)
            return res, pd.DataFrame(index=y.index), removed_terms

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
def extract_model_summary(country: str, order: tuple, res, selected_X: pd.DataFrame, removed_terms: list):
    p, d, q = order
    diag = compute_diagnostics(res)

    return {
        "country": country,
        "model": f"ARIMA({p},{d},{q}) + exog",
        "n_obs": int(res.nobs),
        "n_selected_exog": int(selected_X.shape[1]),
        "selected_exog_terms": ", ".join(selected_X.columns.tolist()) if selected_X.shape[1] > 0 else "",
        "removed_terms": ", ".join([f"{name} (p={pv:.3f})" for name, pv in removed_terms]) if removed_terms else "",
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
                extract_model_summary(country, order, final_res, selected_X, removed_terms)
            )

            coef_rows.extend(
                extract_coefficients(country, final_res)
            )

        except Exception as e:
            print(f"[ERROR] {country}: {e}")
            summary_rows.append({
                "country": country,
                "model": None,
                "n_obs": None,
                "n_selected_exog": None,
                "selected_exog_terms": None,
                "removed_terms": None,
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