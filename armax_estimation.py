import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

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

# Baseline ARMA structure kept fixed initially
MODEL_SPEC = {
    "Thailand": (1, 0, 0),      # AR(1)
    "Philippines": (1, 0, 1),   # ARMA(1,1)
    "Korea": (1, 0, 0),         # AR(1)
    "Indonesia": (1, 0, 0)      # AR(1)
}

# ----------------------------------------------------------------
# IMPORTANT:
# Put here the suffixes of your explanatory variables
# If your columns are like Thailand_m2, Thailand_exrate, etc.
# then use ["m2", "exrate", "short_rate", "long_rate", "reserves"]
# ----------------------------------------------------------------
EXOG_SUFFIXES = [
    "dlog_m2",
    "dlog_exr",
    "dlog_res",
    "d_ltr",
    "d_str"
]

MAX_LAG = 12
SIGNIFICANCE_LEVEL = 0.10   # often acceptable in macro time series
REMOVE_ONLY_EXOG = True     # keep ARMA structure fixed during pruning

# ============================================================
# LOAD MASTER DATAFRAME
# ============================================================
def load_master(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    return df

# ============================================================
# GET COUNTRY SERIES
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
# BUILD EXOG MATRIX WITH LAGS 1..12
# ============================================================
def build_exog_lags(master: pd.DataFrame, country: str, suffixes: list, max_lag: int) -> pd.DataFrame:
    exog_parts = []

    for suffix in suffixes:
        col = f"{country}_{suffix}"
        s = get_series(master, col)

        # If you also want contemporaneous X_t, uncomment next line
        # exog_parts.append(s.rename(f"{suffix}_L0"))

        for lag in range(1, max_lag + 1):
            exog_parts.append(s.shift(lag).rename(f"{suffix}_L{lag}"))

    exog = pd.concat(exog_parts, axis=1)
    return exog

# ============================================================
# PREPARE JOINT DATASET
# ============================================================
def build_country_dataset(master: pd.DataFrame, country: str, suffixes: list, max_lag: int):
    y = get_inflation(master, country)
    X = build_exog_lags(master, country, suffixes, max_lag)

    data = pd.concat([y.rename("infl"), X], axis=1).dropna()

    if data.empty:
        raise ValueError(f"{country}: dataset is empty after aligning inflation and lagged exogenous variables.")

    y_final = data["infl"]
    X_final = data.drop(columns=["infl"])

    return y_final, X_final

# ============================================================
# FIT ARMAX
# ============================================================
def fit_armax(y: pd.Series, X: pd.DataFrame, order: tuple, country: str):
    p, d, q = order

    model = ARIMA(
        y,
        exog=X,
        order=(p, d, q),
        trend="c"
    )

    result = model.fit(method_kwargs={"maxiter": 500})

    print("\n" + "=" * 60)
    print(f"{country} ARMAX{order} | n_exog = {X.shape[1]}")
    print("=" * 60)
    print(result.summary())

    return result

# ============================================================
# STEPWISE PRUNING OF INSIGNIFICANT EXOG LAGS
# ============================================================
def prune_insignificant_exog(y: pd.Series, X: pd.DataFrame, order: tuple, country: str,
                             alpha: float = 0.10, remove_only_exog: bool = True):
    current_X = X.copy()
    removed_terms = []

    while True:
        res = fit_armax(y, current_X, order, country)

        pvalues = res.pvalues.copy()

        # Terms we never remove here
        protected_terms = {"const", "sigma2"}

        if remove_only_exog:
            # protect AR and MA terms so baseline ARMA structure stays fixed
            for name in pvalues.index:
                if name.startswith("ar.") or name.startswith("ma."):
                    protected_terms.add(name)

        candidate_pvals = pvalues.drop(labels=[x for x in protected_terms if x in pvalues.index], errors="ignore")

        # Keep only terms that are currently in X
        candidate_pvals = candidate_pvals[candidate_pvals.index.isin(current_X.columns)]

        if candidate_pvals.empty:
            return res, current_X, removed_terms

        worst_term = candidate_pvals.idxmax()
        worst_p = candidate_pvals.max()

        if worst_p <= alpha:
            return res, current_X, removed_terms

        # remove worst exogenous lag
        current_X = current_X.drop(columns=[worst_term])
        removed_terms.append((worst_term, worst_p))

        print(f"[{country}] Removing insignificant term: {worst_term} (p-value = {worst_p:.4f})")

        if current_X.shape[1] == 0:
            # if everything is removed, fit pure ARMA with no exog
            res = fit_armax(y, pd.DataFrame(index=y.index), order, country)
            return res, pd.DataFrame(index=y.index), removed_terms

# ============================================================
# EXTRACT RESULTS
# ============================================================
def extract_model_summary(country: str, order: tuple, res, selected_X: pd.DataFrame, removed_terms: list):
    p, d, q = order

    return {
        "country": country,
        "model": f"ARMAX({p},{q})",
        "n_obs": int(res.nobs),
        "n_selected_exog": int(selected_X.shape[1]),
        "selected_exog_terms": ", ".join(selected_X.columns.tolist()) if selected_X.shape[1] > 0 else "",
        "removed_terms": ", ".join([f"{name} (p={pv:.3f})" for name, pv in removed_terms]) if removed_terms else "",
        "const": res.params.get("const", np.nan),
        "aic": res.aic,
        "bic": res.bic,
        "loglik": res.llf
    }

def extract_coefficients(country: str, res):
    rows = []

    for param in res.params.index:
        if param == "sigma2":
            continue  # leave sigma2 out of final interpretation table

        rows.append({
            "country": country,
            "term": param,
            "coef": res.params[param],
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
        print("\n" + "#" * 80)
        print(f"PROCESSING: {country}")
        print("#" * 80)

        try:
            y, X = build_country_dataset(master, country, EXOG_SUFFIXES, MAX_LAG)
            order = MODEL_SPEC[country]

            final_res, selected_X, removed_terms = prune_insignificant_exog(
                y=y,
                X=X,
                order=order,
                country=country,
                alpha=SIGNIFICANCE_LEVEL,
                remove_only_exog=REMOVE_ONLY_EXOG
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
                "loglik": None
            })

    summary_df = pd.DataFrame(summary_rows)
    coef_df = pd.DataFrame(coef_rows)

    summary_df.to_excel(OUT_PATH_SUMMARY, index=False)
    coef_df.to_excel(OUT_PATH_COEFS, index=False)

    print(f"\nSaved summary results to: {OUT_PATH_SUMMARY}")
    print(f"Saved coefficient results to: {OUT_PATH_COEFS}")

if __name__ == "__main__":
    main()