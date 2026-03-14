import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# ============================================================
# SUPPRESS WARNINGS
# ============================================================
warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================
MASTER_PATH = "DATA/Processed/armax_master_dataframe.xlsx"
OUT_PATH = "DATA/Processed/arma_baseline_results.xlsx"

COUNTRIES = ["Thailand", "Philippines", "Korea", "Indonesia"]

# Models based on lag test
MODEL_SPEC = {
    "Thailand": (1, 0, 0),      # AR(1)
    "Philippines": (1, 0, 1),   # ARMA(1,1)
    "Korea": (1, 0, 0),         # AR(1)
    "Indonesia": (1, 0, 0)      # AR(1)
}

# ============================================================
# LOAD MASTER DATAFRAME
# ============================================================
def load_master(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    return df

# ============================================================
# GET INFLATION SERIES
# ============================================================
def get_inflation(master: pd.DataFrame, country: str) -> pd.Series:
    col = f"{country}_infl"

    if col not in master.columns:
        raise ValueError(f"Column not found: {col}")

    s = pd.to_numeric(master[col], errors="coerce").dropna().copy()

    # Force monthly start frequency to avoid "No frequency information" warning
    s.index = pd.DatetimeIndex(s.index)
    s = s.asfreq("MS")

    # Drop any missing values that may appear after assigning frequency
    s = s.dropna()

    if len(s) == 0:
        raise ValueError(f"{country}: inflation series is empty after processing.")

    return s

# ============================================================
# ESTIMATE SINGLE ARMA MODEL
# ============================================================
def estimate_arma(series: pd.Series, order: tuple, country: str):
    p, d, q = order

    model = ARIMA(
        series,
        order=(p, d, q),
        trend="c"
    )

    # Using a stable optimizer setup
    result = model.fit(method_kwargs={"maxiter": 500})

    print("\n" + "=" * 35)
    print(f"{country} ARMA({p},{q})")
    print("=" * 35)
    print(result.summary())

    return result

# ============================================================
# MAIN
# ============================================================
def main():
    master = load_master(MASTER_PATH)
    results_table = []

    for country in COUNTRIES:
        infl = get_inflation(master, country)
        order = MODEL_SPEC[country]

        try:
            res = estimate_arma(infl, order, country)

            p, d, q = order
            results_table.append({
                "country": country,
                "model": f"ARMA({p},{q})",
                "n_obs": int(res.nobs),
                "const": res.params.get("const", None),
                "ar_L1": res.params.get("ar.L1", None),
                "ma_L1": res.params.get("ma.L1", None),
                "sigma2": res.params.get("sigma2", None),
                "loglik": res.llf,
                "aic": res.aic,
                "bic": res.bic
            })

        except Exception as e:
            print(f"\n[ERROR] {country}: {e}")
            p, d, q = order
            results_table.append({
                "country": country,
                "model": f"ARMA({p},{q})",
                "n_obs": None,
                "const": None,
                "ar_L1": None,
                "ma_L1": None,
                "sigma2": None,
                "loglik": None,
                "aic": None,
                "bic": None
            })

    results_df = pd.DataFrame(results_table)
    results_df.to_excel(OUT_PATH, index=False)

    print(f"\nResults saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()