import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

# ============================================================
# SETTINGS
# ============================================================
COUNTRIES = ["Thailand", "Philippines", "Korea", "Indonesia"]

MASTER_PATH = "DATA/Processed/armax_master_dataframe.xlsx"
OUT_PATH = "DATA/Processed/arma_main_lag_test_results.xlsx"

MAX_LAG = 12
ALPHA = 0.05   # 95% confidence interval


# ============================================================
# HELPERS
# ============================================================
def load_master(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    return df


def get_inflation_series(master: pd.DataFrame, country: str) -> pd.Series:
    col = f"{country}_infl"
    if col not in master.columns:
        raise ValueError(f"Column not found: {col}")

    s = pd.to_numeric(master[col], errors="coerce").dropna()

    if len(s) == 0:
        raise ValueError(f"{country}: inflation series is empty after dropna().")

    return s


def significant_from_ci(confint: np.ndarray) -> list:
    """
    A lag is significant if its confidence interval does not include zero.
    """
    sig = []
    for i in range(len(confint)):
        low, high = confint[i]
        sig.append((low > 0) or (high < 0))
    return sig


def get_significant_lags(table: pd.DataFrame, value_col: str) -> list:
    """
    Return significant lags excluding lag 0.
    """
    return table.loc[
        (table["lag"] > 0) & (table["significant"] == True),
        "lag"
    ].tolist()


def get_cutoff_lag(sig_lags: list) -> float:
    """
    Standard identification logic:
    take the largest consecutive significant lag starting from 1.
    
    Examples:
    [1,2,3] -> 3
    [1,2,4] -> 2
    [2,3]   -> NaN  (because it does not start at lag 1)
    []      -> NaN
    """
    if not sig_lags:
        return np.nan

    sig_lags = sorted(sig_lags)

    if sig_lags[0] != 1:
        return np.nan

    cutoff = 1
    for lag in sig_lags[1:]:
        if lag == cutoff + 1:
            cutoff = lag
        else:
            break

    return cutoff


def run_main_lag_test(series: pd.Series, max_lag: int = 12, alpha: float = 0.05):
    """
    Main ARMA lag identification:
    - PACF -> suggested AR order p
    - ACF  -> suggested MA order q
    """
    acf_vals, acf_conf = acf(series, nlags=max_lag, alpha=alpha, fft=False)
    pacf_vals, pacf_conf = pacf(series, nlags=max_lag, alpha=alpha, method="ywm")

    acf_sig = significant_from_ci(acf_conf)
    pacf_sig = significant_from_ci(pacf_conf)

    acf_table = pd.DataFrame({
        "lag": range(len(acf_vals)),
        "acf": acf_vals,
        "ci_lower": acf_conf[:, 0],
        "ci_upper": acf_conf[:, 1],
        "significant": acf_sig
    })

    pacf_table = pd.DataFrame({
        "lag": range(len(pacf_vals)),
        "pacf": pacf_vals,
        "ci_lower": pacf_conf[:, 0],
        "ci_upper": pacf_conf[:, 1],
        "significant": pacf_sig
    })

    acf_sig_lags = get_significant_lags(acf_table, "acf")
    pacf_sig_lags = get_significant_lags(pacf_table, "pacf")

    suggested_q = get_cutoff_lag(acf_sig_lags)    # MA(q)
    suggested_p = get_cutoff_lag(pacf_sig_lags)   # AR(p)

    return acf_table, pacf_table, acf_sig_lags, pacf_sig_lags, suggested_p, suggested_q


def make_summary(country: str, series: pd.Series,
                 acf_sig_lags: list,
                 pacf_sig_lags: list,
                 suggested_p,
                 suggested_q) -> pd.DataFrame:
    summary = pd.DataFrame([{
        "country": country,
        "n_obs": len(series),
        "start": series.index.min().date(),
        "end": series.index.max().date(),
        "significant_acf_lags": ", ".join(map(str, acf_sig_lags)) if acf_sig_lags else "",
        "significant_pacf_lags": ", ".join(map(str, pacf_sig_lags)) if pacf_sig_lags else "",
        "suggested_AR_p_from_PACF": suggested_p,
        "suggested_MA_q_from_ACF": suggested_q
    }])
    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    master = load_master(MASTER_PATH)
    all_summaries = []

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        for country in COUNTRIES:
            print(f"\n==============================")
            print(f"Running main lag test for {country}")
            print(f"==============================")

            try:
                infl = get_inflation_series(master, country)

                acf_table, pacf_table, acf_sig_lags, pacf_sig_lags, suggested_p, suggested_q = run_main_lag_test(
                    infl,
                    max_lag=MAX_LAG,
                    alpha=ALPHA
                )

                summary = make_summary(
                    country,
                    infl,
                    acf_sig_lags,
                    pacf_sig_lags,
                    suggested_p,
                    suggested_q
                )

                all_summaries.append(summary)

                sheet_prefix = country[:10]
                summary.to_excel(writer, sheet_name=f"{sheet_prefix}_sum", index=False)
                acf_table.to_excel(writer, sheet_name=f"{sheet_prefix}_acf", index=False)
                pacf_table.to_excel(writer, sheet_name=f"{sheet_prefix}_pacf", index=False)

                print(f"{country}: done")
                print(f"  Significant ACF lags  = {acf_sig_lags if acf_sig_lags else 'none'}")
                print(f"  Significant PACF lags = {pacf_sig_lags if pacf_sig_lags else 'none'}")
                print(f"  Suggested AR order p  = {suggested_p}")
                print(f"  Suggested MA order q  = {suggested_q}")

            except Exception as e:
                print(f"[ERROR] {country}: {e}")

        if all_summaries:
            final_summary = pd.concat(all_summaries, ignore_index=True)
            final_summary.to_excel(writer, sheet_name="summary_all", index=False)

    print(f"\nSaved main lag test results to: {OUT_PATH}")


if __name__ == "__main__":
    main()