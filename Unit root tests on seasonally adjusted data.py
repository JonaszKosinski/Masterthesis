import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# ============================================================
# SETTINGS
# ============================================================
ALPHA = 0.05
COUNTRIES = ["Thailand", "Philippines", "Korea", "Indonesia"]

# --- Seasonally adjusted (Processed) files ---
CPI_SA_PATH = "DATA/Processed/CPI_seasonally_adjusted.xlsx"
RES_SA_PATH = "DATA/Processed/Reserves_seasonally_adjusted.xlsx"
LTR_SA_PATH = "DATA/Processed/Long_term_rate_seasonally_adjusted.xlsx"
STR_SA_PATH = "DATA/Processed/Short_term_rate_seasonally_adjusted.xlsx"
M2_SA_PATH  = "DATA/Processed/M2_seasonally_adjusted.xlsx"

# Exchange rates were NOT seasonally adjusted (per your seasonality results)
EXR_PATH = "DATA/Aggregated data/Exchange rates/Exchange rates.xlsx"
EXR_SHEET = 0
EXR_HEADER = 0


# ============================================================
# HELPERS
# ============================================================
def find_date_column(df: pd.DataFrame) -> str:
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    # exact match
    for c in df.columns:
        if str(c).strip().lower() == "date":
            return c

    # contains date/time/period
    for c in df.columns:
        cl = str(c).strip().lower()
        if "date" in cl or cl in ["time", "period", "dates"]:
            return c

    # fallback
    return df.columns[0]


def make_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.map(lambda x: str(x).strip())
    date_col = find_date_column(df)

    # robust parsing
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # drop Excel junk
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, na=False)]
    return df


def clean_numeric_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def adf_result(series: pd.Series, name: str, autolag: str = "AIC") -> dict:
    s = clean_numeric_series(series)
    if len(s) < 20:
        return {"Series": name, "n": len(s), "lags": np.nan, "ADF": np.nan, "p": np.nan, "Conclusion": "Too few obs"}

    res = adfuller(s, regression="c", autolag=autolag)
    adf_stat, pval, used_lags, nobs = res[0], res[1], res[2], res[3]
    concl = "Stationary (reject unit root)" if pval < ALPHA else "Non-stationary (fail to reject)"
    return {"Series": name, "n": int(nobs), "lags": int(used_lags), "ADF": float(adf_stat), "p": float(pval), "Conclusion": concl}


def load_processed_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=0)
    df = make_datetime_index(df)
    return df


def trimmed_country_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Trim leading/trailing NaNs for that country, keep internal missing for later dropna()."""
    if col not in df.columns:
        return pd.Series(dtype=float)

    s = pd.to_numeric(df[col], errors="coerce")
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None or last is None:
        return pd.Series(dtype=float)
    return s.loc[first:last]


def pretty_print(df: pd.DataFrame):
    print(df.to_string(index=False, formatters={
        "ADF": lambda v: f"{v:.4f}" if pd.notna(v) else "NA",
        "p":   lambda v: f"{v:.6f}" if pd.notna(v) else "NA",
    }))
    print()


# ============================================================
# MAIN: ADF TESTS ON SEASONALLY ADJUSTED DATA
# ============================================================
def main():
    results = []

    # ----------------------------
    # CPI (SA) - file already in log(CPI) units (same as before)
    # test: level and Δlog(CPI) (which is diff because already logged)
    # ----------------------------
    print("\n--- ADF TESTS: CPI (SEASONALLY ADJUSTED) ---\n")
    cpi_sa = load_processed_xlsx(CPI_SA_PATH)

    for c in COUNTRIES:
        if c not in cpi_sa.columns:
            print(f"[SKIP] CPI SA {c}: column not found")
            continue

        x = trimmed_country_series(cpi_sa, c)
        results.append(adf_result(x, f"{c} | CPI_SA level (log CPI)"))

        dx = x.diff()  # Δlog(CPI) because CPI already logged
        results.append(adf_result(dx, f"{c} | Δlog(CPI_SA)"))

    pretty_print(pd.DataFrame(results))
    results = []

    # ----------------------------
    # EXCHANGE RATES (NOT SA)
    # test: level and Δlog(EXR)
    # ----------------------------
    print("\n--- ADF TESTS: EXCHANGE RATES (NOT SEASONALLY ADJUSTED) ---\n")
    exr_df = pd.read_excel(EXR_PATH, sheet_name=EXR_SHEET, header=EXR_HEADER)
    exr_df = make_datetime_index(exr_df)

    for c in COUNTRIES:
        if c not in exr_df.columns:
            print(f"[SKIP] EXR {c}: column not found")
            continue

        x = trimmed_country_series(exr_df, c)
        x = clean_numeric_series(x)

        results.append(adf_result(x, f"{c} | EXR level"))

        x_pos = x.where(x > 0)
        dlog = np.log(x_pos).diff()
        results.append(adf_result(dlog, f"{c} | Δlog(EXR)"))

    pretty_print(pd.DataFrame(results))
    results = []

    # ----------------------------
    # RESERVES (SA)
    # test: level and Δlog(Reserves)
    # ----------------------------
    print("\n--- ADF TESTS: INTERNATIONAL RESERVES (SEASONALLY ADJUSTED) ---\n")
    res_sa = load_processed_xlsx(RES_SA_PATH)

    for c in COUNTRIES:
        if c not in res_sa.columns:
            print(f"[SKIP] RES SA {c}: column not found")
            continue

        x = trimmed_country_series(res_sa, c)
        results.append(adf_result(x, f"{c} | Reserves_SA level"))

        x_pos = x.where(x > 0)
        dlog = np.log(x_pos).diff()
        results.append(adf_result(dlog, f"{c} | Δlog(Reserves_SA)"))

    pretty_print(pd.DataFrame(results))
    results = []

    # ----------------------------
    # LONG-TERM RATE (SA)
    # test: level and Δr
    # ----------------------------
    print("\n--- ADF TESTS: LONG-TERM INTEREST RATE (SEASONALLY ADJUSTED) ---\n")
    ltr_sa = load_processed_xlsx(LTR_SA_PATH)

    for c in COUNTRIES:
        if c not in ltr_sa.columns:
            print(f"[SKIP] LTR SA {c}: column not found")
            continue

        x = trimmed_country_series(ltr_sa, c)
        results.append(adf_result(x, f"{c} | LT_rate_SA level"))

        dx = x.diff()
        results.append(adf_result(dx, f"{c} | ΔLT_rate_SA"))

    pretty_print(pd.DataFrame(results))
    results = []

    # ----------------------------
    # SHORT-TERM RATE (SA)
    # test: level and Δr
    # ----------------------------
    print("\n--- ADF TESTS: SHORT-TERM (3M) INTEREST RATE (SEASONALLY ADJUSTED) ---\n")
    str_sa = load_processed_xlsx(STR_SA_PATH)

    for c in COUNTRIES:
        if c not in str_sa.columns:
            print(f"[SKIP] STR SA {c}: column not found")
            continue

        x = trimmed_country_series(str_sa, c)
        results.append(adf_result(x, f"{c} | ST_rate_SA level"))

        dx = x.diff()
        results.append(adf_result(dx, f"{c} | ΔST_rate_SA"))

    pretty_print(pd.DataFrame(results))
    results = []

    # ----------------------------
    # M2 (SA)
    # test: level and Δlog(M2)
    # ----------------------------
    print("\n--- ADF TESTS: M2 (SEASONALLY ADJUSTED) ---\n")
    m2_sa = load_processed_xlsx(M2_SA_PATH)

    for c in COUNTRIES:
        if c not in m2_sa.columns:
            print(f"[SKIP] M2 SA {c}: column not found")
            continue

        x = trimmed_country_series(m2_sa, c)
        results.append(adf_result(x, f"{c} | M2_SA level"))

        x_pos = x.where(x > 0)
        dlog = np.log(x_pos).diff()
        results.append(adf_result(dlog, f"{c} | Δlog(M2_SA)"))

    pretty_print(pd.DataFrame(results))


if __name__ == "__main__":
    main()