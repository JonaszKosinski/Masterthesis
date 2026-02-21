import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import adfuller 


Exchange_rate_df = pd.read_excel("DATA/Aggregated data/Exchange rates/Exchange rates.xlsx", sheet_name=[0,1])
Expected_inflation_df = pd.read_excel("DATA/Aggregated data/Expected inflation/Expected inflation aggregated.xlsx", sheet_name=0)
Inernational_reserves_df = pd.read_excel("DATA/Aggregated data/International reserves/International reserves aggregated.xlsx", sheet_name=1)
Long_term_interest_rate_df = pd.read_excel("DATA/Aggregated data/Long term interest/Long term interest rate aggregated.xlsx", sheet_name=0)
Short_term_interest_rate_df = pd.read_excel("DATA/Aggregated data/Short term interest/Short term 3 month interest rate aggregated.xlsx", sheet_name=0)

# ============================
# UNIT ROOT TEST: CPI
# ============================

# =========================
# SETTINGS (edit if needed)
# =========================
CPI_PATH = "DATA/Aggregated data/CPI/CPI aggregated.xlsx"
SHEET_NAME = 2       # you used sheet_name=2 before
HEADER_ROW = 1       # you used header=1 before
ALPHA = 0.05         # 5% significance level


# =========================
# HELPERS
# =========================
def load_cpi(path=CPI_PATH, sheet=SHEET_NAME, header=HEADER_ROW):
    df = pd.read_excel(path, sheet_name=sheet, header=header)

    # Clean column names
    df.columns = df.columns.map(lambda x: str(x).strip())

    # Convert Date column (handles "2015-M01" or "2015-01")
    if "Date" not in df.columns:
        raise ValueError("Could not find a 'Date' column in the CPI sheet.")

    df["Date"] = df["Date"].astype(str).str.replace("-M", "-", regex=False)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="coerce")

    # Drop rows with invalid dates, set index
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    return df


def detect_countries(df):
    # Keep columns that are not unnamed and are not empty
    cols = []
    for c in df.columns:
        c_str = str(c).strip()
        if c_str.lower().startswith("unnamed"):
            continue
        if c_str == "":
            continue
        cols.append(c_str)
    return cols


def adf_test(series, name, alpha=ALPHA):
    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) < 20:
        return {
            "Series": name,
            "n": len(series),
            "lags": np.nan,
            "ADF stat": np.nan,
            "p-value": np.nan,
            "Conclusion": "Too few observations"
        }

    res = adfuller(series, regression="c", autolag="AIC")
    stat, pval, used_lags, nobs = res[0], res[1], res[2], res[3]

    conclusion = "Stationary (reject unit root)" if pval < alpha else "Non-stationary (fail to reject)"

    return {
        "Series": name,
        "n": int(nobs),
        "lags": int(used_lags),
        "ADF stat": float(stat),
        "p-value": float(pval),
        "Conclusion": conclusion
    }


# =========================
# MAIN
# =========================
def main():
    cpi_df = load_cpi()

    countries = detect_countries(cpi_df)
    print(f"Countries detected: {countries}\n")

    results = []

    for country in countries:
        cpi = cpi_df[country]

        # 1) CPI level
        results.append(adf_test(cpi, f"{country} | CPI level"))

        # 2) Δlog(CPI)
        cpi_num = pd.to_numeric(cpi, errors="coerce")
        dlog_cpi = np.log(cpi_num).diff()  # diff(log(CPI))
        results.append(adf_test(dlog_cpi, f"{country} | Δlog(CPI)"))

    out = pd.DataFrame(results)

    # nicer formatting
    out["ADF stat"] = out["ADF stat"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    out["p-value"] = out["p-value"].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")

    print("--- ADF RESULTS (constant only; no trend) ---\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()


# =========================
# Exchange rates unit root tests
# =========================

# =========================
# settings (edit if needed)
# =========================
EXR_PATH = "DATA/Aggregated data/Exchange rates/Exchange rates.xlsx"
SHEET_NAME = 0      # change if needed (0 = first sheet) or use "Exchange rates"
HEADER_ROW = 0      # change if your headers are not on the first row
ALPHA = 0.05        # 5% level


# =========================
# HELPERS
# =========================
def find_date_column(df: pd.DataFrame) -> str:
    """
    Try to find the date column even if it's named weirdly.
    """
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    # 1) Exact matches (case-insensitive)
    for c in df.columns:
        if str(c).strip().lower() == "date":
            return c

    # 2) Contains "date"
    for c in df.columns:
        if "date" in str(c).strip().lower():
            return c

    # 3) Fallback: first column (often the date)
    return df.columns[0]


def make_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.map(lambda x: str(x).strip())

    date_col = find_date_column(df)

    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    return df


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Make sure the series is numeric (Excel sometimes imports as strings).
    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def adf_result(series: pd.Series, name: str, autolag: str = "AIC") -> dict:
    """
    ADF test with constant only (regression='c') and autolag=AIC by default.
    """
    series = clean_numeric_series(series)
    if len(series) < 20:
        return {"Series": name, "n": len(series), "lags": np.nan, "ADF": np.nan, "p": np.nan, "Conclusion": "Too few obs"}

    res = adfuller(series, regression="c", autolag=autolag)
    adf_stat, pval, used_lags, nobs = res[0], res[1], res[2], res[3]

    concl = "Stationary (reject unit root)" if pval < ALPHA else "Non-stationary (fail to reject)"
    return {"Series": name, "n": nobs, "lags": used_lags, "ADF": adf_stat, "p": pval, "Conclusion": concl}


# =========================
# MAIN
# =========================
print("\n--- ADF TESTS FOR EXCHANGE RATES (constant only; autolag=AIC) ---\n")

exr_df = pd.read_excel(EXR_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW)
exr_df = make_datetime_index(exr_df)

# drop empty columns like "Unnamed: ..."
exr_df = exr_df.loc[:, ~exr_df.columns.str.contains("^Unnamed", case=False, na=False)]

countries = list(exr_df.columns)
print(f"Series detected: {countries}\n")

results = []

for c in countries:
    x = clean_numeric_series(exr_df[c])

    # 1) level
    results.append(adf_result(x, f"{c} | EXR level"))

    # 2) Δlog(EXR)
    # (replace <=0 with NaN so log works)
    x_pos = x.where(x > 0)
    dlog = np.log(x_pos).diff().dropna()
    results.append(adf_result(dlog, f"{c} | Δlog(EXR)"))

out = pd.DataFrame(results)

# Pretty print
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
print(out.to_string(index=False, formatters={
    "ADF": lambda v: f"{v:.4f}" if pd.notna(v) else "NA",
    "p":   lambda v: f"{v:.6f}" if pd.notna(v) else "NA",
}))
print()

# =========================
# International reserves unit root tests
# =========================

RES_PATH = "DATA/Aggregated data/International reserves/International reserves aggregated.xlsx"
SHEET_NAME = 1      # your base index sheet
HEADER_ROW = 0

print("\n--- ADF TESTS FOR INTERNATIONAL RESERVES (constant only; autolag=AIC) ---\n")

res_df = pd.read_excel(RES_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW)
res_df = make_datetime_index(res_df)

# drop unnamed columns
res_df = res_df.loc[:, ~res_df.columns.str.contains("^Unnamed", case=False, na=False)]

countries = list(res_df.columns)
print(f"Series detected: {countries}\n")

results = []

for c in countries:
    x = clean_numeric_series(res_df[c])

    # 1) level
    results.append(adf_result(x, f"{c} | Reserves level"))

    # 2) Δlog(reserves)
    x_pos = x.where(x > 0)
    dlog = np.log(x_pos).diff().dropna()
    results.append(adf_result(dlog, f"{c} | Δlog(Reserves)"))

out = pd.DataFrame(results)

print(out.to_string(index=False, formatters={
    "ADF": lambda v: f"{v:.4f}" if pd.notna(v) else "NA",
    "p":   lambda v: f"{v:.6f}" if pd.notna(v) else "NA",
}))
print()