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

