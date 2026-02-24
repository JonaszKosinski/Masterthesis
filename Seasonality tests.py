import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import numpy as np

# ========= LOAD CPI (already log(CPI) in your file) =========
CPI_df = pd.read_excel("DATA/Aggregated data/CPI/CPI aggregated.xlsx", sheet_name=2, header=1)
CPI_df.columns = CPI_df.columns.map(lambda x: str(x).strip())
CPI_df["Date"] = CPI_df["Date"].astype(str).str.replace("-M", "-", regex=False)
CPI_df["Date"] = pd.to_datetime(CPI_df["Date"], format="%Y-%m")
CPI_df = CPI_df.set_index("Date").sort_index()

# ========= LOAD EXCHANGE RATES (LEVELS sheet) =========
EXR_PATH = "DATA/Aggregated data/Exchange rates/Exchange rates.xlsx"
EXR_df = pd.read_excel(EXR_PATH, sheet_name=0, header=0)  # levels sheet
EXR_df.columns = EXR_df.columns.map(lambda x: str(x).strip())
# robust date parsing (your file shows 2015-01-01 already)
date_col = "Date" if "Date" in EXR_df.columns else EXR_df.columns[0]
EXR_df[date_col] = pd.to_datetime(EXR_df[date_col], errors="coerce")
EXR_df = EXR_df.dropna(subset=[date_col]).set_index(date_col).sort_index()

def seasonality_f_test(series: pd.Series, title: str, show_plot=True, lags=36):
    """ACF + monthly dummy joint F-test on the provided series."""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if show_plot:
        plot_acf(s, lags=lags)
        plt.title(title)
        plt.show()

    month_dummies = pd.get_dummies(s.index.month, drop_first=True)
    month_dummies.index = s.index
    month_dummies.columns = [f"m{c}" for c in month_dummies.columns]

    X = sm.add_constant(month_dummies).astype(float)
    y = s.astype(float)

    ols = sm.OLS(y, X).fit()

    terms = " = 0, ".join(month_dummies.columns) + " = 0"
    pval = float(ols.f_test(terms).pvalue)
    return pval

countries = ["Thailand", "Philippines", "Korea", "Indonesia"]

print("\n=== CPI seasonality on inflation: Δlog(CPI) (your CPI is already logged) ===")
for c in countries:
    infl = pd.to_numeric(CPI_df[c], errors="coerce").diff().dropna()  # Δlog(CPI)
    p = seasonality_f_test(infl, f"ACF of inflation Δlog(CPI) - {c}", show_plot=True)
    print(f"{c} - CPI inflation seasonality p-value: {p:.6f}")

print("\n=== EXR seasonality on depreciation: Δlog(EXR) ===")
for c in countries:
    exr_level = pd.to_numeric(EXR_df[c], errors="coerce")
    dlog_exr = np.log(exr_level.where(exr_level > 0)).diff().dropna()  # Δlog(EXR)
    p = seasonality_f_test(dlog_exr, f"ACF of Δlog(EXR) - {c}", show_plot=True)
    print(f"{c} - EXR Δlog seasonality p-value: {p:.6f}")