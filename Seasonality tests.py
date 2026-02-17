import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import numpy as np

CPI_df = pd.read_excel("DATA/Aggregated data/CPI/CPI aggregated.xlsx", sheet_name=2, header=1)
Exchange_rate_df = pd.read_excel("DATA/Aggregated data/Exchange rates/Exchange rates.xlsx", sheet_name=[0,1])
Expected_inflation_df = pd.read_excel("DATA/Aggregated data/Expected inflation/Expected inflation aggregated.xlsx", sheet_name=0)
Inernational_reserves_df = pd.read_excel("DATA/Aggregated data/International reserves/International reserves aggregated.xlsx", sheet_name=1)
Long_term_interest_rate_df = pd.read_excel("DATA/Aggregated data/Long term interest/Long term interest rate aggregated.xlsx", sheet_name=0)
Short_term_interest_rate_df = pd.read_excel("DATA/Aggregated data/Short term interest/Short term 3 month interest rate aggregated.xlsx", sheet_name=0)

CPI_df.columns = CPI_df.columns.map(lambda x: str(x).strip())

CPI_df["Date"] = CPI_df["Date"].astype(str).str.replace("-M", "-", regex=False)
CPI_df["Date"] = pd.to_datetime(CPI_df["Date"], format="%Y-%m")
CPI_df = CPI_df.set_index("Date").sort_index()

def seasonality_test(country, show_plot=True):

    log_cpi = CPI_df[country].dropna()
    infl = log_cpi.diff(1).dropna()

    # --- Test 1: ACF plot ---
    if show_plot:
        plot_acf(infl, lags=36)
        plt.title(f"ACF of monthly inflation (Δlog(CPI)) - {country}")
        plt.show()

    # --- Test 2: Month dummies joint F-test ---
    month_dummies = pd.get_dummies(infl.index.month, drop_first=True)
    month_dummies.index = infl.index
    month_dummies.columns = [f"m{c}" for c in month_dummies.columns]  # names like m2..m12

    X = sm.add_constant(month_dummies).astype(float)
    y = infl.astype(float)

    ols = sm.OLS(y, X).fit()

    # Joint test: all seasonal dummies = 0
    terms = " = 0, ".join(month_dummies.columns) + " = 0"
    f_test = ols.f_test(terms)

    pval = float(f_test.pvalue)
    print(f"{country} - Seasonality (month dummies) p-value: {pval:.6f}")

    return pval

countries = ["Thailand", "Philippines", "Korea", "Indonesia"]

for c in countries:
    seasonality_test(c, show_plot=True)