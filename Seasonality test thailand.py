import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

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

# test 1, check for seasonality using ACF of inflation (Δlog(CPI)) and checking plots for significant spikes at seasonal lags (e.g., 12, 24, 36 months)
country1 = "Thailand" 
log_cpi = CPI_df[country1].dropna()
infl = log_cpi.diff(1).dropna()

plot_acf(infl, lags=36)
plt.title(f"ACF of monthly inflation (Δlog(CPI)) - {country1}")
plt.show()

# test for seasonality using month dummies, test 2

month_dummies = pd.get_dummies(infl.index.month, drop_first=True)
month_dummies.index = infl.index  # align index

# rename columns to valid names for statsmodels/patsy
month_dummies.columns = [f"m{int(c):02d}" for c in month_dummies.columns]  # m02 ... m12

X = sm.add_constant(month_dummies).astype(float)
y = infl.astype(float)

ols = sm.OLS(y, X).fit()

# joint test: all month dummies = 0
hypothesis = " = 0, ".join(month_dummies.columns) + " = 0"
f_test = ols.f_test(hypothesis)

print("Seasonality (month dummies) p-value:", float(f_test.pvalue))
print("F-stat:", float(f_test.fvalue))
print("p-value:", float(f_test.pvalue))
