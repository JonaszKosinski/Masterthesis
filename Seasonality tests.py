import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import numpy as np

# ============================================================
# SETTINGS
# ============================================================
CPI_PATH = "DATA/Aggregated data/CPI/CPI aggregated.xlsx"
EXR_PATH = "DATA/Aggregated data/Exchange rates/Exchange rates.xlsx"
RES_PATH = "DATA/Aggregated data/International reserves/International reserves aggregated.xlsx"
LTR_PATH = "DATA/Aggregated data/Long term interest/Long term interest rate aggregated.xlsx"


COUNTRIES = ["Thailand", "Philippines", "Korea", "Indonesia"]


# ============================================================
# HELPERS (robust loading + robust tests)
# ============================================================
def find_date_col(df: pd.DataFrame) -> str:
    cols = [str(c).strip() for c in df.columns]
    for c in cols:
        if c.lower() in ["date", "dates", "time", "period"]:
            return c
    return cols[0]

def load_monthly_df(path: str, sheet_name=0, header=0, date_format=None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    df = df.copy()
    df.columns = df.columns.map(lambda x: str(x).strip())

    date_col = find_date_col(df)

    # Special handling for IMF-like "2015-M01"
    if date_format == "%Y-%m":
        df[date_col] = df[date_col].astype(str).str.replace("-M", "-", regex=False)
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m", errors="coerce")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, na=False)]
    return df

def pick_best_sheet(path: str, target_cols) -> tuple[str, pd.DataFrame, int]:
    sheets = pd.read_excel(path, sheet_name=None)
    best_name, best_df, best_hits = None, None, -1
    for name, df in sheets.items():
        cols = [str(c).strip() for c in df.columns]
        hits = sum(col in cols for col in target_cols)
        if hits > best_hits:
            best_name, best_df, best_hits = name, df, hits
    return best_name, best_df, best_hits

def seasonality_f_test(series: pd.Series, title: str, show_plot=True, lags=36) -> float:
    """
    ACF + monthly dummy joint F-test on a provided series.
    Robust to short/empty series and excessive lags.
    Returns p-value, or np.nan if not testable.
    """
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)

    # guard: empty/too short
    if n < 15:
        print(f"[SKIP] {title}: too few observations after cleaning (n={n})")
        return np.nan

    # cap lags to feasible max
    max_lags = min(lags, n - 1)

    if show_plot:
        plot_acf(s, lags=max_lags)
        plt.title(f"{title} (n={n}, lags={max_lags})")
        plt.show()

    month_dummies = pd.get_dummies(s.index.month, drop_first=True)
    month_dummies.index = s.index

    # guard: if series covers only one month category somehow
    if month_dummies.shape[1] == 0:
        print(f"[SKIP] {title}: not enough month variation for dummy regression (n={n})")
        return np.nan

    month_dummies.columns = [f"m{c}" for c in month_dummies.columns]

    X = sm.add_constant(month_dummies).astype(float)
    y = s.astype(float)

    ols = sm.OLS(y, X).fit()
    terms = " = 0, ".join(month_dummies.columns) + " = 0"
    return float(ols.f_test(terms).pvalue)

def trimmed_country_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Trim to the country’s actual sample (drop leading/trailing NaNs), keep internal missing handled later."""
    s = pd.to_numeric(df[col], errors="coerce")
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None or last is None:
        return pd.Series(dtype=float)
    return s.loc[first:last]

# ============================================================
# LOAD DATA
# ============================================================
# CPI: already log(CPI) in your file; monthly formatted like 2015-M01
CPI_df = load_monthly_df(CPI_PATH, sheet_name=2, header=1, date_format="%Y-%m")

# EXR: levels sheet, Date is normal (2015-01-01)
EXR_df = load_monthly_df(EXR_PATH, sheet_name=0, header=0, date_format=None)

# RESERVES: choose correct sheet automatically (the one with most country columns)
res_sheet_name, res_raw_df, res_hits = pick_best_sheet(RES_PATH, COUNTRIES)
print(f"[RESERVES] Using sheet: {res_sheet_name} (matched {res_hits}/{len(COUNTRIES)} country columns)")
# Now parse dates robustly on that chosen sheet
res_raw_df = res_raw_df.copy()
res_raw_df.columns = res_raw_df.columns.map(lambda x: str(x).strip())
res_date_col = find_date_col(res_raw_df)
res_raw_df[res_date_col] = pd.to_datetime(res_raw_df[res_date_col], errors="coerce")
RES_df = res_raw_df.dropna(subset=[res_date_col]).set_index(res_date_col).sort_index()
RES_df = RES_df.loc[:, ~RES_df.columns.str.contains("^Unnamed", case=False, na=False)]
LTR_df = load_monthly_df(LTR_PATH, sheet_name=0, header=0, date_format=None)


# ============================================================
# RUN TESTS
# ============================================================
print("\n=== CPI seasonality on inflation: Δlog(CPI) (your CPI is already logged) ===")
for c in COUNTRIES:
    if c not in CPI_df.columns:
        print(f"[SKIP] CPI {c}: column not found")
        continue
    infl = trimmed_country_series(CPI_df, c).diff().dropna()  # Δlog(CPI) since CPI already log
    p = seasonality_f_test(infl, f"ACF of inflation Δlog(CPI) - {c}", show_plot=True, lags=36)
    print(f"{c} - CPI inflation seasonality p-value: {p}")

print("\n=== EXR seasonality on depreciation: Δlog(EXR) ===")
for c in COUNTRIES:
    if c not in EXR_df.columns:
        print(f"[SKIP] EXR {c}: column not found")
        continue
    exr_level = trimmed_country_series(EXR_df, c)
    dlog_exr = np.log(exr_level.where(exr_level > 0)).diff().dropna()
    p = seasonality_f_test(dlog_exr, f"ACF of Δlog(EXR) - {c}", show_plot=True, lags=36)
    print(f"{c} - EXR Δlog seasonality p-value: {p}")

print("\n=== INTERNATIONAL RESERVES seasonality on growth: Δlog(Reserves) ===")
for c in COUNTRIES:
    if c not in RES_df.columns:
        print(f"[SKIP] RES {c}: column not found in sheet '{res_sheet_name}'. Available: {list(RES_df.columns)}")
        continue

    res_level = trimmed_country_series(RES_df, c).dropna()
    dlog_res = np.log(res_level.where(res_level > 0)).diff().dropna()

    print(f"{c}: level_n={res_level.shape[0]}, dlog_n={dlog_res.shape[0]}")

    p = seasonality_f_test(dlog_res, f"ACF of Δlog(Reserves) - {c}", show_plot=True, lags=36)
    print(f"{c} - Reserves Δlog seasonality p-value: {p}")

    print("\n=== LONG-TERM INTEREST RATE seasonality on changes: Δr ===")

for c in COUNTRIES:
    if c not in LTR_df.columns:
        print(f"[SKIP] LTR {c}: column not found")
        continue

    r_level = trimmed_country_series(LTR_df, c)
    dr = r_level.diff().dropna()   # ← IMPORTANT: difference, NOT log

    print(f"{c}: level_n={r_level.shape[0]}, diff_n={dr.shape[0]}")

    p = seasonality_f_test(
        dr,
        f"ACF of ΔLong-term rate - {c}",
        show_plot=True,
        lags=36
    )

    print(f"{c} - Long-term rate Δ seasonality p-value: {p}")