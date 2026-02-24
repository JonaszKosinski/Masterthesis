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
STR_PATH = "DATA/Aggregated data/Short term interest/Short term 3 month interest rate aggregated.xlsx"
M2_PATH  = "DATA/Aggregated data/M2/M2 aggregated.xlsx"   # <-- adjust if your folder/file name differs

COUNTRIES = ["Thailand", "Philippines", "Korea", "Indonesia"]

# ============================================================
# HELPERS (loading + cleaning + testing)
# ============================================================
def find_date_col(df: pd.DataFrame) -> str:
    cols = [str(c).strip() for c in df.columns]
    for c in cols:
        if c.lower() in ["date", "dates", "time", "period"]:
            return c
    return cols[0]

def load_monthly_df(path: str, sheet_name=0, header=0, date_format=None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, header=header).copy()
    df.columns = df.columns.map(lambda x: str(x).strip())
    date_col = find_date_col(df)

    # IMF-like "2015-M01"
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

def trimmed_country_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None or last is None:
        return pd.Series(dtype=float)
    return s.loc[first:last]

def seasonality_f_test(series: pd.Series, title: str, show_plot=True, lags=36) -> float:
    """
    ACF + monthly dummy joint F-test.
    Returns p-value, or np.nan if not testable.
    """
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)

    if n < 15:
        print(f"[SKIP] {title}: too few observations after cleaning (n={n})")
        return np.nan

    max_lags = min(lags, n - 1)
    if show_plot:
        plot_acf(s, lags=max_lags)
        plt.title(f"{title} (n={n}, lags={max_lags})")
        plt.show()

    month_dummies = pd.get_dummies(s.index.month, drop_first=True)
    month_dummies.index = s.index

    if month_dummies.shape[1] == 0:
        print(f"[SKIP] {title}: not enough month variation for dummy regression (n={n})")
        return np.nan

    month_dummies.columns = [f"m{c}" for c in month_dummies.columns]
    X = sm.add_constant(month_dummies).astype(float)
    y = s.astype(float)

    ols = sm.OLS(y, X).fit()
    terms = " = 0, ".join(month_dummies.columns) + " = 0"
    return float(ols.f_test(terms).pvalue)

def best_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the candidate column with the most valid numeric obs.
    """
    best_col, best_n = None, -1
    for col in candidates:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            n = int(s.notna().sum())
            if n > best_n:
                best_col, best_n = col, n
    return best_col

# ============================================================
# LOAD DATA
# ============================================================
# CPI is already log(CPI) and uses IMF-like dates
CPI_df = load_monthly_df(CPI_PATH, sheet_name=2, header=1, date_format="%Y-%m")

# EXR uses normal dates (2015-01-01 etc.), and you test depreciation = Δlog(EXR)
EXR_df = load_monthly_df(EXR_PATH, sheet_name=0, header=0, date_format=None)

# Reserves: choose correct sheet automatically
res_sheet_name, res_raw_df, res_hits = pick_best_sheet(RES_PATH, COUNTRIES)
print(f"[RESERVES] Using sheet: {res_sheet_name} (matched {res_hits}/{len(COUNTRIES)} country columns)")
res_raw_df = res_raw_df.copy()
res_raw_df.columns = res_raw_df.columns.map(lambda x: str(x).strip())
res_date_col = find_date_col(res_raw_df)
res_raw_df[res_date_col] = pd.to_datetime(res_raw_df[res_date_col], errors="coerce")
RES_df = res_raw_df.dropna(subset=[res_date_col]).set_index(res_date_col).sort_index()
RES_df = RES_df.loc[:, ~RES_df.columns.str.contains("^Unnamed", case=False, na=False)]

# Long-term rate and Short-term rate: normal dates in your sheets
LTR_df = load_monthly_df(LTR_PATH, sheet_name=0, header=0, date_format=None)
STR_df = load_monthly_df(STR_PATH, sheet_name=0, header=0, date_format=None)

# M2: from your screenshot it looks like IMF-like dates (2015-M01). If that’s true:
# - If your date column is already like "2015-M01", set date_format="%Y-%m"
# - If it is "2015-01-01", set date_format=None
M2_df = load_monthly_df(M2_PATH, sheet_name=0, header=0, date_format="%Y-%m")

# ============================================================
# FIX STR Thailand issue (DO NOT "drop" Excel-derived columns)
# ============================================================
# You said the second Thailand column is the one you want (reversed/fixed).
# In Python, we should just SELECT the correct column, not drop anything.

# Common variants we’ll check; add more if your headers differ
th_candidates = ["Thailand", "Thailand st", "Thailand_st", "Thailand (st)", "Thailand  ", "Thailand st "]
STR_TH_COL = best_matching_column(STR_df, th_candidates)

if STR_TH_COL is None:
    print(f"[WARN] Could not find Thailand column in STR. Available: {list(STR_df.columns)}")
else:
    print(f"[STR] Thailand column chosen: '{STR_TH_COL}'")

# ============================================================
# RUN TESTS
# ============================================================
print("\n=== CPI seasonality on inflation: Δlog(CPI) (CPI already logged) ===")
for c in COUNTRIES:
    if c not in CPI_df.columns:
        print(f"[SKIP] CPI {c}: column not found")
        continue
    infl = trimmed_country_series(CPI_df, c).diff().dropna()
    p = seasonality_f_test(infl, f"ACF of inflation Δlog(CPI) - {c}", show_plot=True, lags=36)
    print(f"{c} - CPI inflation seasonality p-value: {p:.6f}")

print("\n=== EXR seasonality on depreciation: Δlog(EXR) ===")
for c in COUNTRIES:
    if c not in EXR_df.columns:
        print(f"[SKIP] EXR {c}: column not found")
        continue
    exr_level = trimmed_country_series(EXR_df, c)
    dlog_exr = np.log(exr_level.where(exr_level > 0)).diff().dropna()
    p = seasonality_f_test(dlog_exr, f"ACF of Δlog(EXR) - {c}", show_plot=True, lags=36)
    print(f"{c} - EXR Δlog seasonality p-value: {p:.6f}")

print("\n=== INTERNATIONAL RESERVES seasonality on growth: Δlog(Reserves) ===")
for c in COUNTRIES:
    if c not in RES_df.columns:
        print(f"[SKIP] RES {c}: column not found in sheet '{res_sheet_name}'. Available: {list(RES_df.columns)}")
        continue
    res_level = trimmed_country_series(RES_df, c).dropna()
    dlog_res = np.log(res_level.where(res_level > 0)).diff().dropna()
    print(f"{c}: level_n={len(res_level)}, dlog_n={len(dlog_res)}")
    p = seasonality_f_test(dlog_res, f"ACF of Δlog(Reserves) - {c}", show_plot=True, lags=36)
    print(f"{c} - Reserves Δlog seasonality p-value: {p:.6f}")

print("\n=== LONG-TERM INTEREST RATE seasonality on changes: Δr ===")
for c in COUNTRIES:
    if c not in LTR_df.columns:
        print(f"[SKIP] LTR {c}: column not found")
        continue
    r_level = trimmed_country_series(LTR_df, c)
    dr = r_level.diff().dropna()
    print(f"{c}: level_n={len(r_level)}, diff_n={len(dr)}")
    p = seasonality_f_test(dr, f"ACF of ΔLong-term rate - {c}", show_plot=True, lags=36)
    print(f"{c} - Long-term rate Δ seasonality p-value: {p:.6f}")

print("\n=== SHORT-TERM (3-month) INTEREST RATE seasonality on changes: Δr ===")
for c in COUNTRIES:
    # Thailand uses the chosen Thailand column (the “fixed” one)
    col = STR_TH_COL if (c == "Thailand") else c

    if col is None or col not in STR_df.columns:
        print(f"[SKIP] STR {c}: column not found. Available: {list(STR_df.columns)}")
        continue

    r_level = trimmed_country_series(STR_df, col)
    dr = r_level.diff().dropna()
    print(f"{c} (col='{col}'): level_n={len(r_level)}, diff_n={len(dr)}")
    p = seasonality_f_test(dr, f"ACF of ΔShort-term (3m) rate - {c}", show_plot=True, lags=36)
    print(f"{c} - Short-term rate Δ seasonality p-value: {p:.6f}")

print("\n=== M2 seasonality on growth: Δlog(M2) ===")
for c in COUNTRIES:
    if c not in M2_df.columns:
        print(f"[SKIP] M2 {c}: column not found. Available: {list(M2_df.columns)}")
        continue

    m2_level = trimmed_country_series(M2_df, c).dropna()
    dlog_m2 = np.log(m2_level.where(m2_level > 0)).diff().dropna()
    print(f"{c}: level_n={len(m2_level)}, dlog_n={len(dlog_m2)}")
    p = seasonality_f_test(dlog_m2, f"ACF of Δlog(M2) - {c}", show_plot=True, lags=36)
    print(f"{c} - M2 Δlog seasonality p-value: {p:.6f}")