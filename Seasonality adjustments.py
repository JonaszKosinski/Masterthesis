import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


# ============================================================
# OUTPUT
# ============================================================
os.makedirs("DATA/Processed", exist_ok=True)
SEASONAL_PERIOD = 12  # monthly

# ============================================================
# PATHS
# ============================================================
CPI_PATH = "DATA/Aggregated data/CPI/CPI aggregated.xlsx"
RES_PATH = "DATA/Aggregated data/International reserves/International reserves aggregated.xlsx"
LTR_PATH = "DATA/Aggregated data/Long term interest/Long term interest rate aggregated.xlsx"
STR_PATH = "DATA/Aggregated data/Short term interest/Short term 3 month interest rate aggregated.xlsx"
M2_PATH  = "DATA/Aggregated data/M2/M2 aggregated.xlsx"   # <-- adjust if your folder/file name differs

COUNTRIES = ["Thailand", "Philippines", "Korea", "Indonesia"]

# ============================================================
# WHICH SERIES TO ADJUST (based on your seasonality tests)
# ============================================================
ADJUST = {
    "CPI": {        # CPI file already contains log(CPI) series by country
        "Indonesia": True,
        "Philippines": True,
        "Korea": True,
        "Thailand": False,
    },
    "RES": {        # reserves level series
        "Indonesia": True,
        "Philippines": False,
        "Korea": False,
        "Thailand": False,
    },
    "LTR": {        # long-term rate level series
        "Indonesia": False,
        "Philippines": False,
        "Korea": True,
        "Thailand": False,
    },
    "STR": {        # short-term rate level series
        "Indonesia": False,
        "Philippines": False,
        "Korea": True,
        "Thailand": False,
    },
    "M2": {         # based on your printed p-values ~ 0 for all
        "Indonesia": True,
        "Philippines": True,
        "Korea": True,
        "Thailand": True,
    },
}

# ============================================================
# HELPERS
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

def stl_adjust(series: pd.Series, period=12) -> pd.Series:
    """
    STL seasonal adjustment: SA = series - seasonal component
    Returns a series aligned to the original index (NaNs where not computable).
    """
    # Keep full index for alignment
    full_index = series.index

    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.dropna()

    # Need enough data for STL
    if len(s) < 2 * period + 1:
        return pd.Series(index=full_index, dtype=float)

    res = STL(s, period=period, robust=True).fit()
    sa = s - res.seasonal
    return sa.reindex(full_index)

def seasonally_adjust_df(df: pd.DataFrame, adjust_map: dict, label: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for c in COUNTRIES:
        if c not in df.columns:
            print(f"[{label}] [SKIP] {c}: column not found. Available: {list(df.columns)}")
            continue

        raw = trimmed_country_series(df, c)
        do_adj = adjust_map.get(c, False)

        if do_adj:
            sa = stl_adjust(raw, period=SEASONAL_PERIOD)
            out[c] = sa
            print(f"[{label}] {c}: SA (STL)")
        else:
            out[c] = raw
            print(f"[{label}] {c}: kept original (no seasonality)")

    return out

# ============================================================
# LOAD + ADJUST + SAVE
# ============================================================
def main():
    # CPI: your file has log(CPI) in sheet 2, header row 1, date like 2015-M01
    CPI_df = load_monthly_df(CPI_PATH, sheet_name=2, header=1, date_format="%Y-%m")
    CPI_sa = seasonally_adjust_df(CPI_df, ADJUST["CPI"], "CPI")
    CPI_sa.to_excel("DATA/Processed/CPI_seasonally_adjusted.xlsx")
    print("[CPI] Saved -> DATA/Processed/CPI_seasonally_adjusted.xlsx\n")

    # RESERVES: pick correct sheet automatically (the one matching country columns)
    res_sheet, res_raw, hits = pick_best_sheet(RES_PATH, COUNTRIES)
    print(f"[RES] Using sheet '{res_sheet}' (matched {hits}/{len(COUNTRIES)} columns)")
    res_raw = res_raw.copy()
    res_raw.columns = res_raw.columns.map(lambda x: str(x).strip())
    res_date_col = find_date_col(res_raw)
    res_raw[res_date_col] = pd.to_datetime(res_raw[res_date_col], errors="coerce")
    RES_df = res_raw.dropna(subset=[res_date_col]).set_index(res_date_col).sort_index()
    RES_df = RES_df.loc[:, ~RES_df.columns.str.contains("^Unnamed", case=False, na=False)]

    RES_sa = seasonally_adjust_df(RES_df, ADJUST["RES"], "RES")
    RES_sa.to_excel("DATA/Processed/Reserves_seasonally_adjusted.xlsx")
    print("[RES] Saved -> DATA/Processed/Reserves_seasonally_adjusted.xlsx\n")

    # LONG-TERM RATE: levels, normal date parsing
    LTR_df = load_monthly_df(LTR_PATH, sheet_name=0, header=0, date_format=None)
    LTR_sa = seasonally_adjust_df(LTR_df, ADJUST["LTR"], "LTR")
    LTR_sa.to_excel("DATA/Processed/Long_term_rate_seasonally_adjusted.xlsx")
    print("[LTR] Saved -> DATA/Processed/Long_term_rate_seasonally_adjusted.xlsx\n")

    # SHORT-TERM RATE: levels, normal date parsing
    STR_df = load_monthly_df(STR_PATH, sheet_name=0, header=0, date_format=None)

    # If your file contains the extra wrong Thailand column name variants, drop the wrong one
    # (keep the properly ordered Thailand column that you said is correct).
    # If BOTH exist, we keep "Thailand" and drop "Thailand st".
    cols = [str(c).strip() for c in STR_df.columns]
    STR_df.columns = cols
    if "Thailand st" in STR_df.columns and "Thailand" in STR_df.columns:
        STR_df = STR_df.drop(columns=["Thailand st"])
        print("[STR] Dropped 'Thailand st' and kept 'Thailand'")

    STR_sa = seasonally_adjust_df(STR_df, ADJUST["STR"], "STR")
    STR_sa.to_excel("DATA/Processed/Short_term_rate_seasonally_adjusted.xlsx")
    print("[STR] Saved -> DATA/Processed/Short_term_rate_seasonally_adjusted.xlsx\n")

    # M2: adjust sheet/header if yours differs
    M2_df = load_monthly_df(M2_PATH, sheet_name=0, header=0, date_format="%Y-%m")  # if your date is like 2015-M01
    # If your M2 date is already like 2015-01-01, change date_format=None.
    M2_sa = seasonally_adjust_df(M2_df, ADJUST["M2"], "M2")
    M2_sa.to_excel("DATA/Processed/M2_seasonally_adjusted.xlsx")
    print("[M2] Saved -> DATA/Processed/M2_seasonally_adjusted.xlsx\n")

if __name__ == "__main__":
    main()





