import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================
COUNTRIES = ["Thailand", "Philippines", "Korea", "Indonesia"]

# Processed (seasonally adjusted) files
CPI_SA_PATH = "DATA/Processed/CPI_seasonally_adjusted.xlsx"                 # log(CPI) (SA where needed)
M2_SA_PATH  = "DATA/Processed/M2_seasonally_adjusted.xlsx"                  # level M2 (SA)
RES_SA_PATH = "DATA/Processed/Reserves_seasonally_adjusted.xlsx"            # level reserves (SA where needed)
LTR_SA_PATH = "DATA/Processed/Long_term_rate_seasonally_adjusted.xlsx"      # level 10y yield (SA where needed)
STR_SA_PATH = "DATA/Processed/Short_term_rate_seasonally_adjusted.xlsx"     # level 3m rate (SA where needed)

# Not seasonally adjusted
EXR_PATH = "DATA/Aggregated data/Exchange rates/Exchange rates.xlsx"
EXR_SHEET = 0
EXR_HEADER = 0

# Output
OUT_MASTER = "DATA/Processed/armax_master_dataframe.xlsx"

# ============================================================
# HELPERS
# ============================================================
def read_processed_excel(path: str) -> pd.DataFrame:
    """
    Reads your processed SA files which look like:
    col1 = Dates, then country columns.
    """
    df = pd.read_excel(path, sheet_name=0)
    df = df.copy()
    df.columns = df.columns.map(lambda x: str(x).strip())

    # date col typically first column
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # drop unnamed junk
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, na=False)]

    # keep only wanted countries if present
    keep = [c for c in COUNTRIES if c in df.columns]
    return df[keep]

def read_exr(path: str, sheet=0, header=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, header=header).copy()
    df.columns = df.columns.map(lambda x: str(x).strip())

    # find date column robustly
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() == "date" or "date" in str(c).strip().lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, na=False)]

    keep = [c for c in COUNTRIES if c in df.columns]
    return df[keep]

def safe_log_diff(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0)
    return np.log(s).diff()

def safe_diff(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.diff()

def describe_coverage(df: pd.DataFrame, name: str):
    print(f"\n[{name}] coverage per country:")
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        first = s.first_valid_index()
        last = s.last_valid_index()
        n = s.notna().sum()
        print(f"  {c}: n={n}, first={first.date() if first is not None else None}, last={last.date() if last is not None else None}")

# ============================================================
# BUILD MASTER DF
# ============================================================
def main():
    # ---- Load series (levels) ----
    cpi_sa = read_processed_excel(CPI_SA_PATH)   # log(CPI) levels (SA where needed)
    m2_sa  = read_processed_excel(M2_SA_PATH)    # M2 levels
    res_sa = read_processed_excel(RES_SA_PATH)   # reserves levels
    ltr_sa = read_processed_excel(LTR_SA_PATH)   # 10y yield levels
    str_sa = read_processed_excel(STR_SA_PATH)   # 3m rate levels
    exr    = read_exr(EXR_PATH, sheet=EXR_SHEET, header=EXR_HEADER)  # EXR levels

    # ---- Quick coverage check ----
    describe_coverage(cpi_sa, "CPI_SA (log level)")
    describe_coverage(m2_sa, "M2_SA (level)")
    describe_coverage(res_sa, "RES_SA (level)")
    describe_coverage(ltr_sa, "LTR_SA (level)")
    describe_coverage(str_sa, "STR_SA (level)")
    describe_coverage(exr, "EXR (level)")

    # ---- Transform into modeling variables ----
    # Inflation: Δlog(CPI_SA) (since CPI_SA is already log(CPI))
    infl = cpi_sa.diff()

    # Money growth: Δlog(M2_SA)
    m2_g = m2_sa.apply(safe_log_diff)

    # Reserves growth: Δlog(Reserves_SA)
    res_g = res_sa.apply(safe_log_diff)

    # Depreciation: Δlog(EXR)
    exr_dep = exr.apply(safe_log_diff)

    # Rate changes (no logs)
    d_ltr = ltr_sa.apply(safe_diff)
    d_str = str_sa.apply(safe_diff)

    # ---- Stack into one master dataframe with clear column names ----
    def rename_block(df, prefix):
        return df.rename(columns={c: f"{c}_{prefix}" for c in df.columns})

    master = pd.concat([
        rename_block(infl, "infl"),          # Δlog(CPI_SA)
        rename_block(m2_g, "dlog_m2"),
        rename_block(exr_dep, "dlog_exr"),
        rename_block(res_g, "dlog_res"),
        rename_block(d_ltr, "d_ltr"),
        rename_block(d_str, "d_str"),
    ], axis=1).sort_index()

    # ---- Save master (with NaNs retained) ----
    master.to_excel(OUT_MASTER)
    print(f"\nSaved master ARMAX dataframe to: {OUT_MASTER}")

    # ---- (Optional) Create “longest sample” per typical specification ----
    # Example: Indonesia spec uses all X variables
    for country in COUNTRIES:
        cols = [
            f"{country}_infl",
            f"{country}_dlog_m2",
            f"{country}_dlog_exr",
            f"{country}_dlog_res",
            f"{country}_d_ltr",
            f"{country}_d_str",
        ]
        existing = [c for c in cols if c in master.columns]
        model_df = master[existing].dropna()

        if len(model_df) == 0:
            print(f"[WARN] {country}: model_df is empty after dropna().")
            continue

        print(f"\n[{country}] model-ready sample (all vars):")
        print(f"  start = {model_df.index.min().date()}, end = {model_df.index.max().date()}, n = {len(model_df)}")

if __name__ == "__main__":
    main()