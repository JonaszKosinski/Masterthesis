import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import os

os.makedirs("DATA/Processed", exist_ok=True)

SEASONAL_PERIOD = 12

# Decide per country (based on your seasonality tests)
ADJUST_CPI = {
    "Indonesia": True,
    "Philippines": True,
    "Korea": True,
    "Thailand": False,   # keep as-is
}

# ==========
# SETTINGS
# ==========
CPI_PATH = "DATA/Aggregated data/CPI/CPI aggregated.xlsx"
SHEET_NAME = 2
HEADER_ROW = 1

OUT_PATH = "DATA/Processed/CPI_seasonally_adjusted.xlsx"
SEASONAL_PERIOD = 12  # monthly

# Based on your seasonality findings
ADJUST_CPI = {
    "Indonesia": True,
    "Philippines": True,
    "Korea": True,
    "Thailand": False,
}

def load_cpi():
    df = pd.read_excel(CPI_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW)
    df.columns = df.columns.map(lambda x: str(x).strip())

    if "Date" not in df.columns:
        raise ValueError("Could not find 'Date' column in CPI sheet.")

    df["Date"] = df["Date"].astype(str).str.replace("-M", "-", regex=False)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="coerce")

    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df

def stl_adjust(series: pd.Series, period=12) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()

    # If too short, just return NaNs on full index (or original if you prefer)
    if len(s) < 2 * period + 1:
        return pd.Series(index=series.index, dtype=float)

    res = STL(s, period=period, robust=True).fit()
    sa = s - res.seasonal
    return sa

def main():
    df = load_cpi()

    sa_df = pd.DataFrame(index=df.index)

    for country in df.columns:
        raw = pd.to_numeric(df[country], errors="coerce")

        # Default: adjust unless specified
        do_adjust = ADJUST_CPI.get(country, True)

        if do_adjust:
            sa = stl_adjust(raw, period=SEASONAL_PERIOD)
            sa_df[country] = sa.reindex(df.index)  # keep original index
        else:
            sa_df[country] = raw  # keep as-is

        print(f"{country}: {'SA (STL)' if do_adjust else 'kept original (no seasonality)'}")

    sa_df.to_excel(OUT_PATH)
    print(f"\nSaved seasonally adjusted CPI to: {OUT_PATH}")

if __name__ == "__main__":
    main()