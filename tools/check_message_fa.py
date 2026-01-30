import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Check if message_fa == fa + billpayfa + debitfa")
    parser.add_argument("csv_path", nargs="?", default="data/analysis/analysis_df.csv",
                        help="Path to CSV (default: output/analysis/fausebal/trainval_data.csv)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print("CSV not found:", csv_path)
        sys.exit(1)

    print("Reading:", csv_path.resolve())
    df = pd.read_csv(csv_path)
    cols = ["message_fa", "fa", "billpayfa", "debitfa"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        print("Available columns (truncated):", list(df.columns)[:50])
        sys.exit(2)

    # Basic info
    print("Rows:", len(df))
    for c in cols:
        print(f"NaNs in {c}:", int(df[c].isna().sum()))

    # Compute sums and mismatches
    s = df["fa"].astype(float) + df["billpayfa"].astype(float) + df["debitfa"].astype(float)
    m = df["message_fa"].astype(float)

    # Identify rows with all components present
    components_present = (~df["fa"].isna()) & (~df["billpayfa"].isna()) & (~df["debitfa"].isna())
    message_present = ~m.isna()

    comparable = components_present & message_present
    mismask = comparable & (~np.isclose(m, s))

    print("Comparable rows:", int(comparable.sum()))
    print("Mismatches (message_fa != sum):", int(mismask.sum()))
    if int(mismask.sum()) > 0:
        sample = df.loc[mismask, ["message_fa", "fa", "billpayfa", "debitfa"]].copy()
        sample["computed_sum"] = s[mismask]
        sample["diff"] = sample["message_fa"] - sample["computed_sum"]
        print("\nSample mismatches (up to 10):")
        print(sample.head(10).to_string(index=False))

    # Additional diagnostics for rows with partial missingness
    partial_missing = (~comparable) & (message_present | components_present)
    if int(partial_missing.sum()) > 0:
        print("\nRows with partial missingness (message or components missing):", int(partial_missing.sum()))

    # Summary when allowing NaNs treated as zeros
    s0 = df["fa"].fillna(0).astype(float) + df["billpayfa"].fillna(0).astype(float) + df["debitfa"].fillna(0).astype(float)
    m0 = df["message_fa"].fillna(0).astype(float)
    mismask0 = ~np.isclose(m0, s0)
    print("Rows unequal when treating NaNs as zeros:", int(mismask0.sum()), "/", len(df))


if __name__ == "__main__":
    main()
