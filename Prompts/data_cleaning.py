"""
A data cleaning script for financial news CSV files.

Usage:
    python clean_news.py --input path/to/news.csv --stock-name TICKER

Input:
    {stock_name}.csv with columns: ['date', 'headline', 'source', 'summary']

Output:
    {stock_name}_cleaned.csv
"""

import argparse
import os
import sys
import re
from typing import Tuple, Dict

import pandas as pd
import numpy as np
import csv

def preprocess_csv_quote_commas(input_path: str, output_path: str, sep: str = ","):
    """
    Rewrite CSV so any field containing a comma is wrapped in quotes.
    Args:
        input_path: Path to original CSV.
        output_path: Path to save fixed CSV.
        sep: CSV delimiter (default ',').
    """
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8", newline="") as outfile:
        reader = csv.reader(infile, delimiter=sep)
        writer = csv.writer(outfile, delimiter=sep, quoting=csv.QUOTE_MINIMAL)
        header = next(reader)
        # Ensure header uses 'headline'
        header = ["headline" if col == "title" else col for col in header]
        writer.writerow(header)
        summary_idx = header.index("summary") if "summary" in header else 3
        headline_idx = header.index("headline") if "headline" in header else 1
        for row in reader:
            # Remove all double and single quotes from headline and summary
            if len(row) > headline_idx:
                row[headline_idx] = row[headline_idx].replace('"', '').replace("'", '')
            if len(row) > summary_idx:
                row[summary_idx] = row[summary_idx].replace('"', '').replace("'", '')
            for i in range(len(row)):
                if i != headline_idx and i != summary_idx:
                    row[i] = row[i].replace('"', '').replace("'", '')
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a financial news CSV for downstream NLP tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV (UTF-8).")
    parser.add_argument("--stock-name", "-s", required=True, help="Stock name/ticker used to name the output file.")
    parser.add_argument("--sep", default=",", help="CSV delimiter.")
    parser.add_argument("--examples", type=int, default=5, help="How many removed examples to display per category.")
    return parser.parse_args()


def print_header(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def load_and_prepare(path: str, sep: str) -> pd.DataFrame:
    # Load UTF-8 CSV with proper quoting to handle line breaks in fields
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8",
            sep=sep,
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            engine="python"
        )
    except UnicodeDecodeError as e:
        print("ERROR: The file could not be read with UTF-8 encoding. "
              "Please ensure the CSV is saved as UTF-8.", file=sys.stderr)
        raise e

    # Drop rows with more than 4 columns (malformed rows)
    if df.shape[1] > 4:
        df = df.iloc[:, :4].copy()

    # Ensure at least 4 columns; take first 4 and rename as required
    if df.shape[1] < 4:
        raise ValueError(
            f"Expected at least 4 columns (date, headline, source, summary) but found {df.shape[1]}."
        )

    df.columns = ["date", "headline", "source", "summary"]

    # Normalize whitespace in string columns; keep NaN as NaN
    for col in ["headline", "source", "summary"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
            df[col] = df[col].str.strip()

    return df


def standardize_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Parse with dayfirst=True; errors='coerce' -> NaT for invalid
    parsed = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    invalid_mask = parsed.isna()
    removed = df[invalid_mask].copy()

    df = df[~invalid_mask].copy()
    # Use date-only in YYYY-MM-DD format
    df["date"] = parsed[~invalid_mask].dt.strftime("%Y-%m-%d")

    return df, removed


def remove_missing_and_placeholders(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Treat blanks as missing
    df["headline"] = df["headline"].replace(r"^\s*$", pd.NA, regex=True)
    df["summary"] = df["summary"].replace(r"^\s*$", pd.NA, regex=True)

    # Drop missing headline or summary
    missing_mask = df["headline"].isna() | df["summary"].isna()
    missing_removed = df[missing_mask].copy()
    df = df[~missing_mask].copy()

    # Remove rows where headline or summary contains '#NAME?'
    placeholder_mask = (
        df["headline"].str.contains(r"#NAME\?", case=False, na=False) |
        df["summary"].str.contains(r"#NAME\?", case=False, na=False)
    )
    placeholder_removed = df[placeholder_mask].copy()
    df = df[~placeholder_mask].copy()

    removed_all = pd.concat([missing_removed, placeholder_removed], axis=0, ignore_index=True)
    return df, removed_all


def remove_zacks_promotions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows where summary contains (case-insensitive):
      - Full text:
        'Looking for stock market analysis and research with proves results? Zacks.com offers in-depth financial research with over 30years of proven results.'
      - Or keywords: 'Zacks.com', 'proven results'
    """
    full_text = (
        "Looking for stock market analysis and research with proves results? "
        "Zacks.com offers in-depth financial research with over 30years of proven results."
    )
    # Build boolean mask
    s = df["summary"].fillna("").astype(str)

    full_text_mask = s.str.contains(re.escape(full_text), case=False, na=False)
    zacks_mask = s.str.contains(r"\bZacks\.com\b", case=False, na=False)
    proven_results_mask = s.str.contains(r"\bproven\s+results\b", case=False, na=False)

    remove_mask = full_text_mask | zacks_mask | proven_results_mask

    removed = df[remove_mask].copy()
    df = df[~remove_mask].copy()
    return df, removed


def drop_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dup_mask = df.duplicated(subset=["headline", "summary"], keep="first")
    removed = df[dup_mask].copy()
    df = df[~dup_mask].copy()
    return df, removed


def assert_integrity(df: pd.DataFrame):
    # Ensure required columns and constraints hold
    required = ["date", "headline", "source", "summary"]
    assert all(c in df.columns for c in required), "Missing required columns after cleaning."
    assert df["date"].notna().all(), "Found missing dates after cleaning."
    assert df["headline"].notna().all() and df["summary"].notna().all(), \
        "Found missing headline/summary after cleaning."
    # No '#NAME?' placeholders
    assert not (
        df["headline"].fillna("").str.contains(r"#NAME\?", case=False).any() or
        df["summary"].fillna("").str.contains(r"#NAME\?", case=False).any()
    ), "Found '#NAME?' placeholders after cleaning."
    # No duplicate (headline, summary)
    assert not df.duplicated(subset=["headline", "summary"]).any(), \
        "Found duplicate (headline, summary) pairs after cleaning."


def main():
    args = parse_args()

    inp = args.input
    stock = args.stock_name

    if not os.path.isfile(inp):
        print(f"ERROR: Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    # Preprocess CSV to ensure all fields with commas are quoted
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix="_preprocessed.csv") as tmpfile:
        tmp_path = tmpfile.name
    preprocess_csv_quote_commas(inp, tmp_path, sep=args.sep)

    df = load_and_prepare(tmp_path, sep=args.sep)
    total_before = len(df)

    df, _ = standardize_dates(df)
    df, _ = remove_missing_and_placeholders(df)
    df, _ = remove_zacks_promotions(df)
    df, _ = drop_duplicates(df)
    df = df.reset_index(drop=True)
    df = df[["date", "headline", "source", "summary"]]
    assert_integrity(df)
    out_path = f"{stock}_cleaned.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Total rows before cleaning: {total_before}")
    print(f"Total rows after cleaning:  {len(df)}")
    print(f"Exported cleaned dataset to: {out_path}")


if __name__ == "__main__":
    main()