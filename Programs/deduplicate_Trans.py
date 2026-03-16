"""
Weekly duplicate-news removal with safer clustering.

Fixes vs. your current script:
    1) Deduplicates strictly within Monday–Sunday weekly buckets.
    2) Uses Agglomerative Clustering (average linkage) on cosine distance
         to avoid "chaining" that can over-merge clusters.
    3) Bumps default similarity threshold to 0.70 (tunable).
    4) Adds light boilerplate cleanup before vectorization.
    5) Keeps the representative-selection rule: longest summary; tie -> earliest date.
"""

import argparse
import os
from typing import List, Tuple, Dict, Any
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

EXPECTED_COLS = ['date', 'headline', 'source', 'summary']

# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Weekly duplicate-news removal with safer clustering.")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to input CSV file.")
    parser.add_argument("--stock-name", type=str, default=None,
                        help="Optional stock name (used for output file naming). If not set, inferred from CSV filename.")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Cosine similarity threshold (>=) to link articles. Default: 0.70")
    parser.add_argument("--max-examples", type=int, default=5,
                        help="Max duplicate clusters to print as examples. Default: 5")
    parser.add_argument("--date-format", type=str, default=None,
                        help="Optional explicit date format (e.g., %%Y-%%m-%%d). If not set, parsing is inferred.")
    parser.add_argument("--min-chars", type=int, default=10,
                        help="Minimum characters required after combining headline+summary to include (noise filter). Default: 10")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()



# -------------------------
# IO & Validation
# -------------------------
def load_and_validate(csv_path: str, date_format: str = None) -> pd.DataFrame:
    # Load
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Validate structure
    cols = [c.strip().lower() for c in df.columns]
    col_map = {}
    for expected in EXPECTED_COLS:
        if expected in cols:
            col_map[expected] = df.columns[cols.index(expected)]
        else:
            raise ValueError(f"Missing required column: '{expected}'. "
                             f"Found columns: {list(df.columns)}")

    # Keep only the expected columns in the canonical order
    df = df[[col_map[c] for c in EXPECTED_COLS]]
    df.columns = EXPECTED_COLS

    # Parse date
    if date_format:
        df['date'] = pd.to_datetime(df['date'], format=date_format, errors='coerce')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # let pandas infer

    if df['date'].isna().any():
        bad_rows = df[df['date'].isna()]
        raise ValueError(f"Some dates could not be parsed. Check 'date' format.\n"
                         f"First few problematic rows:\n{bad_rows.head()}")

    # Normalize text fields (ensure str)
    for c in ['headline', 'source', 'summary']:
        df[c] = df[c].fillna("").astype(str)

    return df


# -------------------------
# Week bucketing (Monday–Sunday)
# -------------------------
def monday_of_week(dt: pd.Timestamp) -> pd.Timestamp:
    return dt - pd.Timedelta(days=dt.weekday())


def make_week_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['week_start'] = (df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='D')).dt.normalize()
    return df


# -------------------------
# Pre-clean/normalize to reduce boilerplate-driven false positives
# -------------------------
_BOILERPLATE_PATTERNS = [
    r'^(breaking|update|live):\s*',
    r'^wall street breakfast.*?:\s*',
    r'^in the latest trading session[, ]*',
    r'^key insights[:\-]\s*',
    r'\b(read more|click here|to watch more.*)\b',
]

def clean_text_for_similarity(headline: str, summary: str) -> str:
    txt = f"{headline or ''} {summary or ''}".lower()
    txt = re.sub(r'\s+', ' ', txt).strip()
    for pat in _BOILERPLATE_PATTERNS:
        txt = re.sub(pat, '', txt)
    # collapse punctuation -> space, then re-collapse whitespace
    txt = re.sub(r'[^\w\s%\.]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt


# -------------------------
# Vectorization (char n-grams)
# -------------------------
def vectorize_text(texts: List[str]):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast, and effective
    X = model.encode(texts, show_progress_bar=False)
    return X


# -------------------------
# Clustering (Agglomerative, average linkage on cosine distance)
# -------------------------
def cluster_agglomerative_from_tfidf(X, sim_threshold: float) -> List[List[int]]:
    sim = cosine_similarity(X)
    dist = 1 - sim  # convert to distance for clustering

    # average linkage tends to avoid chaining vs connected components on a graph
    clu = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=1 - sim_threshold,  # distance <= (1 - sim_threshold)
        n_clusters=None
    ).fit(dist)

    labels = clu.labels_
    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(idx)
    return list(clusters.values())


# -------------------------
# Representative selection (longest summary; tie -> earliest date)
# -------------------------
def select_representative(df_week: pd.DataFrame, cluster_indices: List[int]) -> int:
    candidates = []
    for pos in cluster_indices:
        row = df_week.iloc[pos]
        summary_len = len((row['summary'] or "").strip())
        candidates.append((summary_len, row['date'], pos))
    # longest summary, then earliest date, then lowest positional index
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    return candidates[0][2]


# -------------------------
# Deduplicate within one week
# -------------------------
def process_week(df_week: pd.DataFrame, sim_threshold: float) -> Tuple[pd.DataFrame, List[List[int]]]:
    if len(df_week) <= 1:
        return df_week.copy(), []

    # Build cleaned combined text
    combined = [
        clean_text_for_similarity(h, s)
        for h, s in zip(df_week['headline'].fillna(''), df_week['summary'].fillna(''))
    ]

    # Vectorize & cluster
    X = vectorize_text(combined)
    clusters = cluster_agglomerative_from_tfidf(X, sim_threshold=sim_threshold)

    keep_positions = set()
    duplicate_clusters = []

    for cl in clusters:
        if len(cl) == 1:
            keep_positions.add(cl[0])
        else:
            rep = select_representative(df_week, cl)
            keep_positions.add(rep)
            duplicate_clusters.append(cl)

    kept_df = df_week.iloc[sorted(list(keep_positions))].copy()
    return kept_df, duplicate_clusters


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    np.random.seed(args.random_seed)

    input_path = args.csv_path
    # Extract TICKER from input file name
    ticker = os.path.basename(input_path).split('_cleaned.csv')[0]
    # 1) Load and inspect
    df = load_and_validate(input_path, date_format=args.date_format)
    original_count = len(df)

    print("=== Step 1: Load & Inspect ===")
    print(f"Rows loaded: {original_count}")
    print(f"Columns: {list(df.columns)}")
    print()

    # 2) Preprocess & weekly grouping
    print("=== Step 2: Preprocess & Weekly Grouping ===")
    df['_combined_text'] = (df['headline'].fillna('') + ' ' + df['summary'].fillna('')).str.lower()
    before_noise = len(df)
    df = df[df['_combined_text'].str.len() >= args.min_chars].copy()
    after_noise = len(df)
    if after_noise < before_noise:
        print(f"Filtered {before_noise - after_noise} very short records (min-chars={args.min_chars}).")

    df = df.sort_values('date').reset_index(drop=True)
    df = make_week_column(df)
    print(f"Data spans from {df['date'].min().date()} to {df['date'].max().date()}.")
    print(f"Weeks to process: {df['week_start'].nunique()}")
    print()

    # 3) Deduplicate per week
    print("=== Step 3–4: Semantic Duplicate Detection & Removal (Weekly) ===")
    kept_all = []
    all_dup_clusters = []  # (week_start, cluster_indices, df_week)

    for wk, df_week in df.groupby('week_start', sort=True):
        df_week = df_week.reset_index(drop=True)
        kept_week, dup_clusters = process_week(df_week, sim_threshold=args.threshold)
        kept_week['week_start'] = wk
        kept_all.append(kept_week)

        if dup_clusters:
            all_dup_clusters.extend([(wk, cl, df_week) for cl in dup_clusters])

    df_dedup = pd.concat(kept_all, ignore_index=True)

    # Keep only requested columns and tidy date format
    df_dedup = df_dedup[EXPECTED_COLS].copy()
    df_dedup['date'] = df_dedup['date'].dt.strftime('%Y-%m-%d')

    final_count = len(df_dedup)
    removed = original_count - final_count

    # 4) Export
    out_name = f"{ticker}_processed.csv"
    df_dedup.to_csv(out_name, index=False, encoding='utf-8')

    # Reporting
    print()
    print("=== Results ===")
    print(f"Similarity threshold : {args.threshold}")
    print(f"Before deduplication : {original_count} rows")
    print(f"After deduplication  : {final_count} rows")
    print(f"Total removed        : {removed} rows")
    print(f"Duplicate groups detected (size ≥ 2): {len(all_dup_clusters)}")

    print(f"Output file: {out_name}")
    print()

    # --- Print example duplicate clusters ---
    if all_dup_clusters:
        print("=== Example Duplicate Clusters ===")
        max_examples = min(args.max_examples, len(all_dup_clusters))
        for i, (wk, cl, df_week) in enumerate(all_dup_clusters[:max_examples], 1):
            rep_idx = select_representative(df_week, cl)
            print(f"\nExample {i} (Week of {wk.date()}):")
            print(f"  Representative article:")
            rep_row = df_week.iloc[rep_idx]
            print(f"    Date: {rep_row['date'].date()} | Source: {rep_row['source']}")
            print(f"    Headline: {rep_row['headline']}")
            print(f"    Summary: {rep_row['summary'][:200]}{'...' if len(rep_row['summary']) > 200 else ''}")
            print(f"  Duplicates:")
            for idx in cl:
                if idx == rep_idx:
                    continue
                row = df_week.iloc[idx]
                print(f"    - Date: {row['date'].date()} | Source: {row['source']}")
                print(f"      Headline: {row['headline']}")
                print(f"      Summary: {row['summary'][:200]}{'...' if len(row['summary']) > 200 else ''}")
        print(f"\n(Showing {max_examples} of {len(all_dup_clusters)} duplicate clusters. Use --max-examples to change.)")
    else:
        print("No duplicate clusters found to show as examples.")

if __name__ == "__main__":
    main()
