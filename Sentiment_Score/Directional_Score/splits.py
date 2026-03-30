import argparse
import os
import pandas as pd

def split_csv(input_csv, chunk_size=100):
    # Derive ticker from filename
    base = os.path.basename(input_csv)
    ticker = base.split('_')[0]
    
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_rows)
        chunk = df.iloc[start:end]
        out_name = f"{ticker}_sample_{i+1}.csv"
        chunk.to_csv(out_name, index=False)
        print(f"Saved {out_name} with {len(chunk)} rows.")

def main():
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple files of 100 rows each.")
    parser.add_argument("input_csv", help="Input CSV filename (e.g. ABT_sample_full.csv)")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of rows per output file (default: 100)")
    args = parser.parse_args()

    # Default directory for input files
    default_dir = os.path.join("Sentiment_Score", "Directional_Score", "Directional_sample")
    input_path = args.input_csv
    if not os.path.isabs(input_path):
        input_path = os.path.join(default_dir, input_path)
    split_csv(input_path, args.chunk_size)

if __name__ == "__main__":
    main()
