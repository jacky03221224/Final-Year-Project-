"""
Usage:
Default: python Sentiment_Score/Directional_Score/combine.py
Custom: python Sentiment_Score/Directional_Score/combine.py --ticker ABT
"""
import argparse
import pandas as pd
import os

def combine_sample_and_output(output_csv_name, sample_csv_name):
    # Directories
    outputs_dir = os.path.join("Sentiment_Score", "Directional_Score", "Directional_Outputs")
    sample_dir = os.path.join("Sentiment_Score", "Directional_Score", "Directional_Sample")
    train_dir = os.path.join("Sentiment_Score", "Directional_Score", "Directional_Train")

    # Build full paths
    output_csv = os.path.join(outputs_dir, output_csv_name)
    sample_csv = os.path.join(sample_dir, sample_csv_name)


    # Read the sampled_full CSV, skipping the header row
    sample_df = pd.read_csv(sample_csv, skiprows=1, header=None)
    # Read the output CSV (index, score), skip no header
    output_df = pd.read_csv(output_csv, header=None, names=["idx", "directional_score"])

    # Check that the number of rows match
    if len(sample_df) != len(output_df):
        raise ValueError(f"Row count mismatch: {len(sample_df)} in sample, {len(output_df)} in output.")

    # Combine: add the score as a new column
    combined_df = sample_df.copy()
    combined_df[4] = output_df["directional_score"].values
    combined_df.columns = ["date", "headline", "source", "summary", "directional_sentiment"]

    # Determine ticker from output file name
    ticker = output_csv_name.split('_')[0]
    train_filename = f"{ticker}_train.csv"
    train_path = os.path.join(train_dir, train_filename)

    # Save to train_dir with header
    combined_df.to_csv(train_path, index=False, header=True)
    print(f"Combined file saved to {train_path}")

def main():
    parser = argparse.ArgumentParser(description="Combine sampled_full and output CSVs and save as <Ticker>_train.csv in Directional_Train.")
    parser.add_argument("--ticker", help="Ticker symbol (e.g. ABT)")
    args = parser.parse_args()

    # Hardcoded array of tickers
    preset_ticker = [
        "ABT", "AMZN", "AVGO", "BEP", "DHR", "ENPH", "FSLR", "ISRG", "LLY", "META", "NEE", "NVO", "PLUG", "SNOW", "TSLA"
    ]

    if args.ticker:
        ticker_list = [args.ticker]
    else:
        ticker_list = preset_ticker

    for ticker in ticker_list:
        sample_name = f"{ticker}_sample_full.csv"
        output_name = f"{ticker}_output.csv"
        try:
            combine_sample_and_output(output_name, sample_name)
        except Exception as e:
            print(f"Error combining for {ticker}: {e}")

if __name__ == "__main__":
    main()
