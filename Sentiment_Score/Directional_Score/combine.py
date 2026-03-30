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

    # Read the sampled_full and output CSVs, skipping the header row
    sample_df = pd.read_csv(sample_csv, skiprows=1, header=None)
    output_df = pd.read_csv(output_csv, skiprows=1, header=None, names=["date", "category", "directional_score"])

    # Check that the number of rows match
    if len(sample_df) != len(output_df):
        raise ValueError(f"Row count mismatch: {len(sample_df)} in sample, {len(output_df)} in output.")

    # Combine columns as specified
    combined_df = pd.concat([
        sample_df.reset_index(drop=True),
        output_df[["category", "directional_score"]].reset_index(drop=True)
    ], axis=1)
    combined_df.columns = ["date", "headline", "source", "summary", "category", "directional_score"]

    # Determine ticker from output file name
    ticker = output_csv_name.split('_')[0]
    train_filename = f"{ticker}_train.csv"
    train_path = os.path.join(train_dir, train_filename)

    # Save to train_dir
    combined_df.to_csv(train_path, index=False)
    print(f"Combined file saved to {train_path}")

def main():
    parser = argparse.ArgumentParser(description="Combine sampled_full and output CSVs and save as <Ticker>_train.csv in Directional_Train.")
    parser.add_argument("--sample", required=True, help="Name of the sampled_full CSV file in Directional_Sample (e.g., ABT_sample_full.csv)")
    args = parser.parse_args()

    # Always infer output file name from sample
    ticker = args.sample.split('_')[0]
    output_name = f"{ticker}_output.csv"
    combine_sample_and_output(output_name, args.sample)

if __name__ == "__main__":
    main()
