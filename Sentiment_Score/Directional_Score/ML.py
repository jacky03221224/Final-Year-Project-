import numpy as np
import os
import pandas as pd
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer


REQUIRED_TRAIN_COLS = [
    "date",
    "headline",
    "source",
    "summary",
    "directional_score",
]

REQUIRED_PREDICT_COLS = [
    "date",
    "headline",
    "source",
    "summary",
]

def calculate_daily_sentiment(df):
    # 1. ENFORCE BOUNDARIES (Clip directional score to -1.0 and 1.0)
    df['directional_score'] = np.clip(df['directional_score'], -1.0, 1.0)
    # 2. CALCULATE PER-ARTICLE SCORE
    if all(col in df.columns for col in ['relevance', 'reliability']):
        df['article_score'] = df['directional_score'] * df['relevance'] * df['reliability']
    else:
        df['article_score'] = df['directional_score']
    # 3. AGGREGATE BY DATE (Using Mean/Average)
    daily_scores = df.groupby('date')['article_score'].mean().reset_index()
    # Rename column for clarity
    daily_scores.rename(columns={'article_score': 'daily_sentiment_signal'}, inplace=True)
    return daily_scores
import argparse

def check_output_row_count(input_csv_path: str, output_csv_path: str) -> None:
    """
    Checks if the number of rows in the output file matches the input processed.csv file.
    Raises AssertionError if the counts do not match.
    """
    input_df = pd.read_csv(input_csv_path)
    output_df = pd.read_csv(output_csv_path)
    assert len(input_df) == len(output_df), (
        f"Row count mismatch: input ({len(input_df)}) vs output ({len(output_df)})"
    )

def score_to_category(score: float) -> str:
    if score >= 0.6:
        return "Bullish"
    if score <= -0.6:
        return "Bearish"
    return "Neutral"


def validate_columns(df: pd.DataFrame, required_cols, file_path: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")


def build_model(train_csv_path: str):
    train_df = pd.read_csv(train_csv_path)
    validate_columns(train_df, REQUIRED_TRAIN_COLS, train_csv_path)

    train_df["headline"] = train_df["headline"].fillna("").astype(str)
    train_df["summary"] = train_df["summary"].fillna("").astype(str)
    train_df["directional_score"] = pd.to_numeric(
        train_df["directional_score"], errors="coerce"
    )

    train_df = train_df[train_df["directional_score"].notna()].copy()
    if train_df.empty:
        raise ValueError("No valid labeled rows found in training file.")

    text_train = (train_df["headline"] + " " + train_df["summary"]).tolist()
    y_train = train_df["directional_score"].astype(float)

    embedder = SentenceTransformer("ProsusAI/finbert")
    x_train = embedder.encode(text_train, show_progress_bar=True)

    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)

    return model, embedder


def predict_for_file(
    model,
    embedder,
    input_csv_path: str,
    output_csv_path: str,
    train_csv_path: str = None,
):
    # 1) Read processed data (defines the full row universe)
    df = pd.read_csv(input_csv_path)
    validate_columns(df, REQUIRED_PREDICT_COLS, input_csv_path)

    df["headline"] = df["headline"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["date"] = df["date"].astype(str)

    text = (df["headline"] + " " + df["summary"]).tolist()

    # 2) Predict one score per row using embeddings
    X = embedder.encode(text, show_progress_bar=True)
    predicted_scores = model.predict(X)

    result_df = pd.DataFrame({
        "date": df["date"].values,
        "headline": df["headline"].values,
        "directional_score": predicted_scores,
    })

    # Clip directional_score between -1 and 1
    result_df["directional_score"] = np.clip(result_df["directional_score"], -1.0, 1.0)

    # 3) Override predictions with training scores where available
    if train_csv_path and os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)
        train_df["headline"] = train_df["headline"].fillna("").astype(str)
        train_df["date"] = train_df["date"].astype(str)

        train_lookup = (
            train_df[["date", "headline", "directional_score"]]
            .dropna(subset=["directional_score"])
            .drop_duplicates(subset=["date", "headline"])
            .set_index(["date", "headline"])
        )

        # Merge training scores back safely (NO MultiIndex)
        result_df = result_df.merge(
            train_df[["date", "headline", "directional_score"]]
                .dropna(subset=["directional_score"])
                .drop_duplicates(subset=["date", "headline"]),
            on=["date", "headline"],
            how="left",
            suffixes=("", "_train")
        )

        # Prefer training score when available
        result_df["directional_score"] = result_df["directional_score_train"].combine_first(
            result_df["directional_score"]
        )

        result_df.drop(columns=["directional_score_train"], inplace=True)

        # Clip again after merging
        result_df["directional_score"] = np.clip(result_df["directional_score"], -1.0, 1.0)

    # 4) Convert scores to categories
    result_df["category"] = result_df["directional_score"].apply(score_to_category)

    # 5) Final output (row count preserved)

    final_df = result_df[["date", "category", "directional_score"]]
    # Check row count before exporting
    if len(final_df) != len(df):
        raise AssertionError(f"Row count mismatch: input ({len(df)}) vs output ({len(final_df)})")

    # Export to a temporary file first, then check row count, then move to output
    temp_output_path = output_csv_path + ".tmp_check"
    final_df.to_csv(temp_output_path, index=False)
    check_output_row_count(input_csv_path, temp_output_path)
    os.replace(temp_output_path, output_csv_path)



def run_single_ticker(processed_dir, outputs_dir, train_dir, ticker):
    ticker = ticker.upper()
    processed_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
    train_file = os.path.join(train_dir, f"{ticker}_train.csv")
    output_file = os.path.join(outputs_dir, f"{ticker}_result.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed file not found: {processed_file}")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")

    os.makedirs(outputs_dir, exist_ok=True)

    model, embedder = build_model(train_file)
    predict_for_file(
        model,
        embedder,
        processed_file,
        output_file,
        train_csv_path=train_file,
    )
    print(f"Created {output_file}")


def main():

    parser = argparse.ArgumentParser(
        description="Train ML model and predict directional sentiment for a single ticker."
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol to process (e.g., ABT)",
    )

    args = parser.parse_args()

    processed_dir = "DataSets/Processed_Datasets_Transformer"
    outputs_dir = "Sentiment_Score/Directional_Score/Directional_Result"
    train_dir = "Sentiment_Score/Directional_Score/Directional_Train"

    run_single_ticker(
        processed_dir,
        outputs_dir,
        train_dir,
        args.ticker,
    )


if __name__ == "__main__":
    main()