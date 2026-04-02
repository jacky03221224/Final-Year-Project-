"""
Usage: python Sentiment_Score\Directional_Score\ml.py --ticker ABT
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import numpy as np
import torch

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)


# =========================
# Required Columns
# =========================
REQUIRED_TRAIN_COLS = [
    "date",
    "headline",
    "source",
    "summary",
    "directional_sentiment",
]

REQUIRED_PREDICT_COLS = [
    "date",
    "headline",
    "source",
    "summary",
]


def validate_columns(df: pd.DataFrame, required_cols, file_path: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")


# =========================
# Build + Train FinBERT
# =========================
def build_model(train_csv_path: str):

    # =========================
    # Load data
    # =========================
    df = pd.read_csv(train_csv_path)
    validate_columns(df, REQUIRED_TRAIN_COLS, train_csv_path)

    df["headline"] = df["headline"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)

    df = df[df["directional_sentiment"].notna()].copy()
    if df.empty:
        raise ValueError("No valid labeled rows found.")

    # =========================
    # Combine text
    # =========================
    df["text"] = df["headline"] + " " + df["summary"]

    # =========================
    # Label mapping (-1,0,1 → 0,1,2)
    # =========================
    label_map = {
        -1: 0,
         0: 1,
         1: 2
    }

    df["labels"] = df["directional_sentiment"].map(label_map)

    if df["labels"].isna().any():
        raise ValueError("Found invalid sentiment labels in dataset")

    df["labels"] = df["labels"].astype(int)

    dataset = Dataset.from_pandas(df[["text", "labels"]])

    # =========================
    # Load FinBERT
    # =========================
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    # =========================
    # Tokenization
    # =========================
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding=True,
            max_length=128
        )

    dataset = dataset.map(tokenize_function, batched=True)

    # remove raw text column (important for Trainer stability)
    dataset = dataset.remove_columns(["text"])
    dataset = dataset.train_test_split(test_size=0.1)

    dataset["train"].set_format("torch")
    dataset["test"].set_format("torch")

    # =========================
    # Data collator (important for padding)
    # =========================
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # =========================
    # Training arguments
    # =========================
    training_args = TrainingArguments(
        output_dir="./finbert_model",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        save_strategy="no",
        report_to="none"
    )

    # =========================
    # Trainer
    # =========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    return model, tokenizer


# =========================
# Prediction
# =========================
def predict_for_file(
    model,
    tokenizer,
    input_csv_path: str,
):

    import torch
    import pandas as pd

    df = pd.read_csv(input_csv_path)
    validate_columns(df, REQUIRED_PREDICT_COLS, input_csv_path)

    df["headline"] = df["headline"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["date"] = df["date"].astype(str)

    texts = (df["headline"] + " " + df["summary"]).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_size = 16
    preds = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):

            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)

    reverse_map = {0: -1, 1: 0, 2: 1}
    preds = [reverse_map[p] for p in preds]

    result_df = pd.DataFrame({
        "date": df["date"],
        "headline": df["headline"],
        "summary": df["summary"],
        "directional_sentiment_pred": preds
    })

    return result_df

# =========================
# Run
# =========================
def run_single_ticker(processed_dir, outputs_dir, train_dir, ticker):

    ticker = ticker.upper()

    processed_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
    train_file = os.path.join(train_dir, f"{ticker}_train.csv")

    final_output_file = os.path.join(outputs_dir, f"{ticker}_result.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Missing processed file: {processed_file}")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Missing train file: {train_file}")

    os.makedirs(outputs_dir, exist_ok=True)

    print(f"Training FinBERT for {ticker}...")
    model, tokenizer = build_model(train_file)

    print(f"Predicting for {ticker}...")

    # ✅ DIRECT DATAFRAME (no CSV write/read)
    result_df = predict_for_file(
        model,
        tokenizer,
        processed_file,
    )

    # =========================
    # Combine predictions + true labels
    # =========================
    print("Combining predicted results with training labels...")

    train_df = pd.read_csv(train_file)[
        ["date", "headline", "summary", "directional_sentiment"]
    ].copy()

    # normalize
    for df in [result_df, train_df]:
        df["date"] = df["date"].astype(str)
        df["headline"] = df["headline"].fillna("").astype(str)
        df["summary"] = df["summary"].fillna("").astype(str)

    train_df = train_df.rename(columns={
        "directional_sentiment": "directional_sentiment_true"
    })

    merged = result_df.merge(
        train_df,
        on=["date", "headline", "summary"],
        how="left"
    )

    # =========================
    # FINAL LABEL (true overrides pred)
    # =========================
    merged["directional_sentiment_final"] = merged[
        "directional_sentiment_true"
    ].combine_first(merged["directional_sentiment_pred"])


    merged = merged[[
        "date",
        "directional_sentiment_final"
    ]].rename(columns={
        "directional_sentiment_final": "directional_sentiment"
    })

    # Ensure result is integer, not float
    merged["directional_sentiment"] = merged["directional_sentiment"].astype(int)

    # Ensure row count matches processed file
    processed_df = pd.read_csv(processed_file)
    if len(merged) != len(processed_df):
        raise ValueError(f"Row count mismatch: result file has {len(merged)} rows, but processed file has {len(processed_df)} rows.")

    merged.to_csv(final_output_file, index=False)

    print(f"Saved final results to {final_output_file}")


# =========================
# Main
# =========================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)

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