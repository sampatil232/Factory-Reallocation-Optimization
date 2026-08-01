# src/preprocessing.py

import pandas as pd
import numpy as np

from pathlib import Path

def load_data(file_path):
    """
    Load the Nassau Candy Distributor dataset.
    """
    return pd.read_csv(file_path)


def fill_missing_values(df):
    """
    Fill missing values:
    - Numeric columns -> Median
    - Categorical columns -> Mode
    """

    # Numeric columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df


def convert_dates(df):
    """
    Convert date columns to datetime format.
    """

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True
    )

    return df


def create_lead_time(df):
    """
    Create Lead Time feature.
    """

    df["Lead Time"] = (
        df["Ship Date"] - df["Order Date"]
    ).dt.days

    return df


def preprocess_data(file_path):
    """
    Complete preprocessing pipeline.
    """

    # Load dataset
    df = load_data(file_path)

    print(f"Dataset Shape : {df.shape}")

    # Duplicate rows
    print(f"Duplicate Rows : {df.duplicated().sum()}")

    # Missing values
    missing = (df.isnull().sum() / len(df)) * 100
    print("\nMissing Values (%)")
    print(missing[missing > 0].sort_values(ascending=False))

    # Fill missing values
    df = fill_missing_values(df)

    # Convert dates
    df = convert_dates(df)

    # Create Lead Time
    df = create_lead_time(df)

    return df


def save_clean_data(df, output_path):
    """
    Save cleaned dataset.
    """
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    INPUT_FILE = BASE_DIR / "data" / "Nassau Candy Distributor.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "cleaned_data.csv"

    print("Input File :", INPUT_FILE)
    print("Output File:", OUTPUT_FILE)

    df = preprocess_data(INPUT_FILE)

    save_clean_data(df, OUTPUT_FILE)

    print("\nPreprocessing Completed Successfully!")
    print(df.head())