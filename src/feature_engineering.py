# src/feature_engineering.py
import pandas as pd
import numpy as np
import joblib

from src.utils.utils import (
    FACTORY_MAPPING,
    FACTORY_COORDINATES,
    REGION_COORDINATES
)
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder, StandardScaler



# ---------------------------------------------------------
# Assign Factory
# ---------------------------------------------------------

def assign_factory(df):

    df["Factory"] = df["Product Name"].map(FACTORY_MAPPING)

    return df


# ---------------------------------------------------------
# Add Coordinates
# ---------------------------------------------------------

def add_coordinates(df):

    df["Factory Latitude"] = df["Factory"].map(
        lambda x: FACTORY_COORDINATES[x][0]
    )

    df["Factory Longitude"] = df["Factory"].map(
        lambda x: FACTORY_COORDINATES[x][1]
    )

    df["Customer Latitude"] = df["Region"].map(
        lambda x: REGION_COORDINATES[x][0]
    )

    df["Customer Longitude"] = df["Region"].map(
        lambda x: REGION_COORDINATES[x][1]
    )

    return df


# ---------------------------------------------------------
# Shipping Distance
# ---------------------------------------------------------

def calculate_distance(row):

    factory = (
        row["Factory Latitude"],
        row["Factory Longitude"]
    )

    customer = (
        row["Customer Latitude"],
        row["Customer Longitude"]
    )

    return geodesic(factory, customer).km


def create_shipping_distance(df):

    df["Shipping Distance (km)"] = df.apply(
        calculate_distance,
        axis=1
    )

    distance_scaler = StandardScaler()

    df["Shipping Distance Scaled"] = distance_scaler.fit_transform(
        df[["Shipping Distance (km)"]]
    )

    # Save scaler
    joblib.dump(
        distance_scaler,
        BASE_DIR / "models" / "shipping_distance_scaler.pkl"
    )

    return df

# ---------------------------------------------------------
# Label Encoding
# ---------------------------------------------------------

def encode_features(df):

    encoders = {}

    columns = [
        "Ship Mode",
        "Factory",
        "Product Name",
        "Division",
        "Region"
    ]

    for col in columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col])

        encoders[col] = encoder

        joblib.dump(
            encoder,
            f"models/{col.lower().replace(' ','_')}_encoder.pkl"
        )

    return df, encoders


# ---------------------------------------------------------
# Lead Time (Business Rule)
# ---------------------------------------------------------

def create_lead_time(df):

    base_time = 2

    distance_days = df["Shipping Distance (km)"] / 600

    ship_mode_days = df["Ship Mode"].map({
        1: 0,
        0: 1,
        2: 2,
        3: 3
    })

    factory_delay = df["Factory"].map({
        0: 0.4,
        1: 0.8,
        2: 0.2,
        3: 0.6,
        4: 1.0
    })

    region_delay = df["Region"].map({
        0: 0.5,
        1: 0.8,
        2: 0.3,
        3: 1.0
    })

    np.random.seed(42)

    variation = np.random.uniform(0, 0.5, len(df))

    df["Lead Time"] = (
        base_time
        + distance_days
        + ship_mode_days
        + factory_delay
        + region_delay
        + variation
    ).round(1)

    return df


# ---------------------------------------------------------
# Remove Outliers
# ---------------------------------------------------------

def remove_outliers(df):

    Q1 = df["Lead Time"].quantile(0.25)
    Q3 = df["Lead Time"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[
        (df["Lead Time"] >= lower) &
        (df["Lead Time"] <= upper)
    ]

    return df


# ---------------------------------------------------------
# Scale Numerical Features
# ---------------------------------------------------------

def scale_features(df):

    columns = [
        "Sales",
        "Units",
        "Cost",
        "Gross Profit"
    ]

    for col in columns:

        scaler = StandardScaler()

        df[f"{col} Scaled"] = scaler.fit_transform(df[[col]])

        joblib.dump(
            scaler,
            BASE_DIR / "models" / f"{col.lower().replace(' ','_')}_scaler.pkl"
        )

    return df


# ---------------------------------------------------------
# Complete Pipeline
# ---------------------------------------------------------

def feature_engineering(input_file, output_file):

    df = pd.read_csv(input_file)

    df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True,
    errors="coerce",
    format="mixed"
)

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True,
        errors="coerce",
        format="mixed"
    )

    print("Invalid Order Dates:")
    print(df[df["Order Date"].isna()])

    print("Invalid Ship Dates:")
    print(df[df["Ship Date"].isna()])

    df = assign_factory(df)
    print("Rows with missing Factory:")
    print(df[df["Factory"].isna()][["Product Name", "Factory"]])

    df = add_coordinates(df)
    df = create_shipping_distance(df)
    df, encoders = encode_features(df)
    df = create_lead_time(df)
    df = remove_outliers(df)
    df = scale_features(df)

    df.to_csv(output_file, index=False)

    print("Feature Engineering Completed Successfully!")

    return df


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    feature_engineering(
        BASE_DIR / "data" / "cleaned_data.csv",
        BASE_DIR / "data" / "ml_ready_data1.csv"
    )