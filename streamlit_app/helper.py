"""
helper.py
Reusable utility functions for the Factory Reallocation &
Shipping Optimization Dashboard.

Contains:
1. load_display_data()
2. load_ml_data()
3. calculate_kpis()
4. decode_dataset()
5. plotly_theme()

No Streamlit UI code should be placed here.
"""

import pandas as pd
import plotly.io as pio
from pathlib import Path
import streamlit as st
import joblib
from geopy.distance import geodesic

FACTORY_COORDINATES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107)
}

REGION_COORDINATES = {
    "Interior": (39.0997, -94.5786),
    "Atlantic": (40.7128, -74.0060),
    "Gulf": (29.7604, -95.3698),
    "Pacific": (34.0522, -118.2437)
}

def load_distance_scaler():
    return joblib.load(BASE_DIR / "models" / "shipping_distance_scaler.pkl")

def calculate_distance(factory_name, customer_lat, customer_lon):
    factory_location = FACTORY_COORDINATES[factory_name]
    customer_location = (customer_lat, customer_lon)
    return geodesic(factory_location, customer_location).km
# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DISPLAY_DATA = BASE_DIR / "data" / "ml_ready_data1.csv"
ML_DATA = BASE_DIR / "data" / "ml_ready_data1.csv"


# -----------------------------------------------------
# Display Dataset
# -----------------------------------------------------

def load_display_data():
    """
    Load dataset used for dashboard visualization.
    """

    df = pd.read_csv(DISPLAY_DATA)

    # Load label encoders
    region_encoder = joblib.load(BASE_DIR / "models" / "region_encoder.pkl")
    division_encoder = joblib.load(BASE_DIR / "models" / "division_encoder.pkl")
    ship_encoder = joblib.load(BASE_DIR / "models" / "ship_mode_encoder.pkl")
    factory_encoder = joblib.load(BASE_DIR / "models" / "factory_encoder.pkl")

    # Decode categorical columns
    df["Region"] = region_encoder.inverse_transform(df["Region"].astype(int))
    df["Division"] = division_encoder.inverse_transform(df["Division"].astype(int))
    df["Ship Mode"] = ship_encoder.inverse_transform(df["Ship Mode"].astype(int))
    df["Factory"] = factory_encoder.inverse_transform(df["Factory"].astype(int))

    return df


# -----------------------------------------------------
# ML Dataset
# -----------------------------------------------------

def load_ml_data():
    """
    Load dataset used for prediction and recommendation.
    """

    df = pd.read_csv(ML_DATA)

    # Load encoders
    region_encoder = joblib.load(BASE_DIR / "models" / "region_encoder.pkl")
    division_encoder = joblib.load(BASE_DIR / "models" / "division_encoder.pkl")
    ship_encoder = joblib.load(BASE_DIR / "models" / "ship_mode_encoder.pkl")
    factory_encoder = joblib.load(BASE_DIR / "models" / "factory_encoder.pkl")
    product_encoder = joblib.load(BASE_DIR / "models" / "product_name_encoder.pkl")

    # Decode columns (only if they exist)
    if "Region" in df.columns:
        df["Region"] = region_encoder.inverse_transform(df["Region"].astype(int))

    if "Destination Region" in df.columns:
        df["Destination Region"] = region_encoder.inverse_transform(
            df["Destination Region"].astype(int)
        )

    if "Division" in df.columns:
        df["Division"] = division_encoder.inverse_transform(df["Division"].astype(int))

    if "Ship Mode" in df.columns:
        df["Ship Mode"] = ship_encoder.inverse_transform(df["Ship Mode"].astype(int))

    if "Factory" in df.columns:
        df["Factory"] = factory_encoder.inverse_transform(df["Factory"].astype(int))

    if "Current Factory" in df.columns:
        df["Current Factory"] = factory_encoder.inverse_transform(
            df["Current Factory"].astype(int)
        )

    if "Suggested Factory" in df.columns:
        df["Suggested Factory"] = factory_encoder.inverse_transform(
            df["Suggested Factory"].astype(int)
        )

    if "Recommended Factory" in df.columns:
        df["Recommended Factory"] = factory_encoder.inverse_transform(
            df["Recommended Factory"].astype(int)
        )

    if "Product Name" in df.columns:
        df["Product Name"] = product_encoder.inverse_transform(
            df["Product Name"].astype(int)
        )

    return df

def load_encoders():
    return {
        "product": joblib.load(BASE_DIR / "models" / "product_name_encoder.pkl"),
        "factory": joblib.load(BASE_DIR / "models" / "factory_encoder.pkl"),
        "ship_mode": joblib.load(BASE_DIR / "models" / "ship_mode_encoder.pkl"),
        "division": joblib.load(BASE_DIR / "models" / "division_encoder.pkl"),
        "region": joblib.load(BASE_DIR / "models" / "region_encoder.pkl"),
    }
# -----------------------------------------------------
# Dataset Decoder
# -----------------------------------------------------

def decode_dataset(df, encoders=None):
    """
    Decode encoded categorical columns.

    Parameters
    ----------
    df : pandas.DataFrame

    encoders : dict, optional
        Dictionary containing fitted LabelEncoders.

    Returns
    -------
    pandas.DataFrame
    """

    if encoders is None:
        return df

    decoded = df.copy()

    try:

        mapping = {
            "Region": "region",
            "Division": "division",
            "Ship Mode": "ship_mode",
            "Factory": "factory",
            "Product Name": "product"
        }

        for column, encoder_name in mapping.items():

            if (
                column in decoded.columns
                and encoder_name in encoders
            ):

                decoded[column] = encoders[
                    encoder_name
                ].inverse_transform(decoded[column])

    except Exception:
        pass

    return decoded


# -----------------------------------------------------
# KPI Calculation
# -----------------------------------------------------

def calculate_kpis(df):
    """
    Calculate executive KPI values.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
    """

    kpis = {

        "Shipments": len(df),

        "Products": (
            df["Product Name"].nunique()
            if "Product Name" in df.columns else 0
        ),

        "Factories": (
            df["Factory"].nunique()
            if "Factory" in df.columns else 0
        ),

        "Regions": (
            df["Region"].nunique()
            if "Region" in df.columns else 0
        ),

        "Average Lead Time": round(
            df["Lead Time"].mean(), 2
        ) if "Lead Time" in df.columns else 0,

        "Median Lead Time": round(
            df["Lead Time"].median(), 2
        ) if "Lead Time" in df.columns else 0,

        "Total Sales": round(
            df["Sales"].sum(), 2
        ) if "Sales" in df.columns else 0,

        "Gross Profit": round(
            df["Gross Profit"].sum(), 2
        ) if "Gross Profit" in df.columns else 0,

        "Average Profit": round(
            df["Gross Profit"].mean(), 2
        ) if "Gross Profit" in df.columns else 0

    }

    return kpis


# -----------------------------------------------------
# Plotly Theme
# -----------------------------------------------------

def plotly_theme():
    """
    Apply a consistent blue Plotly theme.
    """

    pio.templates["shipping_blue"] = {

        "layout": {

            "paper_bgcolor": "#F5F9FF",

            "plot_bgcolor": "#FFFFFF",

            "font": {

                "family": "Arial",

                "size": 14,

                "color": "#1E3A8A"

            },

            "colorway": [

                "#2563EB",
                "#3B82F6",
                "#60A5FA",
                "#93C5FD",
                "#BFDBFE"

            ],

            "title": {

                "font": {

                    "size": 22,

                    "color": "#1E3A8A"

                }

            }

        }

    }

    pio.templates.default = "shipping_blue"

