"""
utils.py

Common utility functions used throughout the project.
"""

from pathlib import Path
import joblib
import pandas as pd
from geopy.distance import geodesic



# =====================================================
# Project Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODELS_DIR = BASE_DIR / "models"


# =====================================================
# Factory Coordinates
# =====================================================

FACTORY_COORDINATES = {

    "Lot's O' Nuts": (32.881893, -111.768036),

    "Wicked Choccy's": (32.076176, -81.088371),

    "Sugar Shack": (48.119140, -96.181150),

    "Secret Factory": (41.446333, -90.565487),

    "The Other Factory": (35.117500, -89.971107)

}


# =====================================================
# Features Used By Model
# =====================================================

FEATURE_COLUMNS = [
        "Product Name",
        "Factory",
        "Ship Mode",
        "Division",
        "Region",
        "Sales Scaled",
        "Units Scaled",
        "Cost Scaled",
        "Gross Profit Scaled",
        "Shipping Distance Scaled"
    ]
# ---------------------------------------------------------
# Factory Mapping
# ---------------------------------------------------------

FACTORY_MAPPING = {

    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",

    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",

    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",

    "Everlasting Gobstopper": "Secret Factory",

    "Hair Toffee": "The Other Factory",

    "Lickable Wallpaper": "Secret Factory",
    "Rainbow Drops": "The Other Factory",
    "Wonka Gum" :	"Secret Factory",
    "Kazookles": "The Other Factory"
}



REGION_COORDINATES = {

    "Interior": (39.0997, -94.5786),
    "Atlantic": (40.7128, -74.0060),
    "Gulf": (29.7604, -95.3698),
    "Pacific": (34.0522, -118.2437)
}



# =====================================================
# Load Trained Model
# =====================================================

def load_model():

    return joblib.load(
        MODELS_DIR / "randomforest.pkl"
    )


# =====================================================
# Load All Encoders
# =====================================================

def load_encoders():

    encoders = {

        "product": joblib.load(
            MODELS_DIR / "product_name_encoder.pkl"
        ),

        "factory": joblib.load(
            MODELS_DIR / "factory_encoder.pkl"
        ),

        "ship_mode": joblib.load(
            MODELS_DIR / "ship_mode_encoder.pkl"
        ),

        "division": joblib.load(
            MODELS_DIR / "division_encoder.pkl"
        ),

        "region": joblib.load(
            MODELS_DIR / "region_encoder.pkl"
        )

    }

    return encoders


# =====================================================
# Load Shipping Distance Scaler
# =====================================================

def load_distance_scaler():

    return joblib.load(
        MODELS_DIR / "shipping_distance_scaler.pkl"
    )


# =====================================================
# Shipping Distance
# =====================================================

def calculate_distance(

    factory_name,

    customer_lat,

    customer_lon

):

    factory_location = FACTORY_COORDINATES[factory_name]

    customer_location = (

        customer_lat,

        customer_lon

    )

    return geodesic(

        factory_location,

        customer_location

    ).km


# =====================================================
# Scale Shipping Distance
# =====================================================

def scale_distance(distance):

    scaler = load_distance_scaler()

    scaled = scaler.transform(

        pd.DataFrame({

            "Shipping Distance (km)": [distance]

        })

    )[0][0]

    return scaled

# =====================================================
# Clustering Features
# =====================================================

CLUSTER_FEATURES = [
    "Lead Time",
    "Shipping Distance (km)",
    "Gross Profit",
    "Sales",
    "Units"
]

# =====================================================
# Project Data Paths
# =====================================================

DATA_DIR = BASE_DIR / "data"