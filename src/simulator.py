"""
7simulator.py

What-If Factory Reallocation Simulator
"""

import pandas as pd

from pathlib import Path

from src.utils.utils import (
    load_model,
    load_encoders,
    load_distance_scaler,
    calculate_distance,
    FEATURE_COLUMNS,
    FACTORY_COORDINATES,
    DATA_DIR
)


# =====================================================
# Paths
# =====================================================

DATA_PATH = DATA_DIR / "ml_ready_data1.csv"


# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(DATA_PATH)


# =====================================================
# Load Model
# =====================================================

model = load_model()


# =====================================================
# Load Encoders
# =====================================================

encoders = load_encoders()

product_encoder = encoders["product"]

factory_encoder = encoders["factory"]

distance_scaler = load_distance_scaler()



# =====================================================
# Scenario Simulation Function
# =====================================================

def simulate_factory(product_name, new_factory):

    # Encode product
    product_code = product_encoder.transform(
        [product_name]
    )[0]

    row = df[df["Product Name"] == product_code].iloc[0]

    current_factory = factory_encoder.inverse_transform(
        [int(row["Factory"])]
    )[0]

    current_sample = row[FEATURE_COLUMNS].copy().to_frame().T

    current_prediction = model.predict(
        current_sample
    )[0]

    distance = calculate_distance(
        new_factory,
        row["Customer Latitude"],
        row["Customer Longitude"]
)

    scaled_distance = distance_scaler.transform(

        pd.DataFrame({

            "Shipping Distance (km)": [distance]

        })

    )[0][0]

    new_sample = current_sample.copy()

    new_sample["Factory"] = factory_encoder.transform(
        [new_factory]
    )[0]

    print("Model expects:", model.feature_names_in_)
    print("Input columns:", new_sample.columns.tolist())
    
    new_prediction = model.predict(
        new_sample
    )[0]

    

    return {
        "Current Factory":current_factory,
        "Current Lead Time":current_prediction,
        "New Factory":new_factory,
        "Predicted Lead Time":new_prediction,
        "Lead Time Saved":
            current_prediction-new_prediction
    }   
# =====================================================
# Example
# =====================================================

if __name__ == "__main__":

    simulate_factory(

        product_name="Laffy Taffy",

        new_factory="Secret Factory"

    )