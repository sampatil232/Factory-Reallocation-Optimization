"""
6optimization.py

Factory Reallocation Recommendation System
"""

import pandas as pd
from pathlib import Path
import pandas as pd


from src.utils.utils import (
    load_model,
    load_encoders,
    load_distance_scaler,
    calculate_distance,
    FEATURE_COLUMNS,
    FACTORY_COORDINATES
)
# ======================================================
# Project Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "ml_ready_data1.csv"

MODEL_PATH = BASE_DIR / "models" / "gradient_boosting.pkl"

OUTPUT_PATH = BASE_DIR / "data" / "factory_recommendations.csv"


# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv(DATA_PATH)


# ======================================================
# Load Model
# ======================================================

model = load_model()


# ======================================================
# Load Encoders
# ======================================================

encoders = load_encoders()

product_encoder = encoders["product"]
factory_encoder = encoders["factory"]
ship_encoder = encoders["ship_mode"]
division_encoder = encoders["division"]
region_encoder = encoders["region"]

# ======================================================
# Load Distance Scaler
# ======================================================

distance_scaler = load_distance_scaler()



# ======================================================
# Product-Level Optimization
# ======================================================

recommendations = []

products = df.groupby("Product Name").first().reset_index()

factory_names = list(FACTORY_COORDINATES.keys())


for _, row in products.iterrows():

    current_sample = row[FEATURE_COLUMNS].copy().to_frame().T

    current_prediction = model.predict(current_sample)[0]

    current_factory = factory_encoder.inverse_transform(
        [int(row["Factory"])]
    )[0]

    best_factory = current_factory

    best_prediction = current_prediction

    

    for factory in factory_names:

        distance = calculate_distance(
            factory,
            row["Customer Latitude"],
            row["Customer Longitude"]
        )

        

        scaled_distance = distance_scaler.transform(

            pd.DataFrame({

                "Shipping Distance (km)": [distance]

            })

        )[0][0]

        sample = current_sample.copy()

        sample["Factory"] = factory_encoder.transform(

            [factory]

        )[0]

        sample["Shipping Distance Scaled"] = scaled_distance

        prediction = model.predict(sample)[0]

        if prediction < best_prediction:

            best_prediction = prediction

            best_factory = factory

    recommendations.append({

        "Product Name":

        product_encoder.inverse_transform(

            [int(row["Product Name"])]

        )[0],

        "Current Factory": current_factory,

        "Recommended Factory": best_factory,

        "Current Lead Time": round(current_prediction,2),

        "Optimized Lead Time": round(best_prediction,2),

        "Lead Time Reduction": round(

            current_prediction-best_prediction,

            2

        ),
        "State/Province": row["State/Province"],
        "Country/Region": row["Country/Region"]
        

    })



# ======================================================
# Save Recommendations
# ======================================================

recommendation_df = pd.DataFrame(recommendations)

# Sort by highest improvement
recommendation_df = recommendation_df.sort_values(
    by="Lead Time Reduction",
    ascending=False
)

# Keep only recommendations with improvement
recommendation_df = recommendation_df[
    recommendation_df["Lead Time Reduction"] > 0
]

# Add priority column
recommendation_df["Priority"] = pd.cut(
    recommendation_df["Lead Time Reduction"],
    bins=[-1, 0.5, 1.5, 10],
    labels=["Low", "Medium", "High"]
)

recommendation_df["Lead Time Reduction (%)"] = (
    recommendation_df["Lead Time Reduction"]
    / recommendation_df["Current Lead Time"]
    * 100
).round(2)

# Save to CSV
recommendation_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nOptimization Completed Successfully!")

print(recommendation_df)