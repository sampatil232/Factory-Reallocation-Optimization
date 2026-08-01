"""
prediction.py

Predict shipping lead time using the trained Gradient Boosting model.
"""


import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)


from src.utils.utils import (
    load_model,
    load_encoders
)

# ==========================================================
# Load Model and Encoders using utils
# ==========================================================

model = load_model()

encoders = load_encoders()

product_encoder = encoders["product"]
factory_encoder = encoders["factory"]
ship_mode_encoder = encoders["ship_mode"]
division_encoder = encoders["division"]
region_encoder = encoders["region"]

# ==========================================================
# Prediction Function
# ==========================================================

def predict_lead_time(
    product_name,
    factory,
    ship_mode,
    division,
    region,
    sales_scaled,
    units_scaled,
    cost_scaled,
    gross_profit_scaled,
    shipping_distance_scaled,
):
    """
    Predict lead time.

    Parameters
    ----------
    product_name : str
    factory : str
    ship_mode : str
    division : str
    region : str
    sales_scaled : float
    units_scaled : float
    cost_scaled : float
    gross_profit_scaled : float
    shipping_distance_scaled : float

    Returns
    -------
    float
        Predicted Lead Time
    """

    product = product_encoder.transform([product_name])[0]

    factory = factory_encoder.transform([factory])[0]

    ship_mode = ship_mode_encoder.transform([ship_mode])[0]

    division = division_encoder.transform([division])[0]

    region = region_encoder.transform([region])[0]

    sample = pd.DataFrame({
        "Product Name": [product],
        "Factory": [factory],
        "Ship Mode": [ship_mode],
        "Division": [division],
        "Region": [region],
        "Sales Scaled": [sales_scaled],
        "Units Scaled": [units_scaled],
        "Cost Scaled": [cost_scaled],
        "Gross Profit Scaled": [gross_profit_scaled],
        "Shipping Distance Scaled": [shipping_distance_scaled]
    })

    prediction = model.predict(sample)[0]

    return round(float(prediction), 2)


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    lead_time = predict_lead_time(

        product_name="Laffy Taffy",

        factory="Sugar Shack",

        ship_mode="Standard Class",

        division="Sugar",

        region="Atlantic",

        sales_scaled=0.52,

        units_scaled=-0.41,

        cost_scaled=0.38,

        gross_profit_scaled=0.61,

        shipping_distance_scaled=-0.32

    )

    print("=" * 40)
    print("Predicted Lead Time")
    print("=" * 40)
    print(f"{lead_time:.2f} days")