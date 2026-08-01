"""
what_if.py
-----------------------------------------
What-If Analysis Page

Responsibilities
----------------
• Load ML dataset
• Product Selector
• Factory Selector
• Predict Lead Time
• Display Recommendation
"""

import streamlit as st
import pandas as pd
import joblib

from helper import (
    load_ml_data,
    load_encoders,
    calculate_distance,
    load_distance_scaler,
    REGION_COORDINATES
)


# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = load_ml_data()

encoders = load_encoders()
distance_scaler = load_distance_scaler()
# -------------------------------------------------------
# Load Trained Model
# -------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/gradient_boosting.pkl")

model = load_model()

# -------------------------------------------------------
# Page Title
# -------------------------------------------------------

st.title("🔄 What-If Factory Simulation")

st.markdown("""
Simulate factory reallocation and predict the expected shipping lead time before implementation.
""")

st.divider()

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.header("Simulation")

# Product

product = st.sidebar.selectbox(
    "Product",
    sorted(df["Product Name"].unique())
)

# Current Factory

current_factory = st.sidebar.selectbox(
    "Current Factory",
    sorted(df["Factory"].unique())
)

# New Factory

new_factory = st.sidebar.selectbox(
    "Suggested Factory",
    sorted(df["Factory"].unique())
)

# Ship Mode

ship_mode = st.sidebar.selectbox(
    "Ship Mode",
    sorted(df["Ship Mode"].unique())
)

# Region

region = st.sidebar.selectbox(
    "Destination Region",
    sorted(df["Region"].unique())
)

division = st.sidebar.selectbox(
    "Division",
    sorted(df["Division"].unique())
)

# -------------------------------------------------------
# Display Current Selection
# -------------------------------------------------------

st.subheader("Simulation Inputs")

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
**Product**

{product}

**Current Factory**

{current_factory}
""")

with col2:
    st.info(f"""
**Suggested Factory**

{new_factory}

**Ship Mode**

{ship_mode}
""")

st.divider()

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

if st.button("🚀 Run Simulation", width="stretch"):
    # Find a matching record
    row = df[
        (df["Product Name"] == product) &
        (df["Factory"] == current_factory) &
        (df["Ship Mode"] == ship_mode) &
        (df["Division"] == division) &
        (df["Region"] == region)
    ]

    # If no exact match exists, use the first row for that product
    if row.empty:
        row = df[df["Product Name"] == product]

    row = row.iloc[0]
    # Customer coordinates from selected region
    customer_lat, customer_lon = REGION_COORDINATES[region]

# Distance from NEW factory
    distance = calculate_distance(
        new_factory,
        customer_lat,
        customer_lon
    )

    scaled_distance = distance_scaler.transform(
    pd.DataFrame({
        "Shipping Distance (km)": [distance]
    })
    )[0][0]

    input_df = pd.DataFrame({
        "Product Name": [product],
        "Factory": [new_factory],
        "Ship Mode": [ship_mode],
        "Division": [division],
        "Region": [region],

        "Sales Scaled": [row["Sales Scaled"]],
        "Units Scaled": [row["Units Scaled"]],
        "Cost Scaled": [row["Cost Scaled"]],
        "Gross Profit Scaled": [row["Gross Profit Scaled"]],
        "Shipping Distance Scaled": [scaled_distance]
    })

    # Encode categorical columns
    input_df["Product Name"] = encoders["product"].transform(input_df["Product Name"])
    input_df["Factory"] = encoders["factory"].transform(input_df["Factory"])
    input_df["Ship Mode"] = encoders["ship_mode"].transform(input_df["Ship Mode"])
    input_df["Division"] = encoders["division"].transform(input_df["Division"])
    input_df["Region"] = encoders["region"].transform(input_df["Region"])

    # Arrange columns exactly as during training
    input_df = input_df[
        [
            "Product Name",
            "Factory",
            "Ship Mode",
            "Division",
            "Region",
            "Sales Scaled",
            "Units Scaled",
            "Cost Scaled",
            "Gross Profit Scaled",
            "Shipping Distance Scaled",
        ]
    ]


    try:

        prediction = model.predict(input_df)[0]

        st.success("Prediction completed successfully.")

        st.metric(
            "Predicted Lead Time",
            f"{prediction:.2f} Days"
        )

        # ------------------------------------------------

        if current_factory != new_factory:

            st.success(f"""
### Recommendation

Move production from

**{current_factory}**

➡

**{new_factory}**

Estimated Lead Time

**{prediction:.2f} Days**
""")

        else:

            st.info("""
The selected factory is already the current factory.

No reallocation is required.
""")

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)

st.divider()

# -------------------------------------------------------
# Additional Information
# -------------------------------------------------------

st.subheader("Simulation Summary")

st.markdown("""
The What-If Analysis estimates the shipping lead time after changing the manufacturing factory.

The prediction is generated using the trained machine learning model and helps logistics planners evaluate alternative factory assignments before implementing operational changes.
""")