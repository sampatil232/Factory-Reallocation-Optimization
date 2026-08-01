"""
5clustering.py

Cluster shipping routes based on their performance.
"""

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.utils import (
    DATA_DIR,
    CLUSTER_FEATURES
)


# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = DATA_DIR / "ml_ready_data1.csv"

OUTPUT_PATH = DATA_DIR / "route_clusters.csv"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset Shape :", df.shape)


# --------------------------------------------------
# Features for Clustering
# --------------------------------------------------

X = df[CLUSTER_FEATURES]

# --------------------------------------------------
# Scale Features
# --------------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])

# --------------------------------------------------
# KMeans Clustering
# --------------------------------------------------

kmeans = KMeans(

    n_clusters=3,

    random_state=42,

    n_init=10

)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# --------------------------------------------------
# Cluster Summary
# --------------------------------------------------

cluster_summary = (
    df.groupby("Cluster")["Lead Time"]
      .mean()
      .sort_values()
)

cluster_order = cluster_summary.index.tolist()

cluster_map = {
    cluster_order[0]: "Fast Route",
    cluster_order[1]: "Average Route",
    cluster_order[2]: "Slow Route"
}

df["Cluster Label"] = df["Cluster"].map(cluster_map)

print(df[["Cluster", "Cluster Label"]].drop_duplicates())

print("\n========== Cluster Summary ==========\n")

summary = df.groupby("Cluster Label").agg({

    "Lead Time": "mean",

    "Shipping Distance (km)": "mean",

    "Gross Profit": "mean",

    "Sales": "mean",

    "Units": "mean"

}).round(2)

print(summary)

# --------------------------------------------------
# Save Output
# --------------------------------------------------

df.to_csv(

    OUTPUT_PATH,

    index=False

)

print("\nClustered data saved successfully.")

