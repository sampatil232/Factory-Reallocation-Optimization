# src/train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.utils.utils import (
    FEATURE_COLUMNS,
    BASE_DIR,
    MODELS_DIR
)


# ---------------------------------------------------------
# Train Models
# ---------------------------------------------------------

def train_model(input_file):

    df = pd.read_csv(input_file)

    # -----------------------------
    # Features
    # -----------------------------

    X = df[FEATURE_COLUMNS] 

    y = df["Lead Time"]

    # -----------------------------
    # Train Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # -----------------------------
    # Models
    # -----------------------------

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            )

    }

    results = []

    best_model = None
    best_rmse = float("inf")
    best_model_name = ""

    # -----------------------------
    # Train & Evaluate
    # -----------------------------

    for name, model in models.items():

        print("=" * 50)
        print(name)
        print("=" * 50)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        print(f"RMSE : {rmse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"R2   : {r2:.4f}")

        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        if rmse < best_rmse:

            best_rmse = rmse
            best_model = model
            best_model_name = name

    # -----------------------------
    # Save Results
    # -----------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE"
    )

    

    results_df.to_csv(
    MODELS_DIR / "model_results.csv",
    index=False
)

    # Save Best Model

    joblib.dump(
    best_model,
    MODELS_DIR / "gradient_boosting.pkl"
)

    print("\n")
    print("=" * 60)
    print(f"Best Model : {best_model_name}")
    print(f"RMSE       : {best_rmse:.4f}")
    print("Model Saved Successfully!")
    print("=" * 60)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    train_model(
        BASE_DIR / "data" / "ml_ready_data1.csv"
    )