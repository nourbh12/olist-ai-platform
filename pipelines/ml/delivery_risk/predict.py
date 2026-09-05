import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "delivery_risk"
    / "delivery_risk_model.joblib"
)


# ============================================================
# Configuration
# ============================================================

FEATURES = [
    "order_item_count",
    "order_total_price",
    "order_total_freight",
    "purchase_hour",
    "purchase_day_of_week",
    "estimated_delivery_duration_days",
]

THRESHOLD = 0.07


# ============================================================
# Load model
# ============================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# Prediction
# ============================================================

def predict_delivery_risk(model, order: dict):

    # Convert input to DataFrame
    X = pd.DataFrame([order])

    # Make sure the feature order is correct
    X = X[FEATURES]

    # Predict probability of late delivery
    probability = model.predict_proba(X)[0, 1]

    # Apply production threshold
    prediction = int(probability >= THRESHOLD)

    risk_label = "HIGH" if prediction == 1 else "LOW"

    return probability, prediction, risk_label


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("DELIVERY RISK - INFERENCE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Example order
    # --------------------------------------------------------

    order = {
        "order_item_count": 3,
        "order_total_price": 150.0,
        "order_total_freight": 25.0,
        "purchase_hour": 18,
        "purchase_day_of_week": 5,
        "estimated_delivery_duration_days": 15,
    }

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    print("\nModel loaded successfully.")
    print(f"Model: {MODEL_PATH}")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    probability, prediction, risk_label = predict_delivery_risk(
        model,
        order
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\nORDER")
    print("-" * 60)

    for feature, value in order.items():
        print(f"{feature:<40} {value}")

    print("\nPREDICTION")
    print("-" * 60)

    print(f"Late probability : {probability:.4f}")
    print(f"Late probability : {probability * 100:.2f}%")
    print(f"Threshold        : {THRESHOLD:.2f}")
    print(f"Prediction       : {prediction}")
    print(f"Risk level       : {risk_label}")

    print("=" * 60)


if __name__ == "__main__":
    main()