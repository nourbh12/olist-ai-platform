from pathlib import Path

import joblib
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "delivery_risk"
    / "delivery_risk_model.joblib"
)

THRESHOLD = 0.07

CORE_FEATURES = [
    "order_item_count",
    "order_total_price",
    "order_total_freight",
    "purchase_hour",
    "purchase_day_of_week",
    "estimated_delivery_duration_days",
]


class DeliveryRiskService:

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Delivery risk model not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

    def predict(self, features: dict) -> dict:

        input_df = pd.DataFrame(
            [features],
            columns=CORE_FEATURES,
        )

        risk_score = float(
            self.model.predict_proba(input_df)[0, 1]
        )

        prediction = int(risk_score >= THRESHOLD)

        if prediction == 1:
            risk_level = "HIGH"
        else:
            risk_level = "LOW"

        return {
            "risk_score": round(risk_score, 4),
            "threshold": THRESHOLD,
            "prediction": prediction,
            "risk_level": risk_level,
        }


delivery_risk_service = DeliveryRiskService()