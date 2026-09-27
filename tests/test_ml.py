from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_delivery_risk_prediction():

    payload = {
        "order_item_count": 2,
        "order_total_price": 150.0,
        "order_total_freight": 25.0,
        "purchase_hour": 14,
        "purchase_day_of_week": 2,
        "estimated_delivery_duration_days": 20,
    }

    response = client.post(
        "/api/v1/ml/delivery-risk",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "risk_score" in data
    assert "threshold" in data
    assert "prediction" in data
    assert "risk_level" in data

    assert 0 <= data["risk_score"] <= 1
    assert data["threshold"] == 0.07
    assert data["prediction"] in [0, 1]
    assert data["risk_level"] in ["LOW", "HIGH"]