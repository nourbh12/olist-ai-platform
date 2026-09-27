import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


ANALYTICS_ENDPOINTS = [
    "/api/v1/analytics/customers",
    "/api/v1/analytics/delivery",
    "/api/v1/analytics/products",
    "/api/v1/analytics/reviews",
    "/api/v1/analytics/sales",
    "/api/v1/analytics/sellers",
]


@pytest.mark.parametrize("endpoint", ANALYTICS_ENDPOINTS)
def test_analytics_endpoint(endpoint):

    response = client.get(endpoint)

    assert response.status_code == 200

    data = response.json()

    assert "domain" in data
    assert "datasets" in data

    assert isinstance(data["domain"], str)
    assert isinstance(data["datasets"], dict)

    assert len(data["datasets"]) > 0