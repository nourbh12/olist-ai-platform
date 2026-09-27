import pytest

from frontend import api_client

from tests.test_frontend.fixtures import (
    ANALYTICS_EXAMPLE_RESPONSE,
    DELIVERY_RISK_RESPONSE,
    HEALTH_RESPONSE,
)


class MockResponse:
    """Simple HTTP response mock for frontend API tests."""

    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(
                f"HTTP {self.status_code}"
            )


def test_get_health(monkeypatch):
    captured = {}

    def mock_get(url, timeout):
        captured["url"] = url
        captured["timeout"] = timeout

        return MockResponse(HEALTH_RESPONSE)

    monkeypatch.setattr(
        api_client.requests,
        "get",
        mock_get,
    )

    result = api_client.get_health()

    assert result == HEALTH_RESPONSE

    assert captured["url"] == (
        f"{api_client.API_BASE_URL}/health"
    )

    assert captured["timeout"] == 10


def test_get_analytics(monkeypatch):
    captured = {}

    def mock_get(url, timeout):
        captured["url"] = url
        captured["timeout"] = timeout

        return MockResponse(
            ANALYTICS_EXAMPLE_RESPONSE
        )

    monkeypatch.setattr(
        api_client.requests,
        "get",
        mock_get,
    )

    endpoint = "/api/v1/analytics/example"

    result = api_client.get_analytics(endpoint)

    assert result == ANALYTICS_EXAMPLE_RESPONSE

    assert captured["url"] == (
        f"{api_client.API_BASE_URL}{endpoint}"
    )

    assert captured["timeout"] == 30


def test_predict_delivery_risk(monkeypatch):
    payload = {
        "order_item_count": 2,
        "order_total_price": 150.0,
        "order_total_freight": 25.0,
        "purchase_hour": 14,
        "purchase_day_of_week": 2,
        "estimated_delivery_duration_days": 20,
    }

    captured = {}

    def mock_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout

        return MockResponse(
            DELIVERY_RISK_RESPONSE
        )

    monkeypatch.setattr(
        api_client.requests,
        "post",
        mock_post,
    )

    result = api_client.predict_delivery_risk(
        payload
    )

    assert result == DELIVERY_RISK_RESPONSE

    assert captured["url"] == (
        f"{api_client.API_BASE_URL}"
        "/api/v1/ml/delivery-risk"
    )

    assert captured["json"] == payload
    assert captured["timeout"] == 30


@pytest.mark.parametrize(
    "function_name",
    [
        "get_health",
        "get_analytics",
    ],
)
def test_get_requests_raise_on_http_error(
    monkeypatch,
    function_name,
):
    def mock_get(url, timeout):
        return MockResponse(
            {"detail": "Server error"},
            status_code=500,
        )

    monkeypatch.setattr(
        api_client.requests,
        "get",
        mock_get,
    )

    if function_name == "get_health":
        with pytest.raises(Exception):
            api_client.get_health()

    else:
        with pytest.raises(Exception):
            api_client.get_analytics(
                "/api/v1/analytics/example"
            )


def test_predict_delivery_risk_raises_on_http_error(
    monkeypatch,
):
    def mock_post(url, json, timeout):
        return MockResponse(
            {"detail": "Server error"},
            status_code=500,
        )

    monkeypatch.setattr(
        api_client.requests,
        "post",
        mock_post,
    )

    with pytest.raises(Exception):
        api_client.predict_delivery_risk({})