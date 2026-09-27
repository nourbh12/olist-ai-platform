import pandas as pd

from frontend.views import analytics

from tests.test_frontend.fixtures import (
    CUSTOMERS_DATASETS,
    DELIVERY_DATASETS,
    PRODUCTS_DATASETS,
    REVIEWS_DATASETS,
    SALES_DATASETS,
    SELLERS_DATASETS,
)


# ============================================================
# ANALYTICS HELPERS
# ============================================================

def test_first_row_returns_first_row():
    dataset = [
        {
            "metric": "first",
            "value": 10,
        },
        {
            "metric": "second",
            "value": 20,
        },
    ]

    result = analytics.first_row(dataset)

    assert result == dataset[0]


def test_first_row_returns_empty_dict_for_empty_dataset():
    result = analytics.first_row([])

    assert result == {}


def test_to_dataframe_converts_dataset():
    dataset = [
        {
            "category": "A",
            "value": 10,
        },
        {
            "category": "B",
            "value": 20,
        },
    ]

    result = analytics.to_dataframe(dataset)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)

    assert list(result.columns) == [
        "category",
        "value",
    ]


def test_to_dataframe_returns_empty_dataframe():
    result = analytics.to_dataframe([])

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_format_number():
    assert analytics.format_number(
        1234567
    ) == "1,234,567"


def test_format_number_handles_none():
    assert analytics.format_number(None) == "—"


def test_format_currency():
    assert analytics.format_currency(
        1234.56
    ) == "€1,234.56"


def test_format_currency_handles_none():
    assert analytics.format_currency(None) == "—"


# ============================================================
# DATASET RETRIEVAL
# ============================================================

def test_get_datasets_returns_datasets(monkeypatch):
    response = {
        "datasets": SALES_DATASETS,
    }

    captured = {}

    def mock_get_analytics(endpoint):
        captured["endpoint"] = endpoint

        return response

    monkeypatch.setattr(
        analytics,
        "get_analytics",
        mock_get_analytics,
    )

    result = analytics.get_datasets("sales")

    assert result == SALES_DATASETS

    assert captured["endpoint"] == (
        "/api/v1/analytics/sales"
    )


def test_get_datasets_returns_empty_dict_when_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        analytics,
        "get_analytics",
        lambda endpoint: {},
    )

    result = analytics.get_datasets("sales")

    assert result == {}


# ============================================================
# ANALYTICS PAGE RENDERING
# ============================================================

def test_sales_page_handles_api_data(monkeypatch):
    monkeypatch.setattr(
        analytics,
        "get_datasets",
        lambda endpoint: SALES_DATASETS,
    )

    analytics.show_sales()


def test_products_page_handles_api_data(monkeypatch):
    monkeypatch.setattr(
        analytics,
        "get_datasets",
        lambda endpoint: PRODUCTS_DATASETS,
    )

    analytics.show_products()


def test_customers_page_handles_api_data(monkeypatch):
    monkeypatch.setattr(
        analytics,
        "get_datasets",
        lambda endpoint: CUSTOMERS_DATASETS,
    )

    analytics.show_customers()


def test_delivery_page_handles_api_data(monkeypatch):
    monkeypatch.setattr(
        analytics,
        "get_datasets",
        lambda endpoint: DELIVERY_DATASETS,
    )

    analytics.show_delivery()


def test_reviews_page_handles_api_data(monkeypatch):
    monkeypatch.setattr(
        analytics,
        "get_datasets",
        lambda endpoint: REVIEWS_DATASETS,
    )

    analytics.show_reviews()


def test_sellers_page_handles_api_data(monkeypatch):
    monkeypatch.setattr(
        analytics,
        "get_datasets",
        lambda endpoint: SELLERS_DATASETS,
    )

    analytics.show_sellers()