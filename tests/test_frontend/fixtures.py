# ============================================================
# API RESPONSES
# ============================================================

HEALTH_RESPONSE = {
    "status": "ok",
    "service": "test-service",
}


DELIVERY_RISK_RESPONSE = {
    "risk_score": 0.73,
    "threshold": 0.50,
    "prediction": 1,
    "risk_level": "HIGH",
}


ANALYTICS_EXAMPLE_RESPONSE = {
    "datasets": {
        "example_dataset": [
            {
                "metric": "example",
                "value": 123,
            }
        ]
    }
}


# ============================================================
# ANALYTICS DATASETS
# ============================================================

SALES_DATASETS = {
    "overall_sales_kpis": [
        {
            "total_orders": 100,
            "total_revenue": 1000.0,
            "average_order_value": 10.0,
            "maximum_order_value": 100.0,
        }
    ],
    "monthly_sales_performance": [
        {
            "month": "2024-01",
            "orders": 10,
            "revenue": 100.0,
            "average_order_value": 10.0,
        }
    ],
    "sales_by_order_status": [
        {
            "order_status": "delivered",
            "total_orders": 10,
            "total_revenue": 100.0,
            "average_order_value": 10.0,
        }
    ],
    "late_orders_impact_on_sales": [
        {
            "is_late": True,
            "total_orders": 1,
            "total_revenue": 10.0,
            "average_order_value": 10.0,
        }
    ],
}


PRODUCTS_DATASETS = {
    "overall_product_kpis": [
        {
            "total_products": 10,
            "average_product_price": 20.0,
            "total_product_revenue": 200.0,
            "total_freight_revenue": 30.0,
        }
    ],
    "revenue_by_product_category": [
        {
            "category": "category_a",
            "orders": 10,
            "products": 5,
            "revenue": 100.0,
            "average_price": 20.0,
        }
    ],
    "top_products_by_revenue": [
        {
            "product_id": "product_a",
            "orders": 5,
            "revenue": 100.0,
            "average_price": 20.0,
        }
    ],
}


CUSTOMERS_DATASETS = {
    "overall_customer_kpis": [
        {
            "total_customers": 10,
            "total_orders": 20,
            "orders_per_customer": 2.0,
            "revenue_per_customer": 50.0,
        }
    ],
    "customer_performance_by_state": [
        {
            "customer_state": "XX",
            "customers": 10,
            "orders": 20,
            "revenue": 500.0,
            "average_order_value": 25.0,
        }
    ],
    "monthly_customer_activity": [
        {
            "month": "2024-01",
            "active_customers": 10,
            "orders": 20,
            "revenue": 500.0,
        }
    ],
}


DELIVERY_DATASETS = {
    "overall_delivery_kpis": [
        {
            "total_orders": 20,
            "avg_estimated_delivery_days": 10.0,
            "avg_actual_delivery_days": 9.0,
            "avg_delivery_delay_days": 1.0,
            "late_delivery_rate": 5.0,
        }
    ],
    "delivery_performance_by_month": [
        {
            "month": "2024-01",
            "orders": 20,
            "avg_delivery_days": 9.0,
            "avg_delay_days": 1.0,
            "late_rate_percentage": 5.0,
        }
    ],
    "late_deliveries_by_seller": [
        {
            "seller_id": "seller_a",
            "orders": 20,
            "late_orders": 1,
            "late_rate_percentage": 5.0,
            "avg_delivery_delay_days": 1.0,
        }
    ],
}


REVIEWS_DATASETS = {
    "overall_review_kpis": [
        {
            "total_reviews": 20,
            "average_review_score": 4.0,
        }
    ],
    "review_score_distribution": [
        {
            "review_score": 5,
            "reviews": 10,
            "percentage": 50.0,
        }
    ],
    "review_score_vs_delivery_performance": [
        {
            "review_score": 5,
            "reviews": 10,
            "average_delivery_delay": 1.0,
            "late_delivery_rate": 5.0,
        }
    ],
}


SELLERS_DATASETS = {
    "overall_seller_kpis": [
        {
            "total_sellers": 5,
            "total_orders": 20,
        }
    ],
    "top_sellers_by_orders": [
        {
            "seller_id": "seller_a",
            "total_orders": 10,
            "late_orders": 1,
            "late_rate_percentage": 10.0,
            "average_delivery_delay_days": 1.0,
        }
    ],
    "worst_sellers_by_late_rate": [
        {
            "seller_id": "seller_b",
            "total_orders": 5,
            "late_orders": 2,
            "late_rate_percentage": 40.0,
            "average_delivery_delay_days": 2.0,
        }
    ],
}