from pathlib import Path
from typing import Any
from api.exceptions import (
    AnalyticsDataNotFoundError,
    AnalyticsQueryError,
) 

import duckdb



PROJECT_ROOT = Path(__file__).resolve().parents[2]

SQL_DIR = PROJECT_ROOT / "pipelines" / "analytics" / "sql"


class AnalyticsService:
    """
    Executes analytical SQL queries directly against Gold Parquet data
    using DuckDB.
    """

    QUERY_CONFIG = {
        "customers": {
            "sql_file": "customers.sql",
            "queries": {
                "overall_customer_kpis": 1,
                "customer_performance_by_state": 2,
                "customers_by_number_of_orders": 3,
                "monthly_customer_activity": 4,
            },
        },
        "delivery": {
            "sql_file": "delivery.sql",
            "queries": {
                "overall_delivery_kpis": 1,
                "delivery_performance_by_month": 2,
                "late_deliveries_by_seller": 3,
                "delivery_performance_by_order_status": 4,
                "late_delivery_by_purchase_day": 5,
            },
        },
        "products": {
            "sql_file": "products.sql",
            "queries": {
                "overall_product_kpis": 1,
                "top_products_by_revenue": 2,
                "revenue_by_product_category": 3,
                "freight_performance_by_category": 4,
            },
        },
        "reviews": {
            "sql_file": "review.sql",
            "queries": {
                "overall_review_kpis": 1,
                "review_score_distribution": 2,
                "review_score_vs_delivery_performance": 3,
                "reviews_by_delivery_status": 4,
                "review_score_by_month": 5,
            },
        },
        "sales": {
            "sql_file": "sales.sql",
            "queries": {
                "overall_sales_kpis": 1,
                "monthly_sales_performance": 2,
                "sales_by_order_status": 3,
                "late_orders_impact_on_sales": 4,
            },
        },
        "sellers": {
            "sql_file": "sellers.sql",
            "queries": {
                "overall_seller_kpis": 1,
                "top_sellers_by_orders": 2,
                "worst_sellers_by_late_rate": 3,
                "seller_performance_distribution": 4,
            },
        },
    }

    def __init__(self):
        self.connection = duckdb.connect(database=":memory:")

    def _load_queries(self, sql_file: str) -> list[str]:
        """
        Load SQL statements from a SQL file.

        The existing SQL files contain multiple SELECT statements
        separated by semicolons.
        """

        file_path = SQL_DIR / sql_file

        if not file_path.exists():
            raise FileNotFoundError(
                f"Analytics SQL file not found: {file_path}"
            )

        sql_content = file_path.read_text(encoding="utf-8")

        queries = [
            query.strip()
            for query in sql_content.split(";")
            if query.strip()
        ]

        return queries

    def _execute_query(
        self,
        sql_file: str,
        query_number: int,
    ) -> list[dict[str, Any]]:

        try:
            queries = self._load_queries(sql_file)

            if query_number > len(queries):
                raise AnalyticsQueryError(
                    f"Query {query_number} does not exist in {sql_file}."
                )

            query = queries[query_number - 1]

            result = self.connection.execute(query).fetchdf()

            result = result.astype(object)
            result = result.where(result.notna(), None)

            return result.to_dict(orient="records")

        except FileNotFoundError as exc:
            raise AnalyticsDataNotFoundError(
                f"Analytics SQL file is missing: {sql_file}"
            ) from exc

        except AnalyticsQueryError:
            raise

        except Exception as exc:
            raise AnalyticsQueryError(
                f"Failed to execute analytics query from {sql_file}."
            ) from exc

    def get_domain(self, domain: str) -> dict[str, Any]:
        """
        Execute all analytics queries for one domain.
        """

        if domain not in self.QUERY_CONFIG:
            raise ValueError(f"Unknown analytics domain: {domain}")

        config = self.QUERY_CONFIG[domain]

        datasets = {}

        for dataset_name, query_number in config["queries"].items():
            datasets[dataset_name] = self._execute_query(
                sql_file=config["sql_file"],
                query_number=query_number,
            )

        return {
            "domain": domain,
            "datasets": datasets,
        }


analytics_service = AnalyticsService()