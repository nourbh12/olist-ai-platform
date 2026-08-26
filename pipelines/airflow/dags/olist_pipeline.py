from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


# ============================================================
# Configuration
# ============================================================

PIPELINE_ROOT = "/opt/olist-ai-platform/pipelines"

SPARK_ROOT = f"{PIPELINE_ROOT}/spark"


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="olist_data_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["olist", "spark", "data-pipeline"],
) as dag:

    # ========================================================
    # 1. BRONZE
    # ========================================================

    bronze_ingestion = BashOperator(
        task_id="bronze_ingestion",
        bash_command=(
            f"python {SPARK_ROOT}/bronze/ingest_olist.py"
        ),
    )

    # ========================================================
    # 2. SILVER TRANSFORMATIONS
    # ========================================================

    silver_transform_customers = BashOperator(
        task_id="silver_transform_customers",
        bash_command=(
            f"python {SPARK_ROOT}/silver/transform_customers.py"
        ),
    )

    silver_transform_products = BashOperator(
        task_id="silver_transform_products",
        bash_command=(
            f"python {SPARK_ROOT}/silver/transform_products.py"
        ),
    )

    silver_transform_sellers = BashOperator(
        task_id="silver_transform_sellers",
        bash_command=(
            f"python {SPARK_ROOT}/silver/transform_sellers.py"
        ),
    )

    silver_transform_orders = BashOperator(
        task_id="silver_transform_orders",
        bash_command=(
            f"python {SPARK_ROOT}/silver/transform_orders.py"
        ),
    )

    silver_transform_order_items = BashOperator(
        task_id="silver_transform_order_items",
        bash_command=(
            f"python {SPARK_ROOT}/silver/transform_order_items.py"
        ),
    )

    silver_transform_order_payments = BashOperator(
        task_id="silver_transform_order_payments",
        bash_command=(
            f"python {SPARK_ROOT}/silver/transform_order_payments.py"
        ),
    )

    silver_transform_order_reviews = BashOperator(
        task_id="silver_transform_order_reviews",
        bash_command=(
            f"python {SPARK_ROOT}/silver/transform_order_reviews.py"
        ),
    )

    silver_transform_geolocation = BashOperator(
        task_id="silver_transform_geolocation",
        bash_command=(
            f"python {SPARK_ROOT}/silver/transform_geolocation.py"
        ),
    )

    silver_transform_category_translation = BashOperator(
        task_id="silver_transform_category_translation",
        bash_command=(
            f"python {SPARK_ROOT}/silver/"
            f"transform_category_translation.py"
        ),
    )

    silver_transforms = [
        silver_transform_customers,
        silver_transform_products,
        silver_transform_sellers,
        silver_transform_orders,
        silver_transform_order_items,
        silver_transform_order_payments,
        silver_transform_order_reviews,
        silver_transform_geolocation,
        silver_transform_category_translation,
    ]

    # ========================================================
    # 3. SILVER VALIDATIONS
    # ========================================================

    silver_validate_customers = BashOperator(
        task_id="silver_validate_customers",
        bash_command=(
            f"python {SPARK_ROOT}/silver/validate_customers.py"
        ),
    )

    silver_validate_products = BashOperator(
        task_id="silver_validate_products",
        bash_command=(
            f"python {SPARK_ROOT}/silver/validate_products.py"
        ),
    )

    silver_validate_sellers = BashOperator(
        task_id="silver_validate_sellers",
        bash_command=(
            f"python {SPARK_ROOT}/silver/validate_sellers.py"
        ),
    )

    silver_validate_orders = BashOperator(
        task_id="silver_validate_orders",
        bash_command=(
            f"python {SPARK_ROOT}/silver/validate_orders.py"
        ),
    )

    silver_validate_order_items = BashOperator(
        task_id="silver_validate_order_items",
        bash_command=(
            f"python {SPARK_ROOT}/silver/"
            f"validate_order_items.py"
        ),
    )

    silver_validate_order_payments = BashOperator(
        task_id="silver_validate_order_payments",
        bash_command=(
            f"python {SPARK_ROOT}/silver/"
            f"validate_order_payments.py"
        ),
    )

    silver_validate_order_reviews = BashOperator(
        task_id="silver_validate_order_reviews",
        bash_command=(
            f"python {SPARK_ROOT}/silver/"
            f"validate_order_reviews.py"
        ),
    )

    silver_validate_geolocation = BashOperator(
        task_id="silver_validate_geolocation",
        bash_command=(
            f"python {SPARK_ROOT}/silver/"
            f"validate_geolocation.py"
        ),
    )

    silver_validate_category_translation = BashOperator(
        task_id="silver_validate_category_translation",
        bash_command=(
            f"python {SPARK_ROOT}/silver/"
            f"validate_category_translation.py"
        ),
    )

    silver_validations = [
        silver_validate_customers,
        silver_validate_products,
        silver_validate_sellers,
        silver_validate_orders,
        silver_validate_order_items,
        silver_validate_order_payments,
        silver_validate_order_reviews,
        silver_validate_geolocation,
        silver_validate_category_translation,
    ]

    # ========================================================
    # 4. GOLD TRANSFORMATIONS
    # ========================================================

    gold_dim_customers = BashOperator(
        task_id="gold_dim_customers",
        bash_command=(
            f"python {SPARK_ROOT}/gold/build_dim_customers.py"
        ),
    )

    gold_dim_products = BashOperator(
        task_id="gold_dim_products",
        bash_command=(
            f"python {SPARK_ROOT}/gold/build_dim_products.py"
        ),
    )

    gold_dim_sellers = BashOperator(
        task_id="gold_dim_sellers",
        bash_command=(
            f"python {SPARK_ROOT}/gold/build_dim_sellers.py"
        ),
    )

    gold_fact_orders = BashOperator(
        task_id="gold_fact_orders",
        bash_command=(
            f"python {SPARK_ROOT}/gold/build_fact_orders.py"
        ),
    )

    gold_fact_order_items = BashOperator(
        task_id="gold_fact_order_items",
        bash_command=(
            f"python {SPARK_ROOT}/gold/build_fact_order_items.py"
        ),
    )

    gold_fact_reviews = BashOperator(
        task_id="gold_fact_reviews",
        bash_command=(
            f"python {SPARK_ROOT}/gold/build_fact_reviews.py"
        ),
    )

    gold_delivery_features = BashOperator(
        task_id="gold_delivery_features",
        bash_command=(
            f"python {SPARK_ROOT}/gold/build_delivery_features.py"
        ),
    )

    gold_seller_performance = BashOperator(
        task_id="gold_seller_performance",
        bash_command=(
            f"python {SPARK_ROOT}/gold/build_seller_performance.py"
        ),
    )

    gold_transforms = [
        gold_dim_customers,
        gold_dim_products,
        gold_dim_sellers,
        gold_fact_orders,
        gold_fact_order_items,
        gold_fact_reviews,
        gold_delivery_features,
        gold_seller_performance,
    ]

    # ========================================================
    # 5. GOLD VALIDATION
    # ========================================================

    gold_validation = BashOperator(
        task_id="gold_validation",
        bash_command=(
            f"python {PIPELINE_ROOT}/quality/validate_gold.py"
        ),
    )

    # ========================================================
    # DEPENDENCIES
    # ========================================================

    # Bronze -> all Silver transformations
    bronze_ingestion >> silver_transforms

    # Each Silver transformation -> its corresponding validation
    silver_transform_customers >> silver_validate_customers
    silver_transform_products >> silver_validate_products
    silver_transform_sellers >> silver_validate_sellers
    silver_transform_orders >> silver_validate_orders
    silver_transform_order_items >> silver_validate_order_items
    silver_transform_order_payments >> silver_validate_order_payments
    silver_transform_order_reviews >> silver_validate_order_reviews
    silver_transform_geolocation >> silver_validate_geolocation
    silver_transform_category_translation >> silver_validate_category_translation

    # ALL Silver validations must pass
    # before ANY Gold transformation can start.
    for validation in silver_validations:
        validation >> gold_transforms

    # ALL Gold transformations -> Gold validation
    gold_transforms >> gold_validation