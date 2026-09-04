-- ============================================================
-- DELIVERY & OPERATIONS KPIs
-- ============================================================

-- 1. Overall delivery KPIs
SELECT
COUNT(DISTINCT f.order_id) AS total_orders,
ROUND(AVG(d.estimated_delivery_duration_days), 2) AS avg_estimated_delivery_days,
ROUND(AVG(f.delivery_duration_days), 2) AS avg_actual_delivery_days,
ROUND(
    AVG(
        CASE
            WHEN f.delivery_delay_days > 0
            THEN f.delivery_delay_days
            ELSE 0
        END
    ),
    2
) AS avg_delivery_delay_days,
ROUND(
    100.0 * SUM(CASE WHEN f.is_late = 1 THEN 1 ELSE 0 END)
    / COUNT(DISTINCT f.order_id),
    2
) AS late_delivery_rate
FROM read_parquet('data/gold/fact_orders/*.parquet') f
JOIN read_parquet('data/gold/delivery_features/*.parquet') d
ON f.order_id = d.order_id;

-- 2. Delivery performance by month

SELECT
DATE_TRUNC('month', order_purchase_timestamp) AS month,
COUNT(DISTINCT order_id) AS orders,

ROUND(
    AVG(delivery_duration_days),
    2
) AS avg_delivery_days,

ROUND(
    AVG(
        CASE
            WHEN delivery_delay_days > 0
            THEN delivery_delay_days
            ELSE 0
        END
    ),
    2
) AS avg_delay_days,

ROUND(
    AVG(is_late) * 100,
    2
) AS late_rate_percentage

FROM read_parquet('data/gold/fact_orders/*.parquet')

GROUP BY 1
ORDER BY 1;

-- 3. Late deliveries by seller
SELECT
seller_id,
MAX(seller_order_count) AS orders,
MAX(seller_late_order_count) AS late_orders,
ROUND(MAX(seller_late_rate) * 100, 2) AS late_rate_percentage,
ROUND(MAX(seller_avg_delivery_delay_days), 2)
AS avg_delivery_delay_days
FROM read_parquet('data/gold/seller_performance/*.parquet')
GROUP BY seller_id
HAVING MAX(seller_order_count) >= 20
ORDER BY late_rate_percentage DESC
LIMIT 20;

-- 4. Delivery performance by order status

SELECT
order_status,
COUNT(DISTINCT order_id) AS orders,

ROUND(
    AVG(
        CASE
            WHEN order_status = 'delivered'
            THEN delivery_duration_days
        END
    ),
    2
) AS avg_delivery_days,

ROUND(
    AVG(
        CASE
            WHEN order_status = 'delivered'
            THEN GREATEST(delivery_delay_days, 0)
        END
    ),
    2
) AS avg_delay_days

FROM read_parquet('data/gold/fact_orders/*.parquet')

GROUP BY order_status

ORDER BY orders DESC;

-- 5. Late delivery by purchase day
SELECT
purchase_day_of_week,
COUNT(DISTINCT order_id) AS orders,
ROUND(AVG(is_late) * 100, 2) AS late_rate_percentage
FROM read_parquet('data/gold/delivery_features/*.parquet')
GROUP BY purchase_day_of_week
ORDER BY purchase_day_of_week;
