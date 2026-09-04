-- ============================================================
-- SELLER KPIs
-- ============================================================

-- 1. Overall seller KPIs
SELECT
COUNT(DISTINCT seller_id) AS total_sellers,
COUNT(DISTINCT order_id) AS total_orders
FROM read_parquet('data/gold/seller_performance/*.parquet');

-- 2. Top sellers by orders
SELECT
seller_id,
MAX(seller_order_count) AS total_orders,
MAX(seller_late_order_count) AS late_orders,
ROUND(MAX(seller_late_rate) * 100, 2) AS late_rate_percentage,
ROUND(AVG(seller_avg_delivery_delay_days), 2) AS average_delivery_delay_days
FROM read_parquet('data/gold/seller_performance/*.parquet')
GROUP BY seller_id
ORDER BY total_orders DESC
LIMIT 20;

-- 3. Worst sellers by late rate
SELECT
seller_id,
MAX(seller_order_count) AS total_orders,
MAX(seller_late_order_count) AS late_orders,
ROUND(MAX(seller_late_rate) * 100, 2) AS late_rate_percentage,
ROUND(AVG(seller_avg_delivery_delay_days), 2) AS average_delivery_delay_days
FROM read_parquet('data/gold/seller_performance/*.parquet')
GROUP BY seller_id
HAVING MAX(seller_order_count) >= 20
ORDER BY late_rate_percentage DESC
LIMIT 20;

-- 4. Seller performance distribution
WITH seller_metrics AS (
SELECT
seller_id,
MAX(seller_order_count) AS total_orders,
MAX(seller_late_rate) AS late_rate
FROM read_parquet('data/gold/seller_performance/*.parquet')
GROUP BY seller_id
)

SELECT
CASE
WHEN late_rate < 0.10 THEN 'Excellent (<10%)'
WHEN late_rate < 0.20 THEN 'Good (10-20%)'
WHEN late_rate < 0.30 THEN 'Average (20-30%)'
ELSE 'Poor (30%+)'
END AS seller_performance,
COUNT(*) AS sellers
FROM seller_metrics
GROUP BY 1
ORDER BY 1;
