-- ============================================================
-- SALES & REVENUE KPIs
-- ============================================================

-- 1. Overall sales KPIs
SELECT
COUNT(DISTINCT order_id) AS total_orders,
ROUND(SUM(order_total_price), 2) AS total_revenue,
ROUND(AVG(order_total_price), 2) AS average_order_value,
ROUND(MIN(order_total_price), 2) AS minimum_order_value,
ROUND(MAX(order_total_price), 2) AS maximum_order_value
FROM read_parquet('data/gold/delivery_features/*.parquet');

-- 2. Monthly sales performance
SELECT
DATE_TRUNC('month', o.order_purchase_timestamp) AS month,
COUNT(DISTINCT f.order_id) AS orders,
ROUND(SUM(f.order_total_price), 2) AS revenue,
ROUND(AVG(f.order_total_price), 2) AS average_order_value
FROM read_parquet('data/gold/delivery_features/*.parquet') f
JOIN read_parquet('data/gold/fact_orders/*.parquet') o
ON f.order_id = o.order_id
GROUP BY 1
ORDER BY 1;

-- 3. Sales by order status
SELECT
order_status,
COUNT(DISTINCT order_id) AS total_orders,
ROUND(SUM(order_total_price), 2) AS total_revenue,
ROUND(AVG(order_total_price), 2) AS average_order_value
FROM read_parquet('data/gold/delivery_features/*.parquet')
GROUP BY order_status
ORDER BY total_revenue DESC;

-- 4. Late orders impact on sales
SELECT
is_late,
COUNT(DISTINCT order_id) AS total_orders,
ROUND(SUM(order_total_price), 2) AS total_revenue,
ROUND(AVG(order_total_price), 2) AS average_order_value
FROM read_parquet('data/gold/delivery_features/*.parquet')
GROUP BY is_late
ORDER BY is_late;
