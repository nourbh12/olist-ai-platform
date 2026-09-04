-- ============================================================
-- CUSTOMER KPIs
-- ============================================================

-- 1. Overall customer KPIs
SELECT
COUNT(DISTINCT c.customer_unique_id) AS total_customers,
COUNT(DISTINCT f.order_id) AS total_orders,
ROUND(
COUNT(DISTINCT f.order_id)::DOUBLE
/ COUNT(DISTINCT c.customer_unique_id),
2
) AS orders_per_customer,
ROUND(
SUM(f.order_total_price)
/ COUNT(DISTINCT c.customer_unique_id),
2
) AS revenue_per_customer
FROM read_parquet('data/gold/delivery_features/*.parquet') f
JOIN read_parquet('data/gold/dim_customers/*.parquet') c
ON f.customer_id = c.customer_id;

-- 2. Customer performance by state
SELECT
c.customer_state,
COUNT(DISTINCT c.customer_unique_id) AS customers,
COUNT(DISTINCT f.order_id) AS orders,
ROUND(SUM(f.order_total_price), 2) AS revenue,
ROUND(AVG(f.order_total_price), 2) AS average_order_value
FROM read_parquet('data/gold/delivery_features/*.parquet') f
JOIN read_parquet('data/gold/dim_customers/*.parquet') c
ON f.customer_id = c.customer_id
GROUP BY c.customer_state
ORDER BY revenue DESC;

-- 3. Customers by number of orders
WITH customer_orders AS (
SELECT
c.customer_unique_id,
COUNT(DISTINCT f.order_id) AS order_count
FROM read_parquet('data/gold/delivery_features/*.parquet') f
JOIN read_parquet('data/gold/dim_customers/*.parquet') c
ON f.customer_id = c.customer_id
GROUP BY c.customer_unique_id
)

SELECT
CASE
WHEN order_count = 1 THEN '1 order'
WHEN order_count BETWEEN 2 AND 3 THEN '2-3 orders'
WHEN order_count BETWEEN 4 AND 5 THEN '4-5 orders'
ELSE '6+ orders'
END AS customer_segment,
COUNT(*) AS customers
FROM customer_orders
GROUP BY 1
ORDER BY
CASE customer_segment
WHEN '1 order' THEN 1
WHEN '2-3 orders' THEN 2
WHEN '4-5 orders' THEN 3
ELSE 4
END;

-- 4. Monthly customer activity
SELECT
DATE_TRUNC('month', o.order_purchase_timestamp) AS month,
COUNT(DISTINCT c.customer_unique_id) AS active_customers,
COUNT(DISTINCT f.order_id) AS orders,
ROUND(SUM(f.order_total_price), 2) AS revenue
FROM read_parquet('data/gold/delivery_features/*.parquet') f
JOIN read_parquet('data/gold/fact_orders/*.parquet') o
ON f.order_id = o.order_id
JOIN read_parquet('data/gold/dim_customers/*.parquet') c
ON f.customer_id = c.customer_id
GROUP BY 1
ORDER BY 1;