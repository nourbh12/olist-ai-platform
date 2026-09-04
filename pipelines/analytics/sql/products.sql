-- ============================================================
-- PRODUCT KPIs
-- ============================================================

-- 1. Overall product KPIs
SELECT
COUNT(DISTINCT product_id) AS total_products,
ROUND(AVG(price), 2) AS average_product_price,
ROUND(SUM(total_item_value), 2) AS total_product_revenue,
ROUND(SUM(freight_value), 2) AS total_freight_revenue
FROM read_parquet('data/gold/fact_order_items/*.parquet');

-- 2. Top products by revenue
SELECT
product_id,
COUNT(DISTINCT order_id) AS orders,
SUM(total_item_value) AS revenue,
AVG(price) AS average_price
FROM read_parquet('data/gold/fact_order_items/*.parquet')
GROUP BY product_id
ORDER BY revenue DESC
LIMIT 20;

-- 3. Revenue by product category
SELECT
COALESCE(p.product_category_name, 'Unknown') AS category,
COUNT(DISTINCT f.order_id) AS orders,
COUNT(DISTINCT f.product_id) AS products,
ROUND(SUM(f.total_item_value), 2) AS revenue,
ROUND(AVG(f.price), 2) AS average_price
FROM read_parquet('data/gold/fact_order_items/*.parquet') f
JOIN read_parquet('data/gold/dim_products/*.parquet') p
ON f.product_id = p.product_id
GROUP BY 1
ORDER BY revenue DESC;

-- 4. Freight performance by category
SELECT
COALESCE(p.product_category_name, 'Unknown') AS category,
ROUND(AVG(f.freight_value), 2) AS average_freight,
ROUND(
SUM(f.freight_value) / NULLIF(SUM(f.total_item_value), 0) * 100,
2
) AS freight_percentage
FROM read_parquet('data/gold/fact_order_items/*.parquet') f
JOIN read_parquet('data/gold/dim_products/*.parquet') p
ON f.product_id = p.product_id
GROUP BY 1
ORDER BY freight_percentage DESC;
