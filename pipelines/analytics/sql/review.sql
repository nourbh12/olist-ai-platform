-- ============================================================
-- CUSTOMER REVIEW KPIs
-- ============================================================

-- 1. Overall review KPIs
SELECT
COUNT(*) AS total_reviews,
ROUND(AVG(review_score), 2) AS average_review_score
FROM read_parquet('data/gold/fact_reviews/*.parquet');

-- 2. Review score distribution
SELECT
review_score,
COUNT(*) AS reviews,
ROUND(
COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
2
) AS percentage
FROM read_parquet('data/gold/fact_reviews/*.parquet')
GROUP BY review_score
ORDER BY review_score;

-- 3. Review score vs delivery performance

SELECT
r.review_score,
COUNT(*) AS reviews,

ROUND(
    AVG(
        CASE
            WHEN f.delivery_delay_days > 0
            THEN f.delivery_delay_days
            ELSE 0
        END
    ),
    2
) AS average_delivery_delay,

ROUND(
    AVG(f.is_late) * 100,
    2
) AS late_delivery_rate

FROM read_parquet('data/gold/fact_reviews/*.parquet') r

JOIN read_parquet('data/gold/fact_orders/*.parquet') f
ON r.order_id = f.order_id

GROUP BY r.review_score

ORDER BY r.review_score;

-- 4. Reviews by delivery status
SELECT
CASE
WHEN f.is_late = 1 THEN 'Late'
ELSE 'On Time'
END AS delivery_status,
COUNT(*) AS reviews,
ROUND(AVG(r.review_score), 2) AS average_review_score
FROM read_parquet('data/gold/fact_reviews/*.parquet') r
JOIN read_parquet('data/gold/fact_orders/*.parquet') f
ON r.order_id = f.order_id
GROUP BY 1
ORDER BY 1;

-- 5. Review score by month
SELECT
DATE_TRUNC('month', review_creation_date) AS month,
COUNT(*) AS reviews,
ROUND(AVG(review_score), 2) AS average_review_score
FROM read_parquet('data/gold/fact_reviews/*.parquet')
GROUP BY 1
ORDER BY 1;
