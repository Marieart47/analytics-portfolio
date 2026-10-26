-- Cumulative share of revenue vs cumulative share of payers (sorted by spend desc).
WITH spend AS (SELECT player_id, SUM(price_usd) AS ltv FROM purchases GROUP BY 1)
SELECT ROW_NUMBER() OVER (ORDER BY ltv DESC)::DOUBLE / COUNT(*) OVER ()  AS payer_share,
       SUM(ltv) OVER (ORDER BY ltv DESC ROWS UNBOUNDED PRECEDING) / SUM(ltv) OVER () AS revenue_share
FROM spend ORDER BY 1;
