-- Revenue concentration: payer tiers by lifetime spend and the Pareto curve.
WITH spend AS (SELECT player_id, SUM(price_usd) AS ltv FROM purchases GROUP BY 1),
tiered AS (
    SELECT *, CASE WHEN ltv < 10 THEN '1_minnow (<$10)'
                   WHEN ltv < 100 THEN '2_dolphin ($10-100)'
                   ELSE '3_whale ($100+)' END AS tier
    FROM spend
)
SELECT tier, COUNT(*) AS payers,
       COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER ()           AS share_of_payers,
       SUM(ltv) AS revenue,
       SUM(ltv) / SUM(SUM(ltv)) OVER ()                    AS share_of_revenue,
       AVG(ltv) AS avg_ltv
FROM tiered GROUP BY 1 ORDER BY 1;
