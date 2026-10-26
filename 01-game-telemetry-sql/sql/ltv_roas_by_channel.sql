-- Cumulative ARPU (gross bookings per install) by age and channel + ROAS vs CPI.
-- Only cohorts old enough are included at each age (censoring-aware).
WITH obs AS (SELECT MAX(event_date) AS last_date FROM activity),
p AS (
    SELECT player_id, channel, cpi, install_date,
           DATEDIFF('day', install_date, (SELECT last_date FROM obs)) AS max_age
    FROM players
),
rev AS (
    SELECT pu.player_id, DATEDIFF('day', p.install_date, pu.event_date) AS age, SUM(price_usd) AS revenue
    FROM purchases pu JOIN p USING (player_id) GROUP BY 1, 2
),
ages AS (SELECT UNNEST([0, 1, 3, 7, 14, 30, 60, 90]) AS age)
SELECT p.channel, ages.age,
       COUNT(DISTINCT p.player_id)                                        AS installs,
       COALESCE(SUM(r.revenue), 0) / COUNT(DISTINCT p.player_id)          AS cum_arpu,
       AVG(p.cpi)                                                         AS cpi,
       CASE WHEN AVG(p.cpi) > 0
            THEN COALESCE(SUM(r.revenue), 0) / COUNT(DISTINCT p.player_id) / AVG(p.cpi) END AS gross_roas
FROM ages
JOIN p ON p.max_age >= ages.age
LEFT JOIN rev r ON r.player_id = p.player_id AND r.age <= ages.age
GROUP BY 1, 2
ORDER BY 1, 2;
