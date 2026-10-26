-- Daily topline KPIs: DAU, installs, revenue, conversion, ARPDAU, ARPPU, sessions per DAU.
WITH dau AS (
    SELECT event_date, COUNT(DISTINCT player_id) AS dau, SUM(sessions) AS sessions, SUM(minutes) AS minutes
    FROM activity GROUP BY 1
), rev AS (
    SELECT event_date, SUM(price_usd) AS revenue, COUNT(DISTINCT player_id) AS payers, COUNT(*) AS transactions
    FROM purchases GROUP BY 1
), inst AS (
    SELECT install_date AS event_date, COUNT(*) AS installs FROM players GROUP BY 1
)
SELECT
    d.event_date,
    d.dau,
    COALESCE(i.installs, 0)                          AS installs,
    COALESCE(r.revenue, 0)                           AS revenue,
    COALESCE(r.payers, 0)                            AS payers,
    COALESCE(r.transactions, 0)                      AS transactions,
    COALESCE(r.revenue, 0) / d.dau                   AS arpdau,
    COALESCE(r.payers, 0)::DOUBLE / d.dau            AS daily_conversion,
    r.revenue / NULLIF(r.payers, 0)                  AS arppu,
    d.sessions::DOUBLE / d.dau                       AS sessions_per_dau,
    d.minutes / d.dau                                AS minutes_per_dau
FROM dau d
LEFT JOIN rev  r USING (event_date)
LEFT JOIN inst i USING (event_date)
ORDER BY 1;
