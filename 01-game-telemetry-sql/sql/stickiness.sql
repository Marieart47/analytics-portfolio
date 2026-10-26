-- DAU / MAU stickiness (rolling 28-day MAU).
WITH days AS (SELECT DISTINCT event_date FROM activity),
mau AS (
    SELECT d.event_date, COUNT(DISTINCT a.player_id) AS mau
    FROM days d
    JOIN activity a ON a.event_date BETWEEN d.event_date - INTERVAL 27 DAY AND d.event_date
    GROUP BY 1
),
dau AS (SELECT event_date, COUNT(DISTINCT player_id) AS dau FROM activity GROUP BY 1)
SELECT event_date, dau, mau, dau::DOUBLE / mau AS stickiness
FROM dau JOIN mau USING (event_date)
WHERE event_date >= (SELECT MIN(event_date) FROM activity) + INTERVAL 27 DAY
ORDER BY 1;
