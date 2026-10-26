-- Classic N-day retention by weekly install cohort.
-- Right-censoring: a cohort only gets a value for day N if ALL its players are at least N days old
-- at the end of the observation window; otherwise the cell is NULL (not a misleading low number).
WITH obs AS (SELECT MAX(event_date) AS last_date FROM activity),
cohorts AS (
    SELECT player_id, install_date, DATE_TRUNC('week', install_date) AS cohort_week FROM players
),
active_age AS (
    SELECT DISTINCT a.player_id, DATEDIFF('day', c.install_date, a.event_date) AS age
    FROM activity a JOIN cohorts c USING (player_id)
),
grid AS (SELECT UNNEST([1, 3, 7, 14, 30, 60, 90]) AS day_n),
sizes AS (
    SELECT cohort_week, COUNT(*) AS cohort_size, MAX(install_date) AS last_install FROM cohorts GROUP BY 1
)
SELECT
    s.cohort_week,
    s.cohort_size,
    g.day_n,
    CASE WHEN s.last_install + g.day_n * INTERVAL 1 DAY <= (SELECT last_date FROM obs)
         THEN COUNT(DISTINCT aa.player_id)::DOUBLE / s.cohort_size END AS retention
FROM sizes s
CROSS JOIN grid g
LEFT JOIN cohorts c   ON c.cohort_week = s.cohort_week
LEFT JOIN active_age aa ON aa.player_id = c.player_id AND aa.age = g.day_n
GROUP BY s.cohort_week, s.cohort_size, s.last_install, g.day_n
ORDER BY 1, 3;
