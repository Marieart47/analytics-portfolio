-- Full retention curve (day 0..90) by acquisition channel, censoring-aware:
-- the denominator for day N contains only players old enough to be observed at day N.
WITH obs AS (SELECT MAX(event_date) AS last_date FROM activity),
p AS (
    SELECT player_id, channel, install_date,
           DATEDIFF('day', install_date, (SELECT last_date FROM obs)) AS max_age
    FROM players
),
act AS (
    SELECT DISTINCT a.player_id, DATEDIFF('day', p.install_date, a.event_date) AS age
    FROM activity a JOIN p USING (player_id)
),
ages AS (SELECT range AS age FROM range(0, 91))
SELECT p.channel, ages.age,
       COUNT(DISTINCT p.player_id)                                   AS eligible,
       COUNT(DISTINCT act.player_id)::DOUBLE / COUNT(DISTINCT p.player_id) AS retention
FROM ages
JOIN p ON p.max_age >= ages.age
LEFT JOIN act ON act.player_id = p.player_id AND act.age = ages.age
GROUP BY 1, 2
ORDER BY 1, 2;
