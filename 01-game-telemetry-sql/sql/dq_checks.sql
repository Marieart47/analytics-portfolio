-- Data-quality checks. Every query returns the number of offending rows; 0 = pass.
-- Each check is separated by a line starting with "-- check:" so run.py can execute them one by one.
-- check: duplicate transaction ids
SELECT COUNT(*) - COUNT(DISTINCT transaction_id) FROM purchases;
-- check: duplicated purchase rows (same player, day, sku, price) beyond a plausible repeat rate
SELECT COALESCE(SUM(n - 1), 0) FROM (
    SELECT player_id, event_date, sku, price_usd, COUNT(*) AS n FROM purchases GROUP BY ALL HAVING COUNT(*) > 2
);
-- check: purchases before install
SELECT COUNT(*) FROM purchases pu JOIN players p USING (player_id) WHERE pu.event_date < p.install_date;
-- check: activity before install
SELECT COUNT(*) FROM activity a JOIN players p USING (player_id) WHERE a.event_date < p.install_date;
-- check: orphan events (player not in installs table)
SELECT (SELECT COUNT(*) FROM activity  WHERE player_id NOT IN (SELECT player_id FROM players))
     + (SELECT COUNT(*) FROM purchases WHERE player_id NOT IN (SELECT player_id FROM players));
-- check: purchases on days without a session
SELECT COUNT(*) FROM purchases pu
LEFT JOIN activity a ON a.player_id = pu.player_id AND a.event_date = pu.event_date
WHERE a.player_id IS NULL;
-- check: non-positive or unknown price points
SELECT COUNT(*) FROM purchases WHERE price_usd <= 0 OR price_usd NOT IN (0.99, 4.99, 9.99, 19.99, 49.99, 99.99);
-- check: installs missing from tutorial_start
SELECT COUNT(*) FROM players WHERE player_id NOT IN (SELECT player_id FROM tutorial WHERE step_no = 1);
-- check: funnel monotonicity violations (later step without earlier step)
SELECT COUNT(*) FROM tutorial t
WHERE t.step_no > 1 AND NOT EXISTS (
    SELECT 1 FROM tutorial t2 WHERE t2.player_id = t.player_id AND t2.step_no = t.step_no - 1);
-- check: implausible session counts (> 50 per day)
SELECT COUNT(*) FROM activity WHERE sessions > 50 OR sessions < 1;
