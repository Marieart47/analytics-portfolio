-- Onboarding funnel: step conversion, step-to-step drop-off, split by channel.
WITH steps AS (
    SELECT p.channel, t.step_no, t.step, COUNT(DISTINCT t.player_id) AS players
    FROM tutorial t JOIN players p USING (player_id)
    GROUP BY 1, 2, 3
)
SELECT channel, step_no, step, players,
       players::DOUBLE / FIRST_VALUE(players) OVER w                      AS conv_from_start,
       players::DOUBLE / LAG(players) OVER w                              AS conv_from_prev
FROM steps
WINDOW w AS (PARTITION BY channel ORDER BY step_no)
ORDER BY channel, step_no;
