# Game Analytics & Economy Portfolio

**Maria Pyankova**: Product / Data Analyst (A/B testing, uplift modelling, economic evaluation) moving
into **game product analytics** and **game economy design**.
MSc Big Data & Machine Learning (ITMO), BSc Economics (HSE).

> Seven end-to-end projects. Each one starts from a business or design question, ends with a decision,
> and is fully reproducible (`make all`). Every number in the READMEs is produced by the code in this repo.

## Projects

| # | Project | Question | Headline result |
|---|---|---|---|
| 01 | [Game telemetry & SQL metrics layer](01-game-telemetry-sql) | How healthy is the game, and where does the money come from? | DuckDB SQL layer (retention, funnel, LTV/ROAS, Pareto). The DQ suite catches **all 5 injected defect types** with zero false alarms |
| 02 | [A/B test deep dive](02-ab-test-deep-dive) | Ship the new daily rewards? | CUPED cuts variance by 54%. 1,000 A/A tests show a naive ratio-metric test gives **17.6% false positives** and the delta method 4.8%. The novelty effect fades from +3.6% to +1.0% |
| 03 | [Uplift modelling for offers](03-uplift-offer-targeting) | Who should get a 50%-off offer? | The naive read says +$2.27/player; the truth is **−$0.61**. S/T/X/DR learners plus off-policy evaluation turn the campaign positive: **+$90k net per 100k players** |
| 04 | [Early LTV → ROAS → bids](04-early-ltv-roas) | Which UA channels to scale, using day-7 data? | Two-stage LightGBM predicts D90 cohort value at the sampling-noise floor (17% MAPE) with −1% bias, giving max-CPI bids per channel |
| 05 | [Gacha banner economy](05-gacha-economy) | How to tune pity, 50/50 and spark? | Exact Markov chain + Monte Carlo. A 150-pull spark costs **−1.4% revenue** and bounds the worst case; removing soft pity is +17% revenue at a cost in player trust |
| 06 | [Medieval trade economy simulation](06-medieval-trade-sim) | Is my game's economy broken? | Bot tournaments found ×421 inflation, an infinite-gold crafting exploit and a 71% dominant strategy. After the balance pass: **every strategy wins 23–41%** |
| 07 | [F2P monetisation teardown](07-f2p-monetization-teardown) | How does Genshin Impact monetise, and what would I test? | Price ladder, pacing and constellation economics quantified with public data, plus four test proposals with metrics and guardrails |

**Two roles, one toolkit.** Projects 01–04 are the **product analyst** track (metrics, experiments,
causal ML, UA economics). Projects 05–07 are the **game economist** track (probability, simulation,
balance, monetisation design). Project 03 bridges the two: it applies the uplift modelling I use in
banking to in-game offers.

<p align="center">
  <img src="03-uplift-offer-targeting/reports/figures/02_uplift_curves.png" width="90%" alt="uplift curves">
</p>

## Why synthetic data, and why that is a feature

Real game telemetry is proprietary, and public datasets are tiny and over-used. All product-analytics
projects run on [`gamesim`](gamesim/simulator.py), a simulator of a mobile F2P game: players with latent
engagement and spending traits, an onboarding funnel, decaying churn hazard, sessions, levels and purchases
over six price points. The upside is **known ground truth**: an experiment can carry a hidden +2.8% effect,
and the project shows whether the methodology recovers it. Most portfolios cannot validate their methods
this way.

## Skills map

| | 01 | 02 | 03 | 04 | 05 | 06 | 07 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| SQL / data modelling / data quality | ● | | | ○ | | | |
| Product metrics (retention, funnels, ARPU, ROAS) | ● | ● | ○ | ● | | | ○ |
| Experiment design & statistics | | ● | ○ | | | ○ | ● |
| Causal inference / uplift / off-policy evaluation | | ○ | ● | | | | |
| ML (LightGBM, calibration, validation) | | | ● | ● | | | |
| Probability & Monte Carlo | | ○ | | | ● | ● | ● |
| Game economy & balance design | | | | | ● | ● | ● |
| Business recommendations | ● | ● | ● | ● | ● | ● | ● |

## Run everything

```bash
git clone https://github.com/Marieart47/analytics-portfolio.git && cd analytics-portfolio
python -m venv .venv && source .venv/bin/activate
make install
make test     # unit tests: stats toolkit, gacha engine, simulator integrity, OPE, economy
make all      # regenerates every figure, table and results.json (~4 min)
```

```
gamesim/                  shared F2P telemetry simulator + plotting style
01-game-telemetry-sql/    sql/ (DuckDB), run.py, reports/
02-ab-test-deep-dive/     abstats.py (delta method, CUPED, SRM, …), run.py, reports/
03-uplift-offer-targeting/ data.py, run.py (S/T/X/DR learners, IPS/SNIPS), reports/
04-early-ltv-roas/        run.py (Tweedie, two-stage, calibration, power-law LTV), reports/
05-gacha-economy/         gacha.py (exact Markov chain + MC), run.py, reports/
06-medieval-trade-sim/    economy.py (agent-based model), run.py, reports/
07-f2p-monetization-teardown/ run.py, README teardown with sources
tests/                    pytest suite (runs in CI)
```

## Кратко по-русски

Портфолио для позиций **продуктового аналитика (middle) в геймдеве** и **игрового экономиста**. Семь
проектов, каждый с бизнес-вопросом, воспроизводимым кодом и выводом в виде рекомендации:

* SQL-слой метрик и проверки качества данных;
* A/B-тест с CUPED, дельта-методом, проверкой SRM и A/A-валидацией;
* uplift-таргетинг офферов (S/T/X/DR-learners, IPS/SNIPS);
* ранний прогноз LTV и ставки для закупки трафика;
* экономика gacha-баннеров (точная марковская цепь + Монте-Карло);
* балансировка экономики моей игры через турниры ботов;
* разбор монетизации Genshin Impact.

Данные для продуктовых проектов генерирует собственный симулятор игры. Поэтому истинные эффекты известны,
и каждый метод проверяется на то, восстанавливает ли он правду.

---
Contacts: see CV · License: MIT
