PY ?= python

.PHONY: all install test p01 p02 p03 p04 p05 p06 p07

all: p01 p02 p03 p04 p05 p06 p07

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

p01: ; $(PY) 01-game-telemetry-sql/run.py
p02: ; $(PY) 02-ab-test-deep-dive/run.py
p03: ; $(PY) 03-uplift-offer-targeting/run.py
p04: ; $(PY) 04-early-ltv-roas/run.py
p05: ; $(PY) 05-gacha-economy/run.py
p06: ; $(PY) 06-medieval-trade-sim/run.py
p07: ; $(PY) 07-f2p-monetization-teardown/run.py
