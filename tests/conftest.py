import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT / "02-ab-test-deep-dive", ROOT / "05-gacha-economy", ROOT / "06-medieval-trade-sim",
          ROOT / "03-uplift-offer-targeting"]:
    sys.path.insert(0, str(p))
