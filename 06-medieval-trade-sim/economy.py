"""
Economy model of a medieval city-management game: the player is a mayor of one of three cities that
trade with each other's markets through a neutral capital. Each mayor balances food, cloth, taxes,
buildings, temples (stability) and an army (defence and raids).

Everything that matters for balance is a parameter of :class:`Rules`, so the designer can run
tournaments of bot strategies, find dominant strategies and exploits, and test fixes before players do.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

GOODS = ["grain", "timber", "iron", "cloth", "weapons"]
GRAIN, TIMBER, IRON, CLOTH, WEAPONS = range(5)

# production per 100 inhabitants per month: each city has a specialisation
PRODUCTIVITY = np.array([
    # grain timber iron cloth
    [14.0, 3.0, 1.0, 4.0],    # Riverhold — breadbasket
    [7.0, 6.0, 6.0, 1.0],     # Stonepeak — mines & forests
    [9.0, 2.0, 1.0, 8.0],     # Weaver's Ford — textiles
])
CITY_NAMES = ["Riverhold", "Stonepeak", "Weaver's Ford"]


@dataclass(frozen=True)
class Rules:
    version: str = "v1"
    base_prices: tuple = (1.0, 2.0, 4.0, 3.0, 10.0)
    capital_spread: float = 0.10       # capital buys at base*(1-spread), sells at base*(1+spread)
    capital_impact: float = 0.0        # price elasticity to recent net sales (0 = infinite liquidity)
    capital_memory: float = 0.85       # decay of the net-sales memory per month
    impact_scale: float = 500.0        # units of net sales that move the price by exp(-impact)
    transport_fee: float = 0.0         # share of trade value lost to caravans
    workshop_capacity: int = 20        # weapons per workshop per month
    building_cost: float = 150.0
    building_cost_growth: float = 50.0
    building_upkeep: float = 2.0
    temple_upkeep: float = 3.0
    soldier_upkeep: float = 1.0
    recruit_cost: float = 5.0
    wealth_per_capita: float = 0.5     # taxable income per inhabitant per month
    tax_stability_penalty: float = 1.5 # stability lost per unit of tax above 15%
    raid_loot_share: float = 0.25
    raid_weariness: float = 0.0        # stability hit to the attacker per recent raid
    raid_army_loss: float = 0.0        # share of attacking army lost per raid
    bandits: bool = False              # rich, badly defended cities get robbed
    turns: int = 120                   # 10 years of months
    temple_bonus: float = 0.05         # stability target per temple
    stability_pop_effect: float = 0.006  # monthly pop growth per unit of (stability - 0.5)
    productivity: tuple = tuple(map(tuple, PRODUCTIVITY))

    @property
    def prices(self) -> np.ndarray:
        return np.array(self.base_prices)


@dataclass(frozen=True)
class Strategy:
    name: str
    tax: float
    army_per_100: float
    build_share: float = 0.0           # share of spare gold invested in production buildings
    temple_share: float = 0.0          # share of spare gold invested in temples
    raid: bool = False
    arbitrage: bool = False            # buy iron+timber, craft weapons, sell to the capital
    workshop_share: float = 0.0        # share of spare gold invested in workshops
    reserve: float = 200.0
    fee_discount: float = 0.0          # share of the transport fee saved (own caravans)


STRATEGIES = {
    "Merchant": Strategy("Merchant", tax=0.12, army_per_100=1.0, arbitrage=True, workshop_share=0.6, build_share=0.1,
                         fee_discount=0.75),
    "Builder": Strategy("Builder", tax=0.18, army_per_100=1.5, build_share=0.6),
    "Warlord": Strategy("Warlord", tax=0.20, army_per_100=8.0, raid=True, build_share=0.1),
    "Zealot": Strategy("Zealot", tax=0.15, army_per_100=1.5, temple_share=0.35, build_share=0.3),
    "Taxman": Strategy("Taxman", tax=0.32, army_per_100=2.0, build_share=0.2),
}


@dataclass
class City:
    idx: int
    strategy: Strategy
    pop: float = 1000.0
    gold: float = 500.0
    stability: float = 0.7
    stock: np.ndarray = field(default_factory=lambda: np.array([60.0, 20.0, 10.0, 20.0, 15.0]))
    army: float = 10.0
    farms: int = 0          # production buildings
    temples: int = 0
    workshops: int = 1
    recent_raids: list = field(default_factory=list)
    raided_this_turn: bool = False

    def buildings(self) -> int:
        return self.farms + self.temples + self.workshops - 1


class World:
    def __init__(self, rules: Rules, strategies: list[str], seed: int = 0, record: bool = False):
        self.r = rules
        self.rng = np.random.default_rng(seed)
        self.cities = [City(i, STRATEGIES[s]) for i, s in enumerate(strategies)]
        self.mem = np.zeros(5)  # capital net-sales memory per good
        self.t = 0
        self.record = record
        self.history: list[dict] = []
        self.flows = {k: 0.0 for k in ["taxes", "sales", "purchases", "upkeep", "construction", "recruitment",
                                       "bandits", "loot_transfer"]}

    # ---------------- capital market ---------------------------------------------------------
    def sell_price(self, g: int, c: City | None = None) -> float:
        fee = self.r.transport_fee * (1 - (c.strategy.fee_discount if c else 0.0))
        p = self.r.prices[g] * (1 - self.r.capital_spread)
        return p * np.exp(-self.r.capital_impact * self.mem[g] / self.r.impact_scale) * (1 - fee)

    def buy_price(self, g: int, c: City | None = None) -> float:
        fee = self.r.transport_fee * (1 - (c.strategy.fee_discount if c else 0.0))
        p = self.r.prices[g] * (1 + self.r.capital_spread)
        return p * np.exp(self.r.capital_impact * self.mem[g] / self.r.impact_scale) * (1 + fee)

    def sell(self, c: City, g: int, q: float) -> None:
        q = max(0.0, min(q, c.stock[g]))
        if q <= 0:
            return
        # integrate the price impact over the order (split into 4 tranches)
        for part in np.full(4, q / 4):
            rev = part * self.sell_price(g, c)
            c.gold += rev
            self.flows["sales"] += rev
            self.mem[g] += part
        c.stock[g] -= q

    def buy(self, c: City, g: int, q: float) -> float:
        bought = 0.0
        for part in np.full(4, max(q, 0) / 4):
            cost = part * self.buy_price(g, c)
            if cost > c.gold or part <= 0:
                break
            c.gold -= cost
            self.flows["purchases"] += cost
            self.mem[g] -= part
            bought += part
        c.stock[g] += bought
        return bought

    # ---------------- one month --------------------------------------------------------------
    def step(self) -> None:
        r = self.r
        for c in self.cities:
            c.raided_this_turn = False
        sat = {}
        for c in self.cities:
            s = c.strategy
            # production
            mult = (1 + 0.12 * c.farms) * (0.6 + 0.4 * c.stability)
            c.stock[:4] += np.array(r.productivity[c.idx]) * c.pop / 100 * mult
            # consumption
            need_grain = 10 * c.pop / 100 + 0.5 * c.army
            need_cloth = 2 * c.pop / 100
            if c.stock[GRAIN] < need_grain:
                self.buy(c, GRAIN, need_grain - c.stock[GRAIN])
            if c.stock[CLOTH] < need_cloth:
                self.buy(c, CLOTH, need_cloth - c.stock[CLOTH])
            food = min(1.0, c.stock[GRAIN] / need_grain)
            cloth = min(1.0, c.stock[CLOTH] / need_cloth)
            c.stock[GRAIN] = max(0.0, c.stock[GRAIN] - need_grain)
            c.stock[CLOTH] = max(0.0, c.stock[CLOTH] - need_cloth)
            weapons_wear = 0.05 * c.army
            c.stock[WEAPONS] = max(0.0, c.stock[WEAPONS] - weapons_wear)
            sat[c.idx] = (food, cloth)

            # crafting: own materials first; arbitrage buys more when the margin is positive
            cap = c.workshops * r.workshop_capacity
            if s.arbitrage:
                for _ in range(4):
                    margin = self.sell_price(WEAPONS, c) - self.buy_price(IRON, c) - self.buy_price(TIMBER, c)
                    if margin <= 0:
                        break
                    q = cap / 4
                    gi = self.buy(c, IRON, max(0, q - c.stock[IRON]))
                    gt = self.buy(c, TIMBER, max(0, q - c.stock[TIMBER]))
                    del gi, gt
                    made = min(q, c.stock[IRON], c.stock[TIMBER])
                    c.stock[IRON] -= made
                    c.stock[TIMBER] -= made
                    c.stock[WEAPONS] += made
                    self.sell(c, WEAPONS, made)
            # weapons for own army
            want_weapons = max(0.0, s.army_per_100 * c.pop / 100 - c.army) + 2 * weapons_wear
            made = min(cap, c.stock[IRON], c.stock[TIMBER], max(0.0, want_weapons - c.stock[WEAPONS]))
            c.stock[[IRON, TIMBER]] -= made
            c.stock[WEAPONS] += made

            # sell surpluses (keep 2 months of needs)
            self.sell(c, GRAIN, c.stock[GRAIN] - 2 * need_grain)
            self.sell(c, CLOTH, c.stock[CLOTH] - 2 * need_cloth)
            self.sell(c, TIMBER, c.stock[TIMBER] - 20)
            self.sell(c, IRON, c.stock[IRON] - 20)
            self.sell(c, WEAPONS, c.stock[WEAPONS] - want_weapons - 10)

            # taxes & upkeep
            tax = s.tax * c.pop * r.wealth_per_capita * (0.5 + 0.5 * c.stability)
            upkeep = (c.army * r.soldier_upkeep + (c.farms + c.workshops - 1) * r.building_upkeep
                      + c.temples * r.temple_upkeep)
            c.gold += tax - upkeep
            self.flows["taxes"] += tax
            self.flows["upkeep"] += upkeep

            # army
            target = s.army_per_100 * c.pop / 100
            if c.army < target:
                n = min(target - c.army, c.stock[WEAPONS], max(0.0, c.gold) / r.recruit_cost)
                c.army += n
                c.stock[WEAPONS] -= n
                c.gold -= n * r.recruit_cost
                self.flows["recruitment"] += n * r.recruit_cost
            elif c.army > 1.3 * target:
                c.army = 1.3 * target
            if c.gold < 0:  # unpaid soldiers desert
                c.army *= 0.9

            # investment
            spare = c.gold - s.reserve
            shares = {"farms": s.build_share, "temples": s.temple_share, "workshops": s.workshop_share}
            if s.arbitrage and self.sell_price(WEAPONS, c) <= self.buy_price(IRON, c) + self.buy_price(TIMBER, c):
                # a rational merchant stops expanding workshops once crafting arbitrage is unprofitable
                shares["farms"] += shares["workshops"]
                shares["workshops"] = 0.0
            for kind, share in shares.items():
                budget = spare * share
                n_existing = getattr(c, kind) - (1 if kind == "workshops" else 0)
                cost = r.building_cost + r.building_cost_growth * n_existing
                if budget >= cost:
                    setattr(c, kind, getattr(c, kind) + 1)
                    c.gold -= cost
                    self.flows["construction"] += cost

        # raids
        for c in self.cities:
            if not c.strategy.raid:
                continue
            others = [o for o in self.cities if o is not c]
            tgt = max(others, key=lambda o: o.gold)
            if c.army <= tgt.army:
                continue
            p = c.army / (c.army + 1.5 * tgt.army + 10)
            if self.rng.random() < p:
                loot = r.raid_loot_share * max(tgt.gold, 0)
                tgt.gold -= loot
                c.gold += loot
                tgt.raided_this_turn = True
                self.flows["loot_transfer"] += loot
            c.army *= 1 - r.raid_army_loss
            c.recent_raids.append(self.t)
        # bandits
        if r.bandits:
            for c in self.cities:
                p = np.clip(0.3 * c.gold / (c.gold + 2000) - 0.01 * c.army / (c.pop / 100), 0, 0.3)
                if self.rng.random() < p:
                    lost = 0.15 * max(c.gold, 0)
                    c.gold -= lost
                    self.flows["bandits"] += lost

        # population and stability
        for c in self.cities:
            food, cloth = sat[c.idx]
            c.recent_raids = [t for t in c.recent_raids if self.t - t < 6]
            target = (0.85 - r.tax_stability_penalty * max(c.strategy.tax - 0.15, 0) + r.temple_bonus * c.temples - 0.4 * (1 - food)
                      - 0.1 * (1 - cloth) - 0.15 * c.raided_this_turn - r.raid_weariness * len(c.recent_raids))
            c.stability = float(np.clip(c.stability + 0.15 * (np.clip(target, 0, 1) - c.stability), 0, 1))
            g = 0.04 * (food - 0.85) + 0.004 * (cloth - 0.5) + r.stability_pop_effect * (c.stability - 0.5)
            c.pop *= 1 + np.clip(g, -0.05, 0.02)

        self.mem *= self.r.capital_memory
        self.t += 1
        if self.record:
            for c in self.cities:
                self.history.append({"t": self.t, "city": CITY_NAMES[c.idx], "strategy": c.strategy.name,
                                     "pop": c.pop, "gold": c.gold, "stability": c.stability, "army": c.army,
                                     "farms": c.farms, "temples": c.temples, "workshops": c.workshops})

    def money_supply(self) -> float:
        return float(sum(c.gold for c in self.cities))

    def score(self, c: City) -> float:
        """Prosperity: population + half of net worth (gold + stock at base prices) + buildings."""
        worth = c.gold + float(c.stock @ self.r.prices)
        return c.pop + 0.5 * worth + 40 * c.buildings()

    def run(self) -> "World":
        self.money = [self.money_supply()]
        for _ in range(self.r.turns):
            self.step()
            self.money.append(self.money_supply())
        return self


def play(rules: Rules, strategies: list[str], seed: int = 0, record: bool = False) -> World:
    return World(rules, strategies, seed, record).run()


V1 = Rules(version="v1")
# v2 = the balance pass. Each change targets a diagnosed problem (see README):
#   capital_impact / transport_fee  -> finite market liquidity: dumping goods depresses prices (sink for gold)
#   weapons base price 10 -> 8      -> closes the craft-and-sell arbitrage loop
#   raid_weariness / army loss      -> raiding has a cost; loot share cut 25% -> 20%
#   bandits                         -> hoarding gold without defence is risky (sink)
#   temple upkeep, tax penalty, stability effect on growth -> no runaway "max stability" or "max tax" strategy
#   Stonepeak grain 7 -> 8.5, iron 6 -> 5; Weaver's Ford grain 9 -> 9.5, iron 1 -> 1.5 -> map balance
V2 = replace(V1, version="v2", base_prices=(1.0, 2.0, 4.0, 3.0, 8.0), capital_impact=0.5, transport_fee=0.08,
             raid_weariness=0.06, raid_army_loss=0.12, raid_loot_share=0.20, bandits=True,
             temple_bonus=0.035, temple_upkeep=4.5, tax_stability_penalty=2.5, stability_pop_effect=0.002,
             productivity=((14.0, 3.0, 1.0, 4.0), (8.5, 6.0, 5.0, 1.0), (9.5, 2.0, 1.5, 8.0)))
