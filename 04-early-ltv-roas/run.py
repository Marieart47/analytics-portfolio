"""
Project 04 — Early LTV prediction for user acquisition.

Predict each install's D90 revenue from its first 7 days, aggregate to channel × week cohorts,
and turn it into ROAS, payback and max-CPI bid recommendations ~12 weeks before D90 data exists.

Usage:  python 04-early-ltv-roas/run.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from gamesim.plotting import GREY, INK, PALETTE, save  # noqa: E402
from gamesim.simulator import CHANNELS, SimConfig, simulate  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
FIG = HERE / "reports" / "figures"
OUT = HERE / "reports"
OUT.mkdir(parents=True, exist_ok=True)
EARLY, TARGET = 7, 90
NET = 0.70
SPLIT_DAY = 60          # cohorts installed before day 60 = training, days 60-89 = "new" cohorts
CH_COLORS = {c: PALETTE[i] for i, c in enumerate(sorted(CHANNELS))}


def build():
    res = simulate(SimConfig(n_players=120_000, install_days=90, horizon_days=180, seed=11))
    p = res.players.copy()
    a = res.activity.merge(p[["player_id", "install_day"]])
    a["age"] = a.day - a.install_day
    b = res.purchases.merge(p[["player_id", "install_day"]])
    b["age"] = b.day - b.install_day
    tut = res.tutorial[res.tutorial.step == "tutorial_complete"].player_id

    ae = a[a.age <= EARLY]
    be = b[b.age <= EARLY]
    f = p.set_index("player_id")[["install_day", "channel", "platform", "country_tier", "cpi"]].copy()
    f["install_weekday"] = (f.install_day % 7)
    g = ae.groupby("player_id").agg(active_days=("age", "nunique"), sessions=("sessions", "sum"),
                                    minutes=("minutes", "sum"), level=("level", "max"), last_age=("age", "max"))
    f = f.join(g)
    for d in [0, 3, 7]:
        f[f"rev_d{d}"] = be[be.age <= d].groupby("player_id").price_usd.sum()
    f["n_purchases_d7"] = be.groupby("player_id").size()
    f["max_purchase_d7"] = be.groupby("player_id").price_usd.max()
    f["active_d4_7"] = ae[ae.age >= 4].groupby("player_id").age.nunique()
    f["tutorial_complete"] = f.index.isin(tut).astype(int)
    f["ltv_d90"] = b[b.age <= TARGET].groupby("player_id").price_usd.sum()
    f = f.fillna({c: 0 for c in f.columns if c not in ("channel", "platform", "country_tier")})
    return f.reset_index(), b


FEATS = ["active_days", "sessions", "minutes", "level", "last_age", "rev_d0", "rev_d3", "rev_d7",
         "n_purchases_d7", "max_purchase_d7", "active_d4_7", "tutorial_complete", "install_weekday",
         "channel", "platform", "country_tier"]


def X_of(df):
    X = df[FEATS].copy()
    for c in ["channel", "platform", "country_tier"]:
        X[c] = X[c].astype("category")
    return X


def power_law(t, a, b):
    return a * np.power(t, b)


def main() -> None:
    print("simulating 120k installs ...")
    df, buys = build()
    tr, te = df[df.install_day < SPLIT_DAY].copy(), df[df.install_day >= SPLIT_DAY].copy()
    R: dict = {"n_train": len(tr), "n_test": len(te)}

    # ---------------------------------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------------------------------
    preds = {}
    k = tr.ltv_d90.sum() / tr.rev_d7.sum()
    preds["D7 × global multiplier"] = te.rev_d7 * k
    k_ch = tr.groupby("channel").ltv_d90.sum() / tr.groupby("channel").rev_d7.sum()
    preds["D7 × channel multiplier"] = te.rev_d7 * te.channel.map(k_ch)
    preds["Channel average (no early signal)"] = te.channel.map(tr.groupby("channel").ltv_d90.mean())

    params = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    def fit_tweedie(d):
        return lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, **params).fit(X_of(d), d.ltv_d90)

    def fit_two_stage(d):
        fut = d.ltv_d90 - d.rev_d7
        clf = lgb.LGBMClassifier(**params).fit(X_of(d), (fut > 0).astype(int))
        pos = fut > 0
        regm = lgb.LGBMRegressor(objective="gamma", **params).fit(X_of(d)[pos], fut[pos])
        return clf, regm

    def pred_two_stage(models, d):
        clf, regm = models
        return d.rev_d7.to_numpy() + clf.predict_proba(X_of(d))[:, 1] * regm.predict(X_of(d))

    # Heavy-tailed targets make tree models under-predict the MEAN (whales are rare in each leaf).
    # Fix: an out-of-fold calibration factor sum(y) / sum(oof prediction), estimated on training cohorts.
    from sklearn.model_selection import KFold
    oof_tw, oof_ts = np.zeros(len(tr)), np.zeros(len(tr))
    for a_idx, b_idx in KFold(4, shuffle=True, random_state=0).split(tr):
        da, db = tr.iloc[a_idx], tr.iloc[b_idx]
        oof_tw[b_idx] = fit_tweedie(da).predict(X_of(db))
        oof_ts[b_idx] = pred_two_stage(fit_two_stage(da), db)
    cal_tw, cal_ts = tr.ltv_d90.sum() / oof_tw.sum(), tr.ltv_d90.sum() / oof_ts.sum()
    R["calibration_factors"] = {"tweedie": float(cal_tw), "two_stage": float(cal_ts)}

    tw = fit_tweedie(tr)
    ts = fit_two_stage(tr)
    clf = ts[0]
    preds["LightGBM Tweedie (raw)"] = tw.predict(X_of(te))
    preds["LightGBM Tweedie, calibrated"] = tw.predict(X_of(te)) * cal_tw
    preds["Two-stage, calibrated"] = pred_two_stage(ts, te) * cal_ts

    # ---------------------------------------------------------------------------------------
    # Evaluation — what UA cares about is COHORT-level accuracy (channel × install week)
    # ---------------------------------------------------------------------------------------
    te["week"] = (te.install_day - SPLIT_DAY) // 7
    y = te.ltv_d90.to_numpy()
    top = y >= np.quantile(y[y > 0], 0.8) if (y > 0).any() else y > 0  # top-20% of future payers
    rows, cohort_tabs = [], {}
    for name, p in preds.items():
        p = np.asarray(p, float)
        te["_p"] = p
        coh = te.groupby(["channel", "week"]).agg(actual=("ltv_d90", "mean"), pred=("_p", "mean"),
                                                  n=("ltv_d90", "size"))
        coh = coh[coh.n >= 300]
        cohort_tabs[name] = coh
        ch = te.groupby("channel").agg(actual=("ltv_d90", "mean"), pred=("_p", "mean"))
        rows.append({
            "model": name,
            "total_bias": p.sum() / y.sum() - 1,
            "cohort_mape": float(np.mean(np.abs(coh.pred / coh.actual - 1))),
            "channel_mape": float(np.mean(np.abs(ch.pred / ch.actual - 1))),
            "auc_top_spenders": float(roc_auc_score(top, p)) if p.std() > 0 else 0.5,
            "user_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        })
    se = te.groupby(["channel", "week"]).ltv_d90.agg(lambda v: v.std() / np.sqrt(len(v)) / v.mean())
    # mean absolute error of a PERFECT forecaster ≈ sqrt(2/pi) * relative SE of the cohort mean
    R["cohort_noise_floor_mape"] = float(np.sqrt(2 / np.pi) * se.mean())
    card = pd.DataFrame(rows).sort_values("cohort_mape")
    card.to_csv(OUT / "model_scorecard.csv", index=False)
    R["scorecard"] = card.round(4).to_dict(orient="records")
    print(card.round(3).to_string())
    best = card.iloc[0].model
    R["best_model"] = best
    te["pred"] = np.asarray(preds[best], float)

    # chart: cohort predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, name in zip(axes, ["D7 × global multiplier", best]):
        coh = cohort_tabs[name].reset_index()
        for ch, g in coh.groupby("channel"):
            ax.scatter(g.actual, g.pred, color=CH_COLORS[ch], label=ch, s=40, alpha=0.85)
        lim = [0, max(coh.actual.max(), coh.pred.max()) * 1.08]
        ax.plot(lim, lim, color=GREY, ls="--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        mape = card.set_index("model").loc[name, "cohort_mape"]
        ax.set_title(f"{name}\ncohort MAPE {mape:.1%}", fontsize=11)
        ax.set_xlabel("actual D90 ARPU of cohort, $")
        ax.grid(axis="x")
    axes[0].set_ylabel("predicted from first 7 days, $")
    axes[1].legend(fontsize=8.5)
    fig.suptitle("Channel × install-week cohorts (hold-out: cohorts installed after training period)",
                 x=0.06, y=1.06, ha="left", fontweight="bold")
    save(fig, FIG / "01_cohort_accuracy.png")

    # feature importance (gain)
    imp_model = tw if "Tweedie" in best else clf
    imp = pd.Series(imp_model.booster_.feature_importance("gain"), index=FEATS).sort_values()
    imp = imp / imp.sum()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(imp.index, imp.values, color=PALETTE[0])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x")
    ax.set_title(f"What predicts D90 value ({'Tweedie' if imp_model is tw else 'payer classifier'}, gain share)")
    save(fig, FIG / "02_feature_importance.png")
    R["top_features"] = imp.sort_values(ascending=False).head(5).round(3).to_dict()

    # ---------------------------------------------------------------------------------------
    # From D90 to lifetime: power-law ARPU curve per channel, validated on D150
    # ---------------------------------------------------------------------------------------
    old = df[df.install_day < 30]  # observed to >= age 150
    b_old = buys[buys.player_id.isin(old.player_id)]
    ages = np.arange(1, 151)
    curves, fits, val = {}, {}, []
    for ch, g in old.groupby("channel"):
        bb = b_old[b_old.player_id.isin(g.player_id)]
        daily = bb.groupby("age").price_usd.sum().reindex(range(0, 151), fill_value=0).cumsum() / len(g)
        curves[ch] = daily
        mask = (ages >= 7) & (ages <= 90)
        (a_, b_), _ = curve_fit(power_law, ages[mask], daily.loc[ages[mask]].to_numpy(), p0=[0.3, 0.7])
        fits[ch] = (a_, b_)
        val.append({"channel": ch, "actual_d150": float(daily.loc[150]), "fit_d150": float(power_law(150, a_, b_)),
                    "b": float(b_)})
    val = pd.DataFrame(val)
    val["error"] = val.fit_d150 / val.actual_d150 - 1
    R["powerlaw_validation_d150"] = val.round(4).to_dict(orient="records")
    # The back-test shows a systematic over-prediction (the curve decelerates faster than a power law).
    # Apply the average back-test error as a haircut to all extrapolations beyond D90.
    haircut = float(1 / (1 + val.error.mean()))
    R["extrapolation_haircut"] = haircut

    fig, ax = plt.subplots(figsize=(9, 4.6))
    t_ext = np.arange(1, 366)
    for ch in curves:
        a_, b_ = fits[ch]
        ax.plot(ages, curves[ch].loc[ages], color=CH_COLORS[ch], lw=2.2, label=ch)
        ax.plot(t_ext, power_law(t_ext, a_, b_), color=CH_COLORS[ch], lw=1.2, ls="--")
    ax.axvspan(7, 90, color=GREY, alpha=0.08)
    ax.text(48, ax.get_ylim()[1] * 0.92, "fit window D7–D90", ha="center", color=INK, fontsize=9)
    ax.axvline(150, color=GREY, lw=0.8, ls=":")
    ax.text(152, ax.get_ylim()[1] * 0.05, "validation\nD150", fontsize=8.5, color=INK)
    ax.set_xlabel("days since install")
    ax.set_ylabel("cumulative gross ARPU, $")
    ax.set_title("Lifetime extrapolation: ARPU(t) = a·t^b, solid = observed, dashed = fit (before haircut)")
    ax.legend()
    save(fig, FIG / "03_ltv_curves.png")

    # ---------------------------------------------------------------------------------------
    # UA decision table for the NEW cohorts — available on day 7, not day 90
    # ---------------------------------------------------------------------------------------
    rows = []
    for ch, g in te[te.cpi > 0].groupby("channel"):
        a_, b_ = fits[ch]
        pred90 = g.pred.mean()
        scale = pred90 / power_law(90, a_, b_)          # anchor the channel curve to the new prediction
        ltv180 = power_law(180, a_, b_) * scale * haircut
        ltv365 = power_law(365, a_, b_) * scale * haircut
        cpi = g.cpi.mean()
        net_curve = NET * power_law(t_ext, a_, b_) * scale * np.where(t_ext > 90, haircut, 1.0)
        payback = int(t_ext[np.argmax(net_curve >= cpi)]) if (net_curve >= cpi).any() else None
        rows.append({"channel": ch, "installs": len(g), "cpi": cpi,
                     "pred_ltv_d90": pred90, "actual_ltv_d90": g.ltv_d90.mean(),
                     "net_roas_d90_pred": NET * pred90 / cpi, "net_roas_d90_actual": NET * g.ltv_d90.mean() / cpi,
                     "net_roas_d365_pred": NET * ltv365 / cpi,
                     "payback_day": payback,
                     "max_cpi_d180_payback": NET * ltv180,
                     "max_cpi_d365_roas120": NET * ltv365 / 1.2})
    ua = pd.DataFrame(rows)
    ua["bid_change"] = ua.max_cpi_d180_payback / ua.cpi - 1   # bid rule: pay back within 180 days
    ua.to_csv(OUT / "ua_decisions.csv", index=False)
    R["ua"] = ua.round(3).to_dict(orient="records")
    print(ua.round(2).to_string())

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(ua))
    w = 0.26
    ax.bar(x - w, ua.net_roas_d90_pred, w, color=PALETTE[0], label="D90, predicted on day 7")
    ax.bar(x, ua.net_roas_d90_actual, w, color=PALETTE[6], label="D90, actual (known 83 days later)")
    ax.bar(x + w, ua.net_roas_d365_pred, w, color=PALETTE[2], label="D365, extrapolated (back-test haircut applied)")
    for i, r in ua.iterrows():
        for dx, v in [(-w, r.net_roas_d90_pred), (0, r.net_roas_d90_actual), (w, r.net_roas_d365_pred)]:
            ax.text(i + dx, v + 0.03, f"{v:.0%}", ha="center", fontsize=8)
    ax.axhline(1, color=INK, ls="--", lw=1)
    ax.set_xticks(x, ua.channel)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Net ROAS of the new cohorts by channel")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, FIG / "04_roas.png")

    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    print(val.round(3))


if __name__ == "__main__":
    main()
