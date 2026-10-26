"""
Project 03 — Uplift modelling for personalised offers.

Who should get the "50% off Gem Chest" offer so that *incremental* revenue is maximised?

Usage:  python 03-uplift-offer-targeting/run.py
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
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from data import CATEGORICAL, FEATURES, build  # noqa: E402
from gamesim.plotting import GREY, INK, PALETTE, save  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
FIG = HERE / "reports" / "figures"
OUT = HERE / "reports"
OUT.mkdir(parents=True, exist_ok=True)
NET = 0.70  # share of gross bookings kept after store fees
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def reg(seed=0):
    return lgb.LGBMRegressor(random_state=seed, **PARAMS)


# ---------------------------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------------------------
def s_learner(X, t, y, Xte):
    m = reg().fit(X.assign(_t=t), y)
    return m.predict(Xte.assign(_t=1)) - m.predict(Xte.assign(_t=0))


def t_learner(X, t, y, Xte, return_models=False):
    m1 = reg(1).fit(X[t == 1], y[t == 1])
    m0 = reg(2).fit(X[t == 0], y[t == 0])
    if return_models:
        return m0, m1
    return m1.predict(Xte) - m0.predict(Xte)


def x_learner(X, t, y, e, Xte, e_te):
    m0, m1 = t_learner(X, t, y, None, return_models=True)
    d1 = y[t == 1] - m0.predict(X[t == 1])
    d0 = m1.predict(X[t == 0]) - y[t == 0]
    tau1 = reg(3).fit(X[t == 1], d1)
    tau0 = reg(4).fit(X[t == 0], d0)
    return e_te * tau0.predict(Xte) + (1 - e_te) * tau1.predict(Xte)


def dr_learner(X, t, y, e, Xte, n_folds=2):
    """Doubly-robust learner with cross-fitting; propensity e is the *logged* one."""
    phi = np.zeros(len(y))
    for tr, va in KFold(n_folds, shuffle=True, random_state=0).split(X):
        m0 = reg(5).fit(X.iloc[tr][t[tr] == 0], y[tr][t[tr] == 0])
        m1 = reg(6).fit(X.iloc[tr][t[tr] == 1], y[tr][t[tr] == 1])
        mu0, mu1 = m0.predict(X.iloc[va]), m1.predict(X.iloc[va])
        tv, yv, ev = t[va], y[va], e[va]
        phi[va] = mu1 - mu0 + tv * (yv - mu1) / ev - (1 - tv) * (yv - mu0) / (1 - ev)
    return reg(7).fit(X, phi).predict(Xte)


def response_model(X, t, y, Xte):
    """The classic mistake: rank by P(purchase | offered) instead of by the *change* in revenue."""
    m = lgb.LGBMClassifier(random_state=8, **PARAMS).fit(X[t == 1], (y[t == 1] > 0).astype(int))
    return m.predict_proba(Xte)[:, 1]


# ---------------------------------------------------------------------------------------------
# Off-policy evaluation
# ---------------------------------------------------------------------------------------------
def policy_value(pi, t, y, e, kind="snips"):
    """Value (mean revenue per player) of deterministic policy pi, estimated from logged data."""
    p = np.where(t == 1, e, 1 - e)
    w = (pi == t) / p
    if kind == "ips":
        return float(np.mean(w * y))
    return float(np.sum(w * y) / np.sum(w))


def uplift_curve(score, t, y, e, ey0, tau, grid):
    order = np.argsort(-score)
    est, true = [], []
    base_est = policy_value(np.zeros_like(t), t, y, e)
    for k in grid:
        pi = np.zeros_like(t)
        pi[order[: int(round(k * len(t)))]] = 1
        est.append(policy_value(pi, t, y, e) - base_est)
        true.append(float(np.mean(tau * pi)))
    return np.array(est), np.array(true)


def main() -> None:
    print("building dataset ...")
    df = build()
    tr_df, te_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df.treatment)
    X, Xte = prep(tr_df), prep(te_df)
    t, y, e = tr_df.treatment.to_numpy(), tr_df.revenue_next_14.to_numpy(), tr_df.propensity.to_numpy()
    tt, yt, et = te_df.treatment.to_numpy(), te_df.revenue_next_14.to_numpy(), te_df.propensity.to_numpy()
    tau_te, ey0_te = te_df._tau.to_numpy(), te_df._ey0.to_numpy()
    R: dict = {"n_train": len(tr_df), "n_test": len(te_df), "treated_share": float(df.treatment.mean())}

    # ---- the naive read-out vs the truth ---------------------------------------------------------
    naive = df[df.treatment == 1].revenue_next_14.mean() - df[df.treatment == 0].revenue_next_14.mean()
    ipw_ate = float(np.mean(df.treatment * df.revenue_next_14 / df.propensity
                            - (1 - df.treatment) * df.revenue_next_14 / (1 - df.propensity)))
    R["ate"] = {"naive_difference": float(naive), "ipw": ipw_ate, "true": float(df._tau.mean())}
    print(R["ate"])

    # ---- who truly responds (ground truth view) ----------------------------------------------
    df["pay_decile"] = pd.qcut(df._z_pay, 10, labels=[f"D{i}" for i in range(1, 11)])
    gt = df.groupby("pay_decile", observed=True).agg(tau=("_tau", "mean"), ey0=("_ey0", "mean"),
                                                    payer=("is_payer", "mean"))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cols = [PALETTE[2] if v > 0 else PALETTE[3] for v in gt.tau]
    ax.bar(gt.index.astype(str), gt.tau, color=cols)
    for i, v in enumerate(gt.tau):
        lab = f"{'+' if v >= 0 else '−'}${abs(v):.2f}"
        if v < -3:
            ax.text(i, v / 2, lab, ha="center", fontsize=8.5, color="white", fontweight="bold")
        else:
            ax.text(i, v + (0.4 if v >= 0 else -1.0), lab, ha="center", fontsize=8.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("latent spending propensity decile (hidden from the model)")
    ax.set_ylabel("true uplift, $ per player")
    ax.set_title("Ground truth: the offer converts 'almost payers' and cannibalises big spenders")
    save(fig, FIG / "01_true_uplift_by_propensity.png")
    R["true_uplift_by_decile"] = gt.round(3).reset_index().astype({"pay_decile": str}).to_dict(orient="records")

    # ---- fit learners -----------------------------------------------------------------------
    print("fitting learners ...")
    scores = {
        "Response model": response_model(X, t, y, Xte),
        "S-learner": s_learner(X, t, y, Xte),
        "T-learner": t_learner(X, t, y, Xte),
        "X-learner": x_learner(X, t, y, e, Xte, et),
        "DR-learner": dr_learner(X, t, y, e, Xte),
    }
    rng = np.random.default_rng(0)
    scores["Random"] = rng.random(len(te_df))
    oracle = tau_te

    grid = np.linspace(0, 1, 41)
    curves = {name: uplift_curve(s, tt, yt, et, ey0_te, tau_te, grid) for name, s in scores.items()}
    curves["Oracle (true τ)"] = uplift_curve(oracle, tt, yt, et, ey0_te, tau_te, grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    styles = {"Response model": (PALETTE[3], "-"), "S-learner": (PALETTE[4], "-"), "T-learner": (PALETTE[1], "-"),
              "X-learner": (PALETTE[0], "-"), "DR-learner": (PALETTE[2], "-"), "Random": (GREY, ":"),
              "Oracle (true τ)": (INK, "--")}
    for name, (est, true) in curves.items():
        c, ls = styles[name]
        axes[0].plot(grid, est, color=c, ls=ls, lw=1.8, label=name)
        axes[1].plot(grid, true, color=c, ls=ls, lw=1.8, label=name)
    for ax, title in [(axes[0], "Estimated from logged data (SNIPS)"), (axes[1], "True (simulator oracle)")]:
        ax.axhline(0, color=INK, lw=0.8)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("share of players targeted, ranked by model score")
        ax.set_title(title)
    axes[0].set_ylabel("incremental revenue, $ per eligible player")
    axes[1].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Uplift curves: targeting the top of each ranking", x=0.07, ha="left", fontweight="bold")
    save(fig, FIG / "02_uplift_curves.png")

    # ---- model scorecard ----------------------------------------------------------------------
    rows = []
    v_none_true = float(np.mean(ey0_te))
    for name, s in scores.items():
        row = {"model": name}
        if name not in ("Response model", "Random"):
            row["pehe_rmse"] = float(np.sqrt(np.mean((s - tau_te) ** 2)))
            pi = (s > 0).astype(int)
            row["share_targeted"] = float(pi.mean())
            row["policy_true_incr"] = float(np.mean(tau_te * pi))
            row["policy_snips_incr"] = policy_value(pi, tt, yt, et) - policy_value(np.zeros_like(tt), tt, yt, et)
            row["policy_ips_incr"] = policy_value(pi, tt, yt, et, "ips") - policy_value(np.zeros_like(tt), tt, yt, et, "ips")
        row["spearman_vs_true_tau"] = float(spearmanr(s, tau_te).statistic)
        est, true = curves[name]
        row["auuc_true"] = float(np.trapezoid(true, grid))
        rows.append(row)
    card = pd.DataFrame(rows)
    card.to_csv(OUT / "model_scorecard.csv", index=False)
    R["scorecard"] = card.round(4).to_dict(orient="records")
    print(card.round(3).to_string())

    best = card.dropna(subset=["policy_snips_incr"]).sort_values("policy_snips_incr").iloc[-1].model
    R["best_model_by_snips"] = best
    s_best = scores[best]

    # ---- policies: estimated value with bootstrap CI vs truth ----------------------------------
    legacy = (te_df.active_days_14.to_numpy() >= np.median(te_df.active_days_14)).astype(int)
    resp_top = (scores["Response model"] >= np.quantile(scores["Response model"], 0.7)).astype(int)
    policies = {
        "Offer to nobody": np.zeros_like(tt),
        "Offer to everyone": np.ones_like(tt),
        "Legacy rule (active players)": legacy,
        "Response model, top 30%": resp_top,
        "Simple rule: non-payers only": (te_df.is_payer.to_numpy() == 0).astype(int),
        f"{best}: top 30% (limited budget)": (s_best >= np.quantile(s_best, 0.7)).astype(int),
        "Non-payers, random 30%": ((te_df.is_payer.to_numpy() == 0) & (rng.random(len(tt)) < 0.3 / 0.9)).astype(int),
        f"{best}: predicted uplift > 0": (s_best > 0).astype(int),
        "Oracle: true uplift > 0": (tau_te > 0).astype(int),
    }
    B = 300
    boot_idx = [rng.integers(0, len(tt), len(tt)) for _ in range(B)]
    prow = []
    for name, pi in policies.items():
        snips = policy_value(pi, tt, yt, et)
        ips = policy_value(pi, tt, yt, et, "ips")
        bs = np.array([policy_value(pi[i], tt[i], yt[i], et[i]) for i in boot_idx])
        bi = np.array([policy_value(pi[i], tt[i], yt[i], et[i], "ips") for i in boot_idx])
        prow.append({"policy": name, "share_targeted": float(pi.mean()), "snips": snips, "snips_lo": np.quantile(bs, .025),
                     "snips_hi": np.quantile(bs, .975), "ips": ips, "ips_sd": bi.std(), "snips_sd": bs.std(),
                     "true": float(np.mean(ey0_te + tau_te * pi))})
    pol = pd.DataFrame(prow)
    pol["true_incr_vs_none"] = pol["true"] - v_none_true
    pol.to_csv(OUT / "policies.csv", index=False)
    R["policies"] = pol.round(4).to_dict(orient="records")
    print(pol.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    yy = np.arange(len(pol))[::-1]
    for y_, r in zip(yy, pol.itertuples()):
        ax.plot([r.snips_lo, r.snips_hi], [y_, y_], color=PALETTE[0], lw=2.4)
        ax.plot(r.snips, y_, "o", color=PALETTE[0])
        ax.plot(r.true, y_, marker="|", color=PALETTE[3], ms=16, mew=2.2)
        ax.text(r.snips_hi + 0.05, y_, f"{r.share_targeted:.0%} targeted", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(yy, pol.policy)
    ax.grid(axis="x")
    ax.set_xlabel("expected revenue per eligible player over 14 days, $")
    ax.set_title("Policy value: SNIPS estimate with 95% bootstrap CI (red tick = true value)")
    ax.set_xlim(right=ax.get_xlim()[1] + 0.5)
    save(fig, FIG / "03_policy_values.png")

    # ---- calibration by decile of predicted uplift ---------------------------------------------
    dec = pd.qcut(pd.Series(s_best).rank(method="first"), 10, labels=False).to_numpy()
    cal = []
    for d in range(10):
        m = dec == d
        obs = np.mean(tt[m] * yt[m] / et[m] - (1 - tt[m]) * yt[m] / (1 - et[m]))
        cal.append({"decile": d + 1, "predicted": float(s_best[m].mean()), "observed_ipw": float(obs),
                    "true": float(tau_te[m].mean())})
    cal = pd.DataFrame(cal)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    w = 0.27
    ax.bar(cal.decile - w, cal.predicted, w, label=f"predicted ({best})", color=PALETTE[0])
    ax.bar(cal.decile, cal.observed_ipw, w, label="observed (IPW)", color=PALETTE[6])
    ax.bar(cal.decile + w, cal.true, w, label="true", color=PALETTE[3])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(cal.decile, [f"D{d}" for d in cal.decile])
    ax.set_xlabel("decile of predicted uplift (D1 = lowest)")
    ax.set_ylabel("uplift, $ per player")
    ax.set_title("Calibration: the model ranks well and gets the magnitude right")
    ax.legend()
    save(fig, FIG / "04_calibration.png")
    R["calibration"] = cal.round(3).to_dict(orient="records")

    # ---- who gets the offer -----------------------------------------------------------------
    te = te_df.assign(target=(s_best > 0).astype(int))
    prof = te.groupby("target").agg(players=("player_id", "size"), payer_share=("is_payer", "mean"),
                                    lifetime_revenue=("lifetime_revenue", "mean"),
                                    active_days_14=("active_days_14", "mean"), level=("level", "mean"),
                                    true_uplift=("_tau", "mean"))
    prof.index = prof.index.map({0: "excluded", 1: "targeted"})
    prof.to_csv(OUT / "targeting_profile.csv")
    R["profile"] = prof.round(3).reset_index().to_dict(orient="records")

    # ---- business case per 100k eligible players --------------------------------------------
    best_row = pol[pol.policy == f"{best}: predicted uplift > 0"].iloc[0]
    all_row = pol[pol.policy == "Offer to everyone"].iloc[0]
    legacy_row = pol[pol.policy.str.startswith("Legacy")].iloc[0]
    none_snips = pol[pol.policy == "Offer to nobody"].snips.iloc[0]
    R["business_per_100k"] = {
        "model_vs_none_net": float((best_row.snips - none_snips) * 100_000 * NET),
        "model_vs_everyone_net": float((best_row.snips - all_row.snips) * 100_000 * NET),
        "model_vs_legacy_net": float((best_row.snips - legacy_row.snips) * 100_000 * NET),
        "true_model_vs_legacy_net": float((best_row.true - legacy_row.true) * 100_000 * NET),
    }
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    print(json.dumps(R["business_per_100k"], indent=2), best)


if __name__ == "__main__":
    main()
