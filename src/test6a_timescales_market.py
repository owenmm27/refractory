"""Test 6a - Two-timescale volatility state for the market index.

Compares 1-, 2- and 3-unit volatility models (leaky integrators with different decay rates):
fit, stability across eras, true out-of-sample forecasts, and what dependence is left in the residuals
(Ljung-Box, long-lag autocorrelation of |z|, BDS with a simulate-and-refit bootstrap).

Usage: python3 -I src/test6a_timescales_market.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import bds

from common import acf, fit_multi, ljung_box, load_market, multi_forecast, multi_variance, simulate_multi, _t_loglik

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"


def summary(f, n):
    return {"loglik": float(f["loglik"]), "n_params": int(f["n_params"]),
            "bic": float(-2 * f["loglik"] + f["n_params"] * np.log(n)),
            "half_lives_days": [float(x) for x in f["half_lives"]], "loadings_w": [float(x) for x in f["w"]],
            "decay_b": [float(x) for x in f["b"]], "loop_gain_W": float(f["W"]),
            "largest_eigenvalue": float(f["eigs"][-1]), "asymmetry_delta": float(f["delta"]), "t_dof": float(f["nu"])}


def residual_diag(z):
    q, p = ljung_box(np.abs(z), 20)
    a = acf(np.abs(z), 250)
    return {"ljung_box_abs_z_20": {"Q": float(q), "p": float(p)},
            "mean_acf_abs_z_lags_1_20": float(a[1:21].mean()), "mean_acf_abs_z_lags_21_250": float(a[21:251].mean()),
            "se_of_mean_21_250_approx": float(1 / np.sqrt(len(z)) / np.sqrt(230 / 20))}, a


def qlike(rv, f):
    x = rv / f
    return x - np.log(x) - 1


def dm_test(l1, l2, lags):
    import statsmodels.api as sm
    d = l1 - l2
    d = d[np.isfinite(d)]
    m = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.params[0]), float(m.tvalues[0])


def _bds_boot(args):
    params, n, seed = args
    rng = np.random.default_rng(seed)
    x = simulate_multi(params, n, 1, rng)[:, 0]
    f = fit_multi(x - x.mean(), 2)
    return bds(f["z"], max_dim=4, distance=1.0)[0]


def main():
    OUT.mkdir(exist_ok=True)
    mkt = load_market(ROOT / "data")
    dates = mkt.index
    r = ((mkt - mkt.mean()) * 100).values
    n = len(r)
    res = {"sample": [str(dates[0].date()), str(dates[-1].date())]}

    # --- 1. fit K = 1, 2, 3 on the full century
    fits = {K: fit_multi(r, K) for K in (1, 2)}
    f2 = fits[2]
    # K=3 with extra starts that nest the K=2 solution, so the comparison is fair
    b1, b2 = f2["b"]
    f3 = fit_multi(r, 3, starts=[(0.5, 0.93, 0.995), (b1, (b1 + b2) / 2, b2), (0.3, b1, b2), (b1, b2, 0.999), (b1, b2, 0.9995)])
    fits[3] = f3
    res["fits"] = {f"K={K}": summary(f, n) for K, f in fits.items()}
    res["lr_test_K2_vs_K1"] = {"LR": float(2 * (f2["loglik"] - fits[1]["loglik"])), "df": 2}
    res["lr_test_K3_vs_K2"] = {"LR": float(2 * (f3["loglik"] - f2["loglik"])), "df": 2}

    # --- 2. stability: refit K=2 by era
    res["k2_by_era"] = {}
    for name, a, b in [("1926-1969", "1926", "1969"), ("1970-1999", "1970", "1999"), ("2000-2026", "2000", "2026")]:
        x = r[np.asarray((dates >= a) & (dates <= b + "-12-31"))]
        g = fit_multi(x - x.mean(), 2)
        g1 = fit_multi(x - x.mean(), 1)
        res["k2_by_era"][name] = summary(g, len(x)) | {"LR_vs_K1": float(2 * (g["loglik"] - g1["loglik"]))}

    # --- 3. residual diagnostics
    acfs = {}
    res["residuals"] = {}
    for K, f in fits.items():
        d, a = residual_diag(f["z"])
        res["residuals"][f"K={K}"] = d
        acfs[K] = a
    acf_raw = acf(np.abs(r), 250)

    # --- 4. true out-of-sample: estimate on 1926-1999, freeze, evaluate 2000-2026
    split = int(np.searchsorted(dates, pd.Timestamp("2000-01-01")))
    rin = r[:split] - r[:split].mean()
    oos = {}
    for K in (1, 2):
        g = fit_multi(rin, K)
        g["h0"] = rin.var()
        h = multi_variance(r, g["omega"], g["a"], g["b"], g["delta"], rin.var())
        ll = _t_loglik(r, h, g["nu"])
        f1 = h                                    # 1-day-ahead variance forecast for day t (made at t-1)
        f21 = multi_forecast(g, r, 21)            # made at t for days t+1..t+21
        rv21 = pd.Series(r ** 2).rolling(21).sum().shift(-21).values
        oos[K] = {"ll": ll[split:], "q1": qlike(np.maximum(r ** 2, 1e-4), f1)[split:],
                  "q21": qlike(rv21, f21)[split:], "half_lives": g["half_lives"]}
    res["oos_2000_2026"] = {
        "mean_loglik_per_day": {f"K={K}": float(np.nanmean(v["ll"])) for K, v in oos.items()},
        "total_loglik_gain_K2": float(np.nansum(oos[2]["ll"] - oos[1]["ll"])),
        "qlike_1day": {f"K={K}": float(np.nanmean(v["q1"])) for K, v in oos.items()},
        "qlike_21day": {f"K={K}": float(np.nanmean(v["q21"])) for K, v in oos.items()},
        "dm_1day_K1_minus_K2": dm_test(oos[1]["q1"], oos[2]["q1"], 5),
        "dm_21day_K1_minus_K2": dm_test(oos[1]["q21"], oos[2]["q21"], 30),
        "dm_loglik_K2_minus_K1": dm_test(oos[2]["ll"], oos[1]["ll"], 5),
        "in_sample_half_lives_K2": [float(x) for x in oos[2]["half_lives"]]}

    # --- 5. BDS on 2000-2026 residuals with a simulate-and-refit bootstrap under K=2
    sub = r[np.asarray(dates >= "2000")]
    sub = sub - sub.mean()
    g2 = fit_multi(sub, 2)
    g1 = fit_multi(sub, 1)
    stat2 = bds(g2["z"], max_dim=4, distance=1.0)[0]
    stat1 = bds(g1["z"], max_dim=4, distance=1.0)[0]
    params = {k: g2[k] for k in ("omega", "a", "b", "delta", "nu")}
    B = 199
    with ProcessPoolExecutor(max_workers=3) as ex:
        boot = np.array(list(ex.map(_bds_boot, [(params, len(sub), 5000 + i) for i in range(B)])))
    res["bds_2000_2026"] = {
        "K1_resid_stat_dims_2_3_4": [float(x) for x in stat1],
        "K2_resid_stat_dims_2_3_4": [float(x) for x in stat2],
        "K2_bootstrap_null_mean": [float(x) for x in boot.mean(0)], "K2_bootstrap_null_sd": [float(x) for x in boot.std(0)],
        "K2_bootstrap_p": [float((1 + np.sum(np.abs(boot[:, j] - boot[:, j].mean()) >= abs(stat2[j] - boot[:, j].mean()))) / (B + 1))
                           for j in range(3)], "n_boot": B}

    (OUT / "test6a.json").write_text(json.dumps(res, indent=2))
    comps = f2["comps"]
    np.savez(OUT / "test6a.npz", acf_raw=acf_raw, acf_k1=acfs[1], acf_k2=acfs[2], acf_k3=acfs[3],
             fast=comps[0], slow=comps[1], omega=f2["omega"], dates=dates.values.astype("datetime64[D]").astype(str),
             a=f2["a"], b=f2["b"])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
