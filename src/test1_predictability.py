"""Test 1 - Where does predictability live, and is there hidden state beyond volatility?

Usage: python3 -I src/test1_predictability.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import bds

from common import acf, fit_gjr, hill, ljung_box, load_industries, load_market, load_stocks, simulate_gjr

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"
ERAS = [("1926-1969", "1926", "1969"), ("1970-1999", "1970", "1999"), ("2000-2026", "2000", "2026")]


# ----------------------------------------------------------------------------- helpers
def walk_forward_r2(y, X, start=2520, step=252):
    """Expanding-window OLS forecast of y from X (already lagged), refit every `step` days.
    Benchmark = expanding historical mean of y. Returns per-day model and benchmark predictions."""
    n = len(y)
    pred = np.full(n, np.nan)
    bench = np.full(n, np.nan)
    Xc = np.column_stack([np.ones(n), X])
    for s in range(start, n, step):
        b, *_ = np.linalg.lstsq(Xc[:s], y[:s], rcond=None)
        e = min(s + step, n)
        pred[s:e] = Xc[s:e] @ b
        bench[s:e] = y[:s].mean()
    return pred, bench


def oos_r2(y, pred, bench, mask=None):
    ok = np.isfinite(pred)
    if mask is not None:
        ok &= mask
    return 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - bench[ok]) ** 2)


def lag_features(x):
    """Yesterday, last-week mean, last-month mean of x, aligned to predict x[t] (uses data up to t-1)."""
    s = pd.Series(x)
    return np.column_stack([s.shift(1), s.shift(1).rolling(5).mean(), s.shift(1).rolling(21).mean()])


def quantile_symbols(x, k):
    ranks = pd.Series(x).rank(method="first").values
    return np.minimum((ranks - 1) * k // len(x), k - 1).astype(int)


def mutual_info(a, b, k):
    joint = np.zeros((k, k))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])).sum())


def mi_profile(x, k=8, lags=20, n_shuffle=200, rng=None):
    """Mutual information (bits) between x[t] and x[t-lag] on k quantile bins, with shuffle baseline."""
    sym = quantile_symbols(x, k)
    real = np.array([mutual_info(sym[:-l], sym[l:], k) for l in range(1, lags + 1)])
    sh = np.array([mutual_info(*(lambda p: (p[:-1], p[1:]))(rng.permutation(sym)), k) for _ in range(n_shuffle)])
    return real, float(sh.mean()), float(np.quantile(sh, 0.99))


def _bds_boot_one(args):
    params, n, seed = args
    rng = np.random.default_rng(seed)
    sim, _ = simulate_gjr(params, n, 1, rng)
    x = sim[:, 0] - sim[:, 0].mean()
    f = fit_gjr(x)
    return bds(f["z"], max_dim=4, distance=1.0)[0]


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    mkt = load_market(ROOT / "data")
    r = ((mkt - mkt.mean()) * 100).values
    dates = mkt.index
    res = {"sample": [str(dates[0].date()), str(dates[-1].date())], "n_days": int(len(r))}

    # (a) autocorrelation of direction vs magnitude
    L = 250
    acf_r, acf_abs = acf(r, L), acf(np.abs(r), L)
    res["acf"] = {"lag1_return": float(acf_r[1]), "lag1_abs": float(acf_abs[1]),
                  "mean_abs_lags_1_20_return": float(np.mean(np.abs(acf_r[1:21]))),
                  "mean_lags_1_20_abs": float(np.mean(acf_abs[1:21])),
                  "abs_lag_100": float(acf_abs[100]), "abs_lag_250": float(acf_abs[250]),
                  "white_noise_band_95": float(1.96 / np.sqrt(len(r)))}
    res["acf_by_era"] = {}
    for name, a, b in ERAS:
        x = r[(dates >= a) & (dates <= b + "-12-31")]
        res["acf_by_era"][name] = {"lag1_return": float(acf(x, 1)[1]), "lag1_abs": float(acf(np.abs(x), 1)[1]),
                                    "se": float(1 / np.sqrt(len(x)))}

    # (b) fat tails
    res["tail_index_hill_2.5pct"] = {"both": float(hill(r)), "left": float(hill(-r[r < 0])), "right": float(hill(r[r > 0]))}
    res["kurtosis"] = float(pd.Series(r).kurt())

    # (c) out-of-sample forecastability: direction vs magnitude (linear, walk-forward)
    X_dir, X_mag = lag_features(r), lag_features(np.abs(r))
    ok = np.isfinite(X_dir).all(1)
    y_dir, y_mag = r[ok], np.abs(r)[ok]
    d_ok = dates[ok]
    p_dir, b_dir = walk_forward_r2(y_dir, X_dir[ok])
    p_mag, b_mag = walk_forward_r2(y_mag, X_mag[ok])
    res["oos_r2"] = {"direction_all": float(oos_r2(y_dir, p_dir, b_dir)), "magnitude_all": float(oos_r2(y_mag, p_mag, b_mag))}
    for name, a, b in ERAS:
        m = np.asarray((d_ok >= a) & (d_ok <= b + "-12-31"))
        if np.isfinite(p_dir[m]).sum() > 500:
            res["oos_r2"][f"direction_{name}"] = float(oos_r2(y_dir, p_dir, b_dir, m))
            res["oos_r2"][f"magnitude_{name}"] = float(oos_r2(y_mag, p_mag, b_mag, m))
    v = np.isfinite(p_dir)
    res["sign_accuracy"] = {"model": float(np.mean(np.sign(p_dir[v]) == np.sign(y_dir[v]))),
                            "always_up": float(np.mean(y_dir[v] + mkt.mean() * 100 > 0))}

    # (d) remove the volatility state; what is left?
    fit = fit_gjr(r)
    z = fit["z"]
    res["garch"] = {k: float(fit[k]) for k in ("omega", "alpha", "gamma", "beta", "nu", "persistence")}
    res["garch"]["half_life_days"] = float(np.log(0.5) / np.log(fit["persistence"]))
    res["ljung_box_20"] = {}
    for label, x in [("return", r), ("abs_return", np.abs(r)), ("resid", z), ("abs_resid", np.abs(z)), ("sq_resid", z ** 2)]:
        q, p = ljung_box(x, 20)
        res["ljung_box_20"][label] = {"Q": float(q), "p": float(p), "lag1_acf": float(acf(x, 1)[1]),
                                      "mean_acf_1_20": float(np.mean(acf(x, 20)[1:]))}
    acf_absz = acf(np.abs(z), L)
    acf_z = acf(z, L)

    # (e) information the past carries about tomorrow, in bits (8 equal-probability bins = 3 bits max)
    mi_r, mi_r_base, mi_r_99 = mi_profile(r, rng=rng)
    mi_z, mi_z_base, mi_z_99 = mi_profile(z, rng=rng)
    res["mutual_info_bits"] = {
        "raw_lag1": float(mi_r[0]), "raw_sum_1_20_excess": float(np.sum(mi_r - mi_r_base)),
        "resid_lag1": float(mi_z[0]), "resid_sum_1_20_excess": float(np.sum(mi_z - mi_z_base)),
        "shuffle_mean": mi_r_base, "shuffle_p99": mi_r_99, "max_possible": 3.0,
        "raw_lags_above_p99": int(np.sum(mi_r > mi_r_99)), "resid_lags_above_p99": int(np.sum(mi_z > mi_z_99))}
    # sign-only information
    sgn_r, sgn_z = (r > 0).astype(int), (z > 0).astype(int)
    res["mutual_info_bits"]["sign_raw_lag1"] = mutual_info(sgn_r[:-1], sgn_r[1:], 2)
    res["mutual_info_bits"]["sign_resid_lag1"] = mutual_info(sgn_z[:-1], sgn_z[1:], 2)

    # (f) BDS test for ANY remaining dependence, with a simulate-and-refit null (2000-2026 subsample;
    #     the test builds n-by-n matrices, so sample length and worker count are kept modest for 8 GB RAM)
    sub = r[np.asarray(dates >= "2000")]
    sub = sub - sub.mean()
    fsub = fit_gjr(sub)
    bds_raw = bds(sub / sub.std(), max_dim=4, distance=1.0)[0]
    bds_res = bds(fsub["z"], max_dim=4, distance=1.0)[0]
    params = {k: float(fsub[k]) for k in ("omega", "alpha", "gamma", "beta", "nu")}
    B = 199
    with ProcessPoolExecutor(max_workers=3) as ex:
        boot = np.array(list(ex.map(_bds_boot_one, [(params, len(sub), 1000 + i) for i in range(B)])))
    pvals = [(1 + np.sum(np.abs(boot[:, j]) >= abs(bds_res[j]))) / (B + 1) for j in range(3)]
    res["bds_2000_2026"] = {"n": int(len(sub)), "raw_stat_dims_2_3_4": [float(x) for x in bds_raw],
                            "resid_stat_dims_2_3_4": [float(x) for x in bds_res],
                            "resid_bootstrap_p_dims_2_3_4": [float(p) for p in pvals],
                            "bootstrap_null_mean": [float(x) for x in boot.mean(0)],
                            "bootstrap_null_sd": [float(x) for x in boot.std(0)], "n_boot": B}

    # (g) a flexible learner on price history alone (stand-in for "an RNN tuned to the market")
    from sklearn.ensemble import HistGradientBoostingRegressor

    def flexible(x, label):
        s = pd.Series(x)
        feats = {f"l{k}": s.shift(k) for k in range(1, 21)}
        for w in (5, 21, 63, 252):
            feats[f"m{w}"] = s.shift(1).rolling(w).mean()
            feats[f"a{w}"] = s.abs().shift(1).rolling(w).mean()
        F = pd.DataFrame(feats)
        good = F.notna().all(1).values
        F, yv = F.values[good], x[good]
        cut = int(len(yv) * 0.6)
        out = {}
        for tname, target in [("direction", yv), ("magnitude", np.abs(yv))]:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=4,
                                              early_stopping=True, validation_fraction=0.2, random_state=0)
            m.fit(F[:cut], target[:cut])
            pr = m.predict(F[cut:])
            base = target[:cut].mean()
            out[tname] = float(1 - np.sum((target[cut:] - pr) ** 2) / np.sum((target[cut:] - base) ** 2))
        res.setdefault("flexible_learner_oos_r2", {})[label] = out

    flexible(r, "raw_returns")
    flexible(z, "garch_residuals")

    # (h) volatility persistence across the cross-section ("how close to 1 is the feedback gain?")
    ind = load_industries(ROOT / "data")
    st, _ = load_stocks(ROOT / "data")
    pers = {"industries": [], "stocks": []}
    for col in ind.columns:
        x = ind[col].dropna().values * 100
        pers["industries"].append(float(fit_gjr(x - x.mean())["persistence"]))
    stock_cols = [c for c in st.columns if st[c].notna().sum() >= 2520]
    jobs = [(st[c].dropna().values * 100) for c in stock_cols]
    with ProcessPoolExecutor(max_workers=8) as ex:
        pers["stocks"] = [float(p) for p in ex.map(_persistence, jobs, chunksize=8)]
    for k, vals in pers.items():
        a = np.array(vals)
        res[f"persistence_{k}"] = {"n": int(len(a)), "median": float(np.median(a)), "p10": float(np.quantile(a, 0.1)),
                                   "p90": float(np.quantile(a, 0.9)), "share_above_0.95": float(np.mean(a > 0.95))}

    (OUT / "test1.json").write_text(json.dumps(res, indent=2))
    np.savez(OUT / "test1.npz", acf_r=acf_r, acf_abs=acf_abs, acf_z=acf_z, acf_absz=acf_absz,
             mi_r=mi_r, mi_z=mi_z, pers_ind=np.array(pers["industries"]), pers_stk=np.array(pers["stocks"]))
    print(json.dumps(res, indent=2))


def _persistence(x):
    return fit_gjr(x - x.mean())["persistence"]


if __name__ == "__main__":
    main()
