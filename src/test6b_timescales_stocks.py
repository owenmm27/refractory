"""Test 6b - Does a two-timescale volatility state explain the leftover avalanche persistence,
and is there cross-stock (network) information beyond each stock's own state?

C1. Avalanche nulls rebuilt with each stock's REAL fitted volatility path from the 1-unit and the
    2-unit (fast + slow) model, at firing thresholds 3, 4, 5. Plus a fully simulated 2-unit null.
C2. Spillover regression. Each stock's standardized idiosyncratic surprise |z_it| should be
    unpredictable if its own state is complete. Regress it on yesterday's surprises of its own,
    of other stocks in the same GICS sector, of stocks in other sectors, and of the market factor.
    Driscoll-Kraay standard errors (robust to same-day and serial correlation).

Usage: python3 -I src/test6b_timescales_stocks.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import test3_avalanches as t3
from common import fit_multi, load_stocks, multi_variance, simulate_multi

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"
THRESHOLDS = (3.0, 4.0, 5.0)
N_REPS = 20
KEYS = ["silent_share", "mean_firing", "lag1_slope", "branching_m", "mr_b", "tau_size", "alpha_dur", "gamma_fit",
        "size_max", "dur_max"]


def _fit(args):
    x, K = args
    x = x[np.isfinite(x)]
    f = fit_multi(x - x.mean(), K)
    return {k: (f[k].tolist() if isinstance(f[k], np.ndarray) else float(f[k]))
            for k in ("omega", "a", "b", "delta", "nu", "half_lives", "W")}


def h_path(series, f):
    e = np.nan_to_num(series - np.nanmean(series))
    return multi_variance(e, f["omega"], np.array(f["a"]), np.array(f["b"]), f["delta"], np.nanvar(series))


def _rep(args):
    kind, seed, thr, payload = args
    t3.K = thr
    rng = np.random.default_rng(seed)
    if kind == "realvol":
        mkt, betas, alphas, h_idio, nu, mask = payload
        eps = rng.standard_t(nu, size=h_idio.shape) * np.sqrt((nu - 2) / nu)
        R = mkt[:, None] * betas[None, :] + alphas[None, :] + np.sqrt(h_idio) * eps
        R[~mask] = np.nan
    else:  # fully simulated 2-unit market + idiosyncratic
        T, betas, mpar, ipar = payload
        m = simulate_multi(mpar, T, 1, rng)[:, 0]
        e = simulate_multi(ipar, T, len(betas), rng)
        R = m[:, None] * betas[None, :] + e
    summ, _, _ = t3.summarize(t3.firing_counts(R))
    return {k: summ[k] for k in KEYS}


def driscoll_kraay(y, X, time_id, lags=10):
    """Pooled OLS with Driscoll-Kraay covariance. X includes the constant."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    g = pd.DataFrame(X * e[:, None]).groupby(time_id).sum().values  # T x k moment sums
    S = g.T @ g
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = g[l:].T @ g[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.diag(V))


def spillover(z_idio, z_mkt, sectors, dates, label):
    """z_idio: T x N standardized idiosyncratic surprises; z_mkt: T market surprises."""
    A = np.abs(z_idio)
    valid = np.isfinite(A)
    A0 = np.where(valid, A, 0.0)
    T, N = A.shape
    sec_codes, sec_idx = np.unique(sectors, return_inverse=True)
    tot_sum, tot_cnt = A0.sum(1), valid.sum(1)
    sec_sum = np.stack([A0[:, sec_idx == s].sum(1) for s in range(len(sec_codes))], 1)
    sec_cnt = np.stack([valid[:, sec_idx == s].sum(1) for s in range(len(sec_codes))], 1)
    own_sec_sum, own_sec_cnt = sec_sum[:, sec_idx], sec_cnt[:, sec_idx]          # T x N
    loo_sector = (own_sec_sum - A0) / np.maximum(own_sec_cnt - valid, 1)
    other_sector = (tot_sum[:, None] - own_sec_sum) / np.maximum(tot_cnt[:, None] - own_sec_cnt, 1)
    lag = lambda M: np.vstack([np.full((1,) + M.shape[1:], np.nan), M[:-1]])
    own_l, sec_l, oth_l = lag(A), lag(loo_sector), lag(other_sector)
    mkt_l = np.repeat(lag(np.abs(z_mkt)[:, None]), N, axis=1)
    ok = valid & np.isfinite(own_l) & np.isfinite(sec_l) & np.isfinite(oth_l) & np.isfinite(mkt_l)
    tid = np.repeat(np.arange(T)[:, None], N, axis=1)[ok]
    y = A[ok]
    raw = np.column_stack([own_l[ok], sec_l[ok], oth_l[ok], mkt_l[ok]])
    names = ["own_yesterday", "same_sector_yesterday", "other_sectors_yesterday", "market_factor_yesterday"]
    Xs = (raw - raw.mean(0)) / raw.std(0)
    out = {"n_obs": int(len(y)), "mean_abs_z": float(y.mean()), "regressor_corr": np.corrcoef(raw, rowvar=False).round(3).tolist()}
    for spec, cols in [("joint", [0, 1, 2, 3]), ("own_only", [0]), ("population_only", [1, 2, 3])]:
        X = np.column_stack([np.ones(len(y)), Xs[:, cols]])
        b, se = driscoll_kraay(y, X, tid)
        out[spec] = {names[c]: {"coef_per_sd": float(b[i + 1]), "t": float(b[i + 1] / se[i + 1]),
                                "pct_of_mean_vol": float(100 * b[i + 1] / y.mean())} for i, c in enumerate(cols)}
        resid = y - X @ b
        out[spec]["r2"] = float(1 - resid.var() / y.var())
    # firing version: probability of a >2.5 sd idiosyncratic surprise
    yf = (y > 2.5).astype(float)
    X = np.column_stack([np.ones(len(y)), Xs])
    b, se = driscoll_kraay(yf, X, tid)
    out["firing_lpm"] = {"base_rate": float(yf.mean())} | {
        names[i]: {"pp_per_sd": float(100 * b[i + 1]), "t": float(b[i + 1] / se[i + 1])} for i in range(4)}
    # split by period
    for pname, (a, bnd) in {"1990-2007": ("1990", "2007-12-31"), "2008-2026": ("2008", "2026-12-31")}.items():
        tmask = np.asarray((dates >= a) & (dates <= bnd))[tid]
        X = np.column_stack([np.ones(tmask.sum()), Xs[tmask]])
        b, se = driscoll_kraay(y[tmask], X, tid[tmask])
        out[f"joint_{pname}"] = {names[i]: {"pct_of_mean_vol": float(100 * b[i + 1] / y[tmask].mean()), "t": float(b[i + 1] / se[i + 1])}
                                 for i in range(4)}
    print(label, json.dumps({k: v for k, v in out.items() if k in ("joint", "firing_lpm")}), flush=True)
    return out


def main():
    st, _ = load_stocks(ROOT / "data")
    panel = st.loc[:, st.notna().mean() >= 0.98]
    panel = panel.loc[panel.notna().mean(axis=1) >= 0.95]
    R = panel.values * 100
    T, N = R.shape
    dates = panel.index
    meta = pd.read_csv(ROOT / "data" / "meta" / "constituents.csv").set_index("Symbol")
    sectors = np.array([meta.loc[s, "GICS Sector"] for s in panel.columns])
    mask = np.isfinite(R)
    mkt = np.nanmean(R, axis=1)
    betas, alphas, idio = np.empty(N), np.empty(N), np.empty_like(R)
    for i in range(N):
        ok = mask[:, i]
        bb = np.polyfit(mkt[ok], R[ok, i], 1)
        betas[i], alphas[i] = bb
        idio[:, i] = R[:, i] - (bb[1] + bb[0] * mkt)
    print(f"panel {T} x {N}; fitting volatility models...", flush=True)

    fits = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for K in (1, 2):
            fits[K] = list(ex.map(_fit, [(idio[:, i], K) for i in range(N)], chunksize=6))
    mfit = {K: _fit((mkt, K)) for K in (1, 2)}
    res = {"n_stocks": N, "n_days": T, "sample": [str(dates[0].date()), str(dates[-1].date())],
           "market_factor_half_lives": {f"K={K}": mfit[K]["half_lives"] for K in (1, 2)},
           "idio_half_lives_median": {"fast": float(np.median([f["half_lives"][0] for f in fits[2]])),
                                      "slow": float(np.median([f["half_lives"][1] for f in fits[2]])),
                                      "single_unit": float(np.median([f["half_lives"][0] for f in fits[1]]))},
           "idio_fast_loading_share_median": float(np.median([f["a"][0] / (1 - f["b"][0]) / f["W"] for f in fits[2]]))}
    h = {K: np.column_stack([h_path(idio[:, i], fits[K][i]) for i in range(N)]) for K in (1, 2)}
    h_m = {K: h_path(mkt, mfit[K]) for K in (1, 2)}
    nu = {K: np.array([f["nu"] for f in fits[K]]) for K in (1, 2)}

    # ---------------- C1. avalanche nulls
    res["avalanche"] = {}
    for thr in THRESHOLDS:
        t3.K = thr
        real, _, _ = t3.summarize(t3.firing_counts(R))
        row = {"real": {k: real[k] for k in KEYS}}
        jobs = {f"realvol_K{K}": [("realvol", 900 + r, thr, (mkt, betas, alphas, h[K], nu[K], mask)) for r in range(N_REPS)] for K in (1, 2)}
        ipar = {k: np.array([f[k] for f in fits[2]]) for k in ("omega", "a", "b", "delta", "nu")}
        mpar = {k: np.array(mfit[2][k]) for k in ("omega", "a", "b", "delta", "nu")}
        jobs["simulated_K2"] = [("sim", 1900 + r, thr, (T, betas, mpar, ipar)) for r in range(N_REPS)]
        for name, jl in jobs.items():
            with ProcessPoolExecutor(max_workers=4) as ex:
                reps = pd.DataFrame(list(ex.map(_rep, jl)))
            row[name] = {k: {"mean": float(reps[k].mean()), "p2.5": float(reps[k].quantile(0.025)),
                             "p97.5": float(reps[k].quantile(0.975)),
                             "real_pct": float((reps[k] < real[k]).mean() + 0.5 * (reps[k] == real[k]).mean())}
                         for k in KEYS}
        res["avalanche"][f"K_threshold={thr:g}"] = row
        print(f"threshold {thr}: lag1 real {real['lag1_slope']:.3f} | " + " | ".join(
            f"{n} {row[n]['lag1_slope']['mean']:.3f} [{row[n]['lag1_slope']['p2.5']:.3f},{row[n]['lag1_slope']['p97.5']:.3f}]"
            for n in jobs), flush=True)

    # ---------------- C2. spillover regressions
    res["spillover"] = {}
    for K in (1, 2):
        z = idio / np.sqrt(h[K])
        z[~mask] = np.nan
        zm = mkt / np.sqrt(h_m[K])
        res["spillover"][f"K={K}"] = spillover(z, zm, sectors, dates, f"K={K}")

    (OUT / "test6b.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("avalanche", "spillover")}, indent=1))


if __name__ == "__main__":
    main()
