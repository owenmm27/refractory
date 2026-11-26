"""Test 6c - Closing the two leftovers found in Test 6a/6b.

(1) Market: the two-unit model overreacts to big shocks (next-day |z| is too small after a big |z|).
    Refit with a power input |r|^d (d < 2 = saturating response) and re-check the residuals.
(2) Stocks: add a shared POPULATION unit to the real-volatility avalanche null. Each simulated day,
    yesterday's simulated idiosyncratic surprises (own sector and other sectors) scale today's volatility,
    with the strengths estimated in Test 6b. If the real day-to-day persistence of firing is now matched,
    the leftover was this population feedback.

Usage: python3 -I src/test6c_closure.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import bds

import test3_avalanches as t3
from common import fit_multi, fit_power, load_market, load_stocks, simulate_power
from test6b_timescales_stocks import KEYS, _fit, h_path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"
N_REPS = 20


def lagdep(z):
    a = np.abs(z)
    return {"corr_abs_lag1": float(np.corrcoef(a[:-1], a[1:])[0, 1]),
            "corr_signed_to_abs": float(np.corrcoef(z[:-1], a[1:])[0, 1]),
            "bds_m2": float(bds(z, max_dim=2, distance=1.0)[0])}


def _power_boot(args):
    params, n, seed = args
    x = simulate_power(params, n, np.random.default_rng(seed))
    f = fit_power(x - x.mean(), 2)
    return [lagdep(f["z"])[k] for k in ("corr_abs_lag1", "corr_signed_to_abs", "bds_m2")]


def market_part():
    mkt = load_market(ROOT / "data")
    out = {}
    for name, start in [("1926-2026", "1926"), ("2000-2026", "2000")]:
        x = mkt.loc[start:]
        r = ((x - x.mean()) * 100).values
        f2 = fit_multi(r, 2)
        p2 = fit_power(r, 2)
        p1 = fit_power(r, 1)
        out[name] = {
            "K2_square_input": {"loglik": float(f2["loglik"])} | lagdep(f2["z"]),
            "K2_power_input": {"loglik": float(p2["loglik"]), "d": float(p2["d"]),
                               "half_lives": [float(v) for v in p2["half_lives"]]} | lagdep(p2["z"]),
            "K1_power_input": {"loglik": float(p1["loglik"]), "d": float(p1["d"])} | lagdep(p1["z"]),
            "LR_power_vs_square_K2": float(2 * (p2["loglik"] - f2["loglik"]))}
        if name == "2000-2026":
            params = {k: p2[k] for k in ("omega", "a", "b", "delta", "nu", "d")}
            with ProcessPoolExecutor(max_workers=3) as ex:
                boot = np.array(list(ex.map(_power_boot, [(params, len(r), 7000 + i) for i in range(60)])))
            out[name]["power_null_bootstrap"] = {k: {"mean": float(boot[:, j].mean()), "sd": float(boot[:, j].std())}
                                                 for j, k in enumerate(("corr_abs_lag1", "corr_signed_to_abs", "bds_m2"))}
        print(name, json.dumps(out[name]), flush=True)
    return out


def _pop_rep(args):
    seed, thr, payload, coefs = args
    t3.K = thr
    mkt, betas, alphas, h_idio, nu, mask, sec_idx, n_sec, means, sds = payload
    rng = np.random.default_rng(seed)
    T, N = h_idio.shape
    eps = rng.standard_t(nu, size=(T, N)) * np.sqrt((nu - 2) / nu)   # unit-variance shocks = surprises z
    R = np.empty((T, N))
    mult = np.ones(N)
    onehot = np.zeros((N, n_sec))
    onehot[np.arange(N), sec_idx] = 1
    sec_n = onehot.sum(0)
    for t in range(T):
        R[t] = mkt[t] * betas + alphas + np.sqrt(h_idio[t]) * mult * eps[t]
        a = np.abs(eps[t])
        sec_sum = a @ onehot
        loo = (sec_sum[sec_idx] - a) / np.maximum(sec_n[sec_idx] - 1, 1)
        other = (a.sum() - sec_sum[sec_idx]) / np.maximum(N - sec_n[sec_idx], 1)
        x_sec = (loo - means[0]) / sds[0]
        x_oth = (other - means[1]) / sds[1]
        mult = np.clip(1 + coefs[0] * x_sec + coefs[1] * x_oth, 0.3, 3.0)
    R[~mask] = np.nan
    summ, _, _ = t3.summarize(t3.firing_counts(R))
    return {k: summ[k] for k in KEYS}


def stock_part(spill):
    st, _ = load_stocks(ROOT / "data")
    panel = st.loc[:, st.notna().mean() >= 0.98]
    panel = panel.loc[panel.notna().mean(axis=1) >= 0.95]
    R = panel.values * 100
    T, N = R.shape
    meta = pd.read_csv(ROOT / "data" / "meta" / "constituents.csv").set_index("Symbol")
    sectors = np.array([meta.loc[s, "GICS Sector"] for s in panel.columns])
    sec_codes, sec_idx = np.unique(sectors, return_inverse=True)
    mask = np.isfinite(R)
    mkt = np.nanmean(R, axis=1)
    betas, alphas, idio = np.empty(N), np.empty(N), np.empty_like(R)
    for i in range(N):
        ok = mask[:, i]
        bb = np.polyfit(mkt[ok], R[ok, i], 1)
        betas[i], alphas[i] = bb
        idio[:, i] = R[:, i] - (bb[1] + bb[0] * mkt)
    with ProcessPoolExecutor(max_workers=4) as ex:
        fits = list(ex.map(_fit, [(idio[:, i], 2) for i in range(N)], chunksize=6))
    h = np.column_stack([h_path(idio[:, i], fits[i]) for i in range(N)])
    nu = np.array([f["nu"] for f in fits])
    # standardization constants for the population regressors, from the real panel (as in Test 6b)
    z = np.abs(idio / np.sqrt(h))
    z[~mask] = np.nan
    valid = np.isfinite(z)
    z0 = np.where(valid, z, 0)
    onehot = np.zeros((N, len(sec_codes)))
    onehot[np.arange(N), sec_idx] = 1
    sec_sum, sec_cnt = z0 @ onehot, valid.astype(float) @ onehot
    loo = (sec_sum[:, sec_idx] - z0) / np.maximum(sec_cnt[:, sec_idx] - valid, 1)
    other = (z0.sum(1, keepdims=True) - sec_sum[:, sec_idx]) / np.maximum(valid.sum(1, keepdims=True) - sec_cnt[:, sec_idx], 1)
    means = np.array([np.nanmean(loo[valid]), np.nanmean(other[valid])])
    sds = np.array([np.nanstd(loo[valid]), np.nanstd(other[valid])])
    mz = float(spill["mean_abs_z"])
    coefs = np.array([spill["joint"]["same_sector_yesterday"]["coef_per_sd"],
                      spill["joint"]["other_sectors_yesterday"]["coef_per_sd"]]) / mz
    payload = (mkt, betas, alphas, h, nu, mask, sec_idx, len(sec_codes), means, sds)
    out = {"feedback_vol_multiplier_per_sd": {"same_sector": float(coefs[0]), "other_sectors": float(coefs[1])}}
    prev = json.loads((OUT / "test6b.json").read_text())["avalanche"]
    for thr in (3.0, 4.0, 5.0):
        with ProcessPoolExecutor(max_workers=4) as ex:
            reps = pd.DataFrame(list(ex.map(_pop_rep, [(2900 + r, thr, payload, coefs) for r in range(N_REPS)])))
        real = prev[f"K_threshold={thr:g}"]["real"]
        out[f"K_threshold={thr:g}"] = {
            "real": real, "realvol_K2": prev[f"K_threshold={thr:g}"]["realvol_K2"],
            "realvol_K2_plus_population": {k: {"mean": float(reps[k].mean()), "p2.5": float(reps[k].quantile(0.025)),
                                               "p97.5": float(reps[k].quantile(0.975)),
                                               "real_pct": float((reps[k] < real[k]).mean() + 0.5 * (reps[k] == real[k]).mean())}
                                           for k in KEYS}}
        v = out[f"K_threshold={thr:g}"]["realvol_K2_plus_population"]["lag1_slope"]
        print(f"threshold {thr}: real {real['lag1_slope']:.3f} | 2-unit {prev[f'K_threshold={thr:g}']['realvol_K2']['lag1_slope']['mean']:.3f}"
              f" | 2-unit + population {v['mean']:.3f} [{v['p2.5']:.3f}, {v['p97.5']:.3f}]", flush=True)
    return out


def main():
    spill = json.loads((OUT / "test6b.json").read_text())["spillover"]["K=2"]
    res = {"market_power_input": market_part(), "stocks_population_unit": stock_part(spill)}
    (OUT / "test6c.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
