"""Test 6d - Is the leftover persistence carried by sector factors?

Null: each stock = beta * (real market factor) + gamma * (real leave-one-out sector factor)
      + its own fitted 2-unit residual volatility path * fresh i.i.d. shocks.
If this matches the real day-to-day persistence of collective firing, the leftover in Tests 3/6
was multi-factor structure (sector co-movement with its own volatility clustering), not stock-to-stock
cascades.

Usage: python3 -I src/test6d_sector_null.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import test3_avalanches as t3
from common import load_stocks
from test6b_timescales_stocks import KEYS, _fit, h_path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"
N_REPS = 20


def _rep(args):
    seed, thr, systematic, h, nu, mask = args
    t3.K = thr
    rng = np.random.default_rng(seed)
    eps = rng.standard_t(nu, size=h.shape) * np.sqrt((nu - 2) / nu)
    R = systematic + np.sqrt(h) * eps
    R[~mask] = np.nan
    summ, _, _ = t3.summarize(t3.firing_counts(R))
    return {k: summ[k] for k in KEYS}


def main():
    st, _ = load_stocks(ROOT / "data")
    panel = st.loc[:, st.notna().mean() >= 0.98]
    panel = panel.loc[panel.notna().mean(axis=1) >= 0.95]
    R = panel.values * 100
    T, N = R.shape
    meta = pd.read_csv(ROOT / "data" / "meta" / "constituents.csv").set_index("Symbol")
    sectors = np.array([meta.loc[s, "GICS Sector"] for s in panel.columns])
    codes, sidx = np.unique(sectors, return_inverse=True)
    mask = np.isfinite(R)
    mkt = np.nanmean(R, axis=1)
    idio = np.empty_like(R)
    sys1 = np.empty_like(R)
    for i in range(N):
        ok = mask[:, i]
        b = np.polyfit(mkt[ok], R[ok, i], 1)
        sys1[:, i] = b[1] + b[0] * mkt
        idio[:, i] = R[:, i] - sys1[:, i]
    # leave-one-out sector factor from the market-residuals
    I0 = np.where(mask, idio, 0.0)
    onehot = np.zeros((N, len(codes)))
    onehot[np.arange(N), sidx] = 1
    ssum, scnt = I0 @ onehot, mask.astype(float) @ onehot
    sec_loo = (ssum[:, sidx] - I0) / np.maximum(scnt[:, sidx] - mask, 1)
    resid = np.empty_like(R)
    systematic = np.empty_like(R)
    gam, r2s = np.empty(N), np.empty(N)
    for i in range(N):
        ok = mask[:, i]
        g = np.polyfit(sec_loo[ok, i], idio[ok, i], 1)
        gam[i] = g[0]
        fitted = g[1] + g[0] * sec_loo[:, i]
        resid[:, i] = idio[:, i] - fitted
        systematic[:, i] = sys1[:, i] + fitted
        r2s[i] = 1 - np.nanvar(resid[ok, i]) / np.nanvar(idio[ok, i])
    print(f"sector factor explains median {np.median(r2s):.1%} of market-residual variance", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        fits = list(ex.map(_fit, [(resid[:, i], 2) for i in range(N)], chunksize=6))
    h = np.column_stack([h_path(resid[:, i], fits[i]) for i in range(N)])
    nu = np.array([f["nu"] for f in fits])
    prev = json.loads((OUT / "test6b.json").read_text())["avalanche"]
    res = {"sector_share_of_residual_variance_median": float(np.median(r2s)), "sector_beta_median": float(np.median(gam))}
    for thr in (3.0, 4.0, 5.0):
        with ProcessPoolExecutor(max_workers=4) as ex:
            reps = pd.DataFrame(list(ex.map(_rep, [(3900 + r, thr, systematic, h, nu, mask) for r in range(N_REPS)])))
        real = prev[f"K_threshold={thr:g}"]["real"]
        res[f"K_threshold={thr:g}"] = {k: {"mean": float(reps[k].mean()), "p2.5": float(reps[k].quantile(0.025)),
                                          "p97.5": float(reps[k].quantile(0.975)),
                                          "real_pct": float((reps[k] < real[k]).mean() + 0.5 * (reps[k] == real[k]).mean())}
                                      for k in KEYS}
        v = res[f"K_threshold={thr:g}"]["lag1_slope"]
        print(f"threshold {thr}: real {real['lag1_slope']:.3f} | market+sector null {v['mean']:.3f} [{v['p2.5']:.3f}, {v['p97.5']:.3f}]", flush=True)
    (OUT / "test6d.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
