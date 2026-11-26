"""Test 9b - A one-day 'refractory period' in market volatility?

After a big surprise, is the NEXT day damped while the persistent state still absorbs the shock?
  refractory: h_t = level_t * exp(-rho * (s_{t-1} - mean s)), s = |r| / sqrt(level) (yesterday's surprise)
  delayed:    the shock enters the persistent units partly with a one-day delay (weight psi)
Compared with the 2-unit base on CRSP 2000-2026, SPY 2000-2026 and CRSP 1926-1999 (pre-2000 placebo).

Usage: python3 -I src/test9b_refractory.py <project_root>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from common import _unpack_multi, acf, fit_multi, load_market
from test5_options import get_spy
from test9_scheduled_news import diagnostics, t_ll

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()


def path(p, r, mode):
    omega, a, b, delta, nu = _unpack_multi(p[:7], 2)
    v = r.var()
    sq = r ** 2 * (1 + delta * (r < 0)) / (1 + delta / 2)
    if mode == "delayed":
        psi = 1 / (1 + np.exp(-p[7]))
        sq = (1 - psi) * sq + psi * np.r_[v, sq[:-1]]
    u = np.r_[v, sq[:-1]]
    level = np.full(len(r), omega)
    for ak, bk in zip(a, b):
        level = level + lfilter([ak], [1.0, -bk], u, zi=[bk * ak / (1 - bk) * v])[0]
    h = level
    extra = {}
    if mode == "refractory":
        rho = p[7]
        s = np.abs(r) / np.sqrt(level)
        damp = np.exp(-rho * (np.r_[s.mean(), s[:-1]] - s.mean()))
        h = level * damp
        extra = {"rho": float(rho), "next_day_variance_after_3sd_surprise": float(np.exp(-rho * (3 - s.mean())))}
    if mode == "delayed":
        extra = {"psi_share_entering_one_day_late": float(psi)}
    return h, nu, extra


def fit(r, mode, p0):
    def nll(p):
        h, nu, _ = path(p, r, mode)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        return -t_ll(r, h, nu).sum()
    best = None
    for x0 in ([0.1], [0.4], [-0.1]) if mode == "refractory" else ([0.0], [-2.0], [2.0]):
        res = minimize(nll, np.r_[p0, x0], method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 30000, "xatol": 1e-6, "fatol": 1e-7})
        res = minimize(nll, res.x, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    h, nu, extra = path(best.x, r, mode)
    return -best.fun, r / np.sqrt(h), extra


def main():
    out = {}
    mkt = load_market(ROOT / "data")
    spy = get_spy()
    series = {"CRSP 2000-2026": (mkt.loc["2000":], mkt.loc["2000":].index),
              "SPY 2000-2026": (spy["cc"].loc["2000":], spy.loc["2000":].index),
              "CRSP 1926-1999 (placebo era)": (mkt.loc[:"1999"], mkt.loc[:"1999"].index)}
    for name, (x, idx) in series.items():
        r = ((x - x.mean()) * 100).values
        base = fit_multi(r, 2)
        a = acf(np.abs(base["z"]), 10)
        row = {"base": {"loglik": float(base["loglik"]), "acf_abs_z_lags_1_10": [round(float(v), 3) for v in a[1:]]} | diagnostics(base["z"], idx)}
        for mode in ("refractory", "delayed"):
            ll, z, extra = fit(r, mode, base["params_vec"])
            row[mode] = {"loglik": float(ll), "LR_vs_base": float(2 * (ll - base["loglik"])), **extra,
                         "acf_abs_z_lags_1_5": [round(float(v), 3) for v in acf(np.abs(z), 5)[1:]]} | diagnostics(z, idx)
        out[name] = row
        print(name, json.dumps(row), flush=True)
    (ROOT / "results" / "test9b.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
