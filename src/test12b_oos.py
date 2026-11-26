"""Test 12b - Out of sample: fit every piece of the market-brain model on 2000-2012, forecast 2013-2026.

One-day-ahead predictive log-likelihood (Student-t) on 2013-2026, parameters frozen at their 2000-2012 values:
  market: GJR-GARCH -> 2-unit state -> + refractory -> + reversal (the full market unit)
  stocks (idiosyncratic) and sectors: 2-unit state -> + earnings bump -> + market spillover (full unit)
Differences are tested with HAC t-statistics on the daily log-likelihood gains (Diebold-Mariano style).

Usage: python3 -I src/test12b_oos.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

import test12_brain_model as B
import test12c_model_v2 as C
from common import _unpack_multi, fit_gjr, fit_multi, gjr_variance
from test9_scheduled_news import t_ll

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
SPLIT = "2013-01-01"


def hac_t(d, lags=10):
    d = d[np.isfinite(d)]
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[0] / r.bse[0])


def mkt_path_fixed(p, m, K, refr, mean, sbar, v):
    """As B.mkt_path, but with the centering constant and start value fixed from the training sample."""
    nb = B.nbase(K)
    omega, a, b, delta, nu = _unpack_multi(p[:nb], K)
    j = nb
    rho = p[j] if refr else 0.0
    j += int(refr)
    a1, a2 = (p[j], p[j + 1]) if mean else (0.0, 0.0)
    e = m - B.mkt_mean(m, a1, a2)
    sq = e ** 2 * (1 + delta * (e < 0)) / (1 + delta / 2)
    level = B.filt(np.r_[v, sq[:-1]], a, b, v, omega)
    s = np.abs(e) / np.sqrt(level)
    h = level * np.exp(-rho * (np.r_[sbar, s[:-1]] - sbar))
    return e, h, nu, s


def fit_market_vec(m, K=2, refr=True, mean=True):
    f0 = fit_multi(m, K)

    def nll(p):
        e, h, nu, _, _ = B.mkt_path(p, m, K, refr, mean)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        return -t_ll(e, h, nu).sum()
    best = None
    for r0 in ([0.0, 0.3] if refr else [None]):
        p0 = np.r_[f0["params_vec"], [r0] if refr else [], [0.0, 0.0] if mean else []]
        res = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 30000, "xatol": 1e-7, "fatol": 1e-8})
        res = minimize(nll, res.x, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    _, _, _, s, sbar = B.mkt_path(best.x, m, K, refr, mean)
    e = B.mkt_path(best.x, m, K, refr, mean)[0]
    return best.x, sbar, float(np.mean(e ** 2))


def unit_oos(args):
    """One stock (or sector): fit three unit variants on the training span, return test-span daily log-liks."""
    x, mask, m2, smc, E, ntr = args
    earn = E[:ntr].sum() > 10
    tr = np.zeros(len(x), bool)
    tr[:ntr] = True
    out = {}
    specs = {"base": (False, False), "earnings": (False, earn), "full": (True, earn)}
    for name, (spill, ea) in specs.items():
        mtr = mask & tr
        f0 = fit_multi(x[mtr], 2)
        p0 = np.r_[f0["params_vec"], [0.0, 0.0] if spill else [], [0.0] if ea else []]
        bounds = [(None, None)] * 7 + ([(0.0, 5.0), (-1.5, 1.5)] if spill else []) + ([(-6.0, 4.0)] if ea else [])

        def nll(p):
            h, nu, _ = B.unit_path(p, x[:ntr], mask[:ntr], m2[:ntr], smc[:ntr], E[:ntr], spill, ea)
            if not np.all(np.isfinite(h)) or h.min() <= 0:
                return 1e12
            return -t_ll(x[:ntr][mask[:ntr]], h[mask[:ntr]], nu).sum()
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        h, nu, _ = B.unit_path(res.x, x, mask, m2, smc, E, spill, ea)
        ll = np.where(mask, t_ll(x, h, nu), np.nan)
        out[name] = ll[ntr:]
    return out


def main():
    panel, m_s, sectors, E = B.load_panel()
    R, m = panel.values, m_s.values
    mc = m - m[:np.searchsorted(panel.index, np.datetime64(SPLIT))].mean()
    ntr = int(np.searchsorted(panel.index, np.datetime64(SPLIT)))
    res = {"train": [str(panel.index[0].date()), str(panel.index[ntr - 1].date())],
           "test": [str(panel.index[ntr].date()), str(panel.index[-1].date())]}
    # ---------------- market
    mtr = mc[:ntr]
    ll = {}
    g = fit_gjr(mtr)
    hg = gjr_variance(mc, g["omega"], g["alpha"], g["gamma"], g["beta"], mtr.var())
    ll["gjr_garch"] = t_ll(mc, hg, g["nu"])[ntr:]
    for name, (refr, mean) in {"two_unit": (False, False), "two_unit_refractory": (True, False),
                               "full_market_unit": (True, True)}.items():
        p, sbar, v = fit_market_vec(mtr, 2, refr, mean)
        e, h, nu, s = mkt_path_fixed(p, mc, 2, refr, mean, sbar, v)
        ll[name] = t_ll(e, h, nu)[ntr:]
        if name == "full_market_unit":
            smc = np.r_[0.0, s[:-1] - sbar]
    # v2 market unit: reversal strength keyed to trailing 21-day volatility (constants frozen at training values)
    f2 = C.fit_market2(mtr)
    lr = C.regime(mc)
    om, a_, b_, d_, nu2 = f2["omega"], f2["a"], f2["b"], f2["delta"], f2["nu"]
    e2 = mc - B.mkt_mean(mc, f2["a1"], f2["a2"]) * np.exp(f2["c"] * np.minimum(lr - f2["Rbar"], f2["Rmax"]))
    lvl2 = B.filt(np.r_[f2["vbar"], (e2 ** 2 * (1 + d_ * (e2 < 0)) / (1 + d_ / 2))[:-1]], a_, b_, f2["vbar"], om)
    s2 = np.abs(e2) / np.sqrt(lvl2)
    h2 = lvl2 * np.exp(-f2["rho"] * (np.r_[f2["sbar"], s2[:-1]] - f2["sbar"]))
    ll["full_market_unit_v2"] = t_ll(e2, h2, nu2)[ntr:]
    names = list(ll)
    res["market"] = {"loglik_test": {k: float(v.sum()) for k, v in ll.items()},
                     "step_gain": {f"{names[i]} -> {names[i + 1]}": {"loglik_gain": float((ll[names[i + 1]] - ll[names[i]]).sum()),
                                                                    "hac_t": hac_t(ll[names[i + 1]] - ll[names[i]])}
                                   for i in range(len(names) - 1)}}
    print("market", json.dumps(res["market"]), flush=True)
    # ---------------- sectors and stocks
    secs = sorted(set(sectors))
    sidx = np.array([secs.index(s) for s in sectors])
    F = np.empty((len(m), len(secs)))
    for k in range(len(secs)):
        ew = np.nanmean(R[:, sidx == k], axis=1)
        bb = np.polyfit(mc[:ntr], ew[:ntr], 1)
        F[:, k] = ew - np.polyval(bb, mc)
    m2 = mc ** 2
    jobs = [(F[:, k], np.ones(len(m), bool), m2, smc, np.zeros(len(m)), ntr) for k in range(len(secs))]
    for i in range(R.shape[1]):
        mask = np.isfinite(R[:, i])
        X = np.column_stack([np.ones(len(m)), mc, F[:, sidx[i]]])
        trm = mask.copy()
        trm[ntr:] = False
        coef = np.linalg.lstsq(X[trm], R[trm, i], rcond=None)[0]
        jobs.append((np.where(mask, R[:, i] - X @ coef, 0.0), mask, m2, smc, E[:, i], ntr))
    with ProcessPoolExecutor(max_workers=3) as ex:
        outs = list(ex.map(unit_oos, jobs, chunksize=4))
    for label, sl in [("sectors", slice(0, len(secs))), ("stocks", slice(len(secs), None))]:
        part = outs[sl]
        r = {}
        for a, b in [("base", "earnings"), ("earnings", "full"), ("base", "full")]:
            d = np.array([o[b] - o[a] for o in part])                  # units x test days
            daily = np.nansum(d, axis=0)
            per_unit = np.nansum(d, axis=1)
            r[f"{a} -> {b}"] = {"loglik_gain_total": float(np.nansum(d)), "hac_t_daily_sum": hac_t(daily),
                                "share_units_improved": float(np.mean(per_unit > 0)), "n_units": len(part)}
        res[label] = r
        print(label, json.dumps(r), flush=True)
    (ROOT / "results" / "test12b.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
