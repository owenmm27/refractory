"""Test 9 - Does scheduled news explain the post-2000 overreaction?

Since 2000, the day after a big market surprise is calmer than the 2-unit volatility model expects
(corr(|z_t-1|, |z_t|) = -0.09; BDS -6.6). Hypothesis: scheduled announcements (FOMC decisions, CPI,
jobs reports) cause one-day variance spikes that should NOT feed the persistent volatility state.

A. Event-day variance vs the model; the overreaction split by whether yesterday was an announcement day.
B. Augmented model: h_t = level_t * (1 + kappa_type * A_type,t). The shock fed into the persistent units is
   r^2 / (1 + kappa*A)^theta: theta = 0 lets the announcement spike persist, theta = 1 strips it out.
C. Second candidate: overnight gaps (close -> open) vs trading-session moves feeding volatility with
   different weights (SPY, 1993+). CPI and jobs reports land in the overnight gap (8:30 a.m.).

Usage: python3 -I src/test9_scheduled_news.py <project_root>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import betaln, gammaln
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import bds

from common import _unpack_multi, fit_multi, load_market
from test5_options import get_spy

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
DATA, OUT = ROOT / "data", ROOT / "results"
TYPES = ["fomc", "cpi", "jobs"]


def t_ll(r, h, nu):
    # log G((nu+1)/2) - log G(nu/2) written as log G(1/2) - log B(nu/2, 1/2): identical, but stays accurate as
    # nu -> infinity (thin-tailed samples), where the difference of two huge gammaln values loses all precision
    return (gammaln(0.5) - betaln(nu / 2, 0.5) - 0.5 * np.log(np.pi * (nu - 2))
            - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(r ** 2 / (h * (nu - 2))))


def event_matrix(index):
    """T x 3 indicator of scheduled FOMC / CPI / jobs days on trading days, plus unscheduled FOMC."""
    td = pd.DatetimeIndex(index)

    def on_trading(dates):
        pos = np.searchsorted(td.values, pd.DatetimeIndex(dates).values)
        pos = pos[pos < len(td)]
        v = np.zeros(len(td), bool)
        v[pos] = True
        return v

    fomc = pd.read_csv(DATA / "macro" / "fomc_dates.csv", parse_dates=["date"])
    cpi = pd.to_datetime(open(DATA / "macro" / "cpi_release_dates.txt").read().split(), format="%Y%m%d")
    jobs = pd.to_datetime(open(DATA / "macro" / "jobs_report_release_dates.txt").read().split(), format="%Y%m%d")
    A = np.column_stack([on_trading(fomc.loc[fomc["kind"] == "scheduled", "date"]), on_trading(cpi), on_trading(jobs)])
    U = on_trading(fomc.loc[fomc["kind"] != "scheduled", "date"])
    return A.astype(float), U


def diagnostics(z, dates=None):
    a = np.abs(z)
    out = {"corr_abs_lag1": float(np.corrcoef(a[:-1], a[1:])[0, 1]),
           "spearman_abs_lag1": float(spearmanr(a[:-1], a[1:]).statistic),
           "bds_dims_2_3_4": [float(x) for x in bds(z, max_dim=4, distance=1.0)[0]]}
    if dates is not None:
        keep = ~(((dates >= "2008-09-01") & (dates <= "2009-06-30")) | ((dates >= "2020-02-15") & (dates <= "2020-06-30")))
        k2 = keep[:-1] & keep[1:]
        out["corr_abs_lag1_ex_2008_2020_crises"] = float(np.corrcoef(a[:-1][k2], a[1:][k2])[0, 1])
    return out


# ----------------------------------------------------------------------------- generalized 2-unit model
def variance_path(p, r, A, theta_mode, on2=None, id2=None, K=2):
    """theta_mode: None (no announcements), 0.0, 1.0 or 'free'. on2/id2: overnight / session squared returns
    (if given, the input is lam_on * on2 + id2 instead of r^2)."""
    nb = 1 + 2 * K + 2
    omega, a, b, delta, nu = _unpack_multi(p[:nb], K)
    j = nb
    kap = np.zeros(A.shape[1])
    theta = 0.0
    if theta_mode is not None:
        kap = np.exp(p[j:j + A.shape[1]])
        j += A.shape[1]
        if theta_mode == "free":
            theta = 1 / (1 + np.exp(-p[j]))
            j += 1
        else:
            theta = float(theta_mode)
    lam_on = None
    if on2 is not None:
        lam_on = np.exp(p[j])
        j += 1
    bump = 1 + A @ kap
    sq = r ** 2 if on2 is None else lam_on * on2 + id2
    v = r.var()
    u = np.empty_like(r)
    u[0] = v
    u[1:] = (sq[:-1] / bump[:-1] ** theta) * (1 + delta * (r[:-1] < 0)) / (1 + delta / 2)
    level = np.full(len(r), omega)
    for ak, bk in zip(a, b):
        level = level + lfilter([ak], [1.0, -bk], u, zi=[bk * ak / (1 - bk) * v])[0]
    h = level * bump
    return h, nu, {"kappa": dict(zip(TYPES, kap.tolist())), "theta": theta, "lambda_overnight": lam_on}


def fit_variant(r, A, theta_mode, on2=None, id2=None, p_base=None):
    extra = []
    if theta_mode is not None:
        extra += [np.log(0.5)] * A.shape[1]
        if theta_mode == "free":
            extra += [0.0]
    if on2 is not None:
        extra += [0.0]
    p0 = np.concatenate([p_base, extra])

    def nll(p):
        h, nu, _ = variance_path(p, r, A, theta_mode, on2, id2)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        return -t_ll(r, h, nu).sum()

    res = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 30000, "xatol": 1e-6, "fatol": 1e-7})
    res = minimize(nll, res.x, method="L-BFGS-B")
    h, nu, info = variance_path(res.x, r, A, theta_mode, on2, id2)
    return {"loglik": float(-res.fun), "n_params": int(len(res.x)), "h": h, "z": r / np.sqrt(h), "info": info}


def run(r, dates, A, U, label, on2=None, id2=None):
    """Fit the variants on one return series and report fit + residual diagnostics."""
    base = fit_multi(r, 2)
    out = {"base_2unit": {"loglik": float(base["loglik"])} | diagnostics(base["z"], dates)}
    z = base["z"]
    a = np.abs(z)
    # A. event-day variance vs model, and next-day behaviour
    ev_any = A.any(1)
    rows = {}
    for j, t in enumerate(TYPES + ["unscheduled_fomc", "any_scheduled"]):
        m = (A[:, j] > 0) if j < 3 else (U if t == "unscheduled_fomc" else ev_any)
        nxt = np.r_[False, m[:-1]]
        prv = np.r_[m[1:], False]
        rows[t] = {"n": int(m.sum()), "z2_on_day": float(np.mean(z[m] ** 2)), "z2_next_day": float(np.mean(z[nxt] ** 2)),
                   "z2_day_before": float(np.mean(z[prv] ** 2))}
    rows["all_days_z2"] = float(np.mean(z ** 2))
    out["A_event_days_vs_model"] = rows
    # overreaction split: pairs where yesterday was / wasn't an announcement day
    prev_ev = ev_any[:-1] | U[:-1]
    big = a[:-1] > np.quantile(a, 0.9)
    out["A_after_big_surprise"] = {
        "next_day_z2_if_yesterday_announcement": float(np.mean(z[1:][big & prev_ev] ** 2)),
        "n_announcement": int((big & prev_ev).sum()),
        "next_day_z2_if_yesterday_ordinary": float(np.mean(z[1:][big & ~prev_ev] ** 2)),
        "n_ordinary": int((big & ~prev_ev).sum()),
        "corr_abs_lag1_announcement_pairs": float(np.corrcoef(a[:-1][prev_ev], a[1:][prev_ev])[0, 1]),
        "corr_abs_lag1_ordinary_pairs": float(np.corrcoef(a[:-1][~prev_ev], a[1:][~prev_ev])[0, 1])}
    # B. announcement-aware models
    for name, mode in [("bump_persists", 0.0), ("bump_stripped", 1.0), ("bump_theta_free", "free")]:
        f = fit_variant(r, A, mode, p_base=base["params_vec"])
        out[name] = {"loglik": f["loglik"], "LR_vs_base": float(2 * (f["loglik"] - base["loglik"])),
                     "n_extra_params": int(f["n_params"] - len(base["params_vec"])), "info": f["info"]} | diagnostics(f["z"], dates)
    # C. overnight vs session inputs
    if on2 is not None:
        f = fit_variant(r, A, None, on2, id2, p_base=base["params_vec"])
        out["split_overnight_session"] = {"loglik": f["loglik"], "LR_vs_base": float(2 * (f["loglik"] - base["loglik"])),
                                          "info": f["info"]} | diagnostics(f["z"], dates)
        f = fit_variant(r, A, "free", on2, id2, p_base=base["params_vec"])
        out["split_plus_announcements"] = {"loglik": f["loglik"], "LR_vs_base": float(2 * (f["loglik"] - base["loglik"])),
                                           "info": f["info"]} | diagnostics(f["z"], dates)
    print(label, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("info",)})
                             for k, v in out.items()}, default=float)[:3000], flush=True)
    return out


def main():
    res = {"reference_null_from_test6": {"corr_abs_lag1": "0.00 +/- 0.012", "bds_m2": "about 0 +/- 1.0"}}
    mkt = load_market(DATA).loc["2000":]
    r = ((mkt - mkt.mean()) * 100).values
    A, U = event_matrix(mkt.index)
    res["crsp_market_2000_2026"] = run(r, mkt.index, A, U, "CRSP")

    spy = get_spy().loc["2000":]
    rs = (spy["cc"] - spy["cc"].mean()).values * 100
    on2, id2 = (spy["on"].values * 100) ** 2, (spy["id"].values * 100) ** 2
    A2, U2 = event_matrix(spy.index)
    res["spy_2000_2026"] = run(rs, spy.index, A2, U2, "SPY", on2, id2)
    res["spy_overnight_share_of_variance"] = float(np.var(spy["on"]) / (np.var(spy["on"]) + np.var(spy["id"])))
    res["spy_overnight_share_on_cpi_jobs_days"] = float(
        np.mean(spy["on"].values[A2[:, 1:].any(1)] ** 2) / np.mean(spy["cc"].values[A2[:, 1:].any(1)] ** 2))

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k not in ("h", "z")}
        return o
    (OUT / "test9.json").write_text(json.dumps(clean(res), indent=2, default=float))


if __name__ == "__main__":
    main()
