"""Test 15 - Does the refractory period survive volatility models that don't overreact to big shocks?

The obvious critique of Tests 9-14: the base model feeds volatility with SQUARED returns, which is known to
over-respond to big shocks (Harvey 2013; Andersen-Bollerslev-Diebold 2007). If the state jumps too far after a
big day, the next day looks "quieter than the state implies" without any refractory mechanism.

Here the refractory term is added, unchanged, to two other two-component baselines with the same number of
parameters (7) as the squared-return model:
  * EGARCH (Nelson 1991): the log-variance state is fed by |z| (linear in the size of the move) plus a sign term;
  * Beta-t-EGARCH (Harvey & Chakravarty 2008; Harvey 2013), the standard score-driven "robust" fix: the
    log-variance state is fed by the Student-t score u = (nu+1) r^2 / ((nu-2) e^lam + r^2) - 1, which is bounded,
    so an extreme day moves the state only a limited amount; leverage term k* sgn(-r) (u + 1) in the fast unit.
Both:  lam_t = omega + c1_t + c2_t (fast and slow units),  log h_t = lam_t - rho * (s_{t-1} - mean s),
       s = |r| / exp(lam / 2) (yesterday's surprise relative to the state; the transient never feeds the state).
The squared-return model is Test 10's (`test10_refractory_where.gpath`).

If the refractory term is an artifact of overreaction, it should vanish under Beta-t-EGARCH. A simulation null
makes this concrete: Beta-t-EGARCH paths with NO refractory term (fitted to the real data) are refitted with
each model + refractory term, to see how much spurious "refractoriness" the squared-return model would report.

Usage: python3 -I src/test15_score_driven.py <project_root> [n_sim]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import gammaln

import test10_refractory_where as T10
import test14_intl as T14
from common import _t_loglik, load_market

np.seterr(over="ignore", divide="ignore", invalid="ignore")   # bad trial parameters are rejected via 1e12
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
N_SIM = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ERAS = {"pre-2000": (None, "1999-12-31"), "2000-2012": ("2000-01-01", "2012-12-31"),
        "2013-2026": ("2013-01-01", "2026-12-31")}


# ----------------------------------------------------------------------------- recursions
@njit(cache=True)
def betat_state(r, omega, phi1, phi2, k1, k2, ks, nu):
    T = len(r)
    lam = np.empty(T)
    c1, c2 = 0.0, 0.0
    for t in range(T):
        l = min(max(omega + c1 + c2, -30.0), 30.0)
        lam[t] = l
        e2 = r[t] * r[t]
        u = (nu + 1.0) * e2 / ((nu - 2.0) * np.exp(l) + e2) - 1.0
        sg = 1.0 if r[t] < 0 else -1.0
        c1 = phi1 * c1 + k1 * u + ks * sg * (u + 1.0)
        c2 = phi2 * c2 + k2 * u
    return lam


@njit(cache=True)
def egarch_state(r, omega, phi1, phi2, a1, a2, th, eabs):
    T = len(r)
    lam = np.empty(T)
    c1, c2 = 0.0, 0.0
    for t in range(T):
        l = min(max(omega + c1 + c2, -30.0), 30.0)
        lam[t] = l
        z = r[t] * np.exp(-0.5 * l)
        g = abs(z) - eabs
        c1 = phi1 * c1 + a1 * g + th * z
        c2 = phi2 * c2 + a2 * g
    return lam


@njit(cache=True)
def betat_sim(eps, omega, phi1, phi2, k1, k2, ks, nu):
    T = len(eps)
    r = np.empty(T)
    c1, c2 = 0.0, 0.0
    for t in range(T):
        l = omega + c1 + c2
        x = np.exp(0.5 * l) * eps[t]
        r[t] = x
        e2 = x * x
        u = (nu + 1.0) * e2 / ((nu - 2.0) * np.exp(l) + e2) - 1.0
        sg = 1.0 if x < 0 else -1.0
        c1 = phi1 * c1 + k1 * u + ks * sg * (u + 1.0)
        c2 = phi2 * c2 + k2 * u
    return r


def eabs_t(nu):
    """E|z| for a unit-variance Student-t."""
    return float(np.sqrt(nu - 2) * np.exp(gammaln((nu - 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi))


def unpack(p):
    sig = lambda x: 1 / (1 + np.exp(-x))
    phi1 = 0.9999 * sig(p[1])
    phi2 = phi1 + (0.99999 - phi1) * sig(p[2])
    nu = 2.05 + np.exp(min(p[6], np.log(1000.0)))
    return p[0], phi1, phi2, p[3], p[4], p[5], nu


def logh_path(p, r, family, n_rho):
    omega, phi1, phi2, k1, k2, ks, nu = unpack(p)
    if family == "betat":
        lam = betat_state(r, omega, phi1, phi2, k1, k2, ks, nu)
    else:
        lam = egarch_state(r, omega, phi1, phi2, k1, k2, ks, eabs_t(nu))
    s = np.abs(r) * np.exp(-0.5 * lam)
    sm = float(s.mean())
    logh = lam.copy()
    if n_rho == 1:
        logh[1:] -= p[7] * (s[:-1] - sm)
    elif n_rho == 2:      # separate damping after down and up days
        dn = (r[:-1] < 0)
        logh[1:] -= np.where(dn, p[7], p[8]) * (s[:-1] - sm)
    return logh, nu, sm, lam


def nll(p, r, family, n_rho):
    logh, nu, _, _ = logh_path(p, r, family, n_rho)
    if not np.all(np.isfinite(logh)):
        return 1e12
    v = -_t_loglik(r, np.exp(logh), nu).sum()
    return v if np.isfinite(v) else 1e12


def fit_base(r, family):
    logit = lambda x: np.log(x / (1 - x))
    best = None
    k = (0.06, 0.02, 0.03) if family == "betat" else (0.10, 0.04, -0.06)
    for phi1, phi2 in [(0.90, 0.995), (0.80, 0.99), (0.95, 0.998)]:
        p0 = np.array([np.log(r.var()), logit(phi1 / 0.9999), logit((phi2 - phi1) / (0.99999 - phi1)), *k, np.log(5.0)])
        res = minimize(nll, p0, args=(r, family, 0), method="Nelder-Mead",
                       options={"maxiter": 8000, "maxfev": 12000, "xatol": 1e-6, "fatol": 1e-7})
        res = minimize(nll, res.x, args=(r, family, 0), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return finish(best, r, family, 0)


def fit_ext(r, family, base, n_rho=1):
    """Nested: one start exactly at the base solution, so LR >= 0."""
    best = None
    for x0 in ([0.0] * n_rho, [0.3] * n_rho):
        res = minimize(nll, np.r_[base["x"], x0], args=(r, family, n_rho), method="L-BFGS-B")
        res = minimize(nll, res.x, args=(r, family, n_rho), method="Nelder-Mead",
                       options={"maxiter": 6000, "maxfev": 9000, "xatol": 1e-6, "fatol": 1e-7})
        res = minimize(nll, res.x, args=(r, family, n_rho), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    f = finish(best, r, family, n_rho)
    f["LR"] = float(2 * (f["ll"] - base["ll"]))
    return f


def finish(res, r, family, n_rho):
    logh, nu, sm, lam = logh_path(res.x, r, family, n_rho)
    omega, phi1, phi2, k1, k2, ks, nu = unpack(res.x)
    z = r / np.exp(0.5 * logh)
    out = {"ll": float(-res.fun), "x": res.x, "nu": float(nu), "phi": [float(phi1), float(phi2)],
           "half_lives": [float(np.log(0.5) / np.log(phi1)), float(np.log(0.5) / np.log(phi2))],
           "k": [float(k1), float(k2), float(ks)], "s_mean": sm,
           "absz_lag1_corr": float(np.corrcoef(np.abs(z[:-1]), np.abs(z[1:]))[0, 1])}
    if n_rho == 1:
        out["rho"] = float(res.x[7])
        out["next_day_after_3sd"] = float(np.exp(-res.x[7] * (3 - sm)))
    if n_rho == 2:
        out["rho_down"], out["rho_up"] = float(res.x[7]), float(res.x[8])
        out["next_day_after_3sd_down"] = float(np.exp(-res.x[7] * (3 - sm)))
        out["next_day_after_3sd_up"] = float(np.exp(-res.x[8] * (3 - sm)))
    return out


def fit_sq(r):
    mask = np.ones(len(r), bool)
    base = T10.base_fit(r, mask)
    own = T10.extend(r, mask, base, own=True)
    zb = base["z"]
    return ({"ll": base["ll"], "absz_lag1_corr": float(np.corrcoef(np.abs(zb[:-1]), np.abs(zb[1:]))[0, 1])},
            {"ll": own["ll"], "LR": own["LR"], "rho": own["rho_own"], "next_day_after_3sd": own["next_day_after_3sd_own"],
             "absz_lag1_corr": float(np.corrcoef(np.abs(own["z"][:-1]), np.abs(own["z"][1:]))[0, 1])})


def slim(f):
    return {k: v for k, v in f.items() if k != "x"}


# ----------------------------------------------------------------------------- jobs
def series_job(args):
    name, era, x, sign_split = args
    t0 = time.time()
    r = np.asarray(x, float)
    r = r - r.mean()
    out = {"name": name, "era": era, "n": len(r)}
    sb, se = fit_sq(r)
    out["sq"] = {"base": sb, "refr": se}
    for fam in ("egarch", "betat"):
        b = fit_base(r, fam)
        e = fit_ext(r, fam, b)
        out[fam] = {"base": slim(b), "refr": slim(e)}
        if sign_split:
            out[fam]["refr_by_sign"] = slim(fit_ext(r, fam, b, n_rho=2))
            out[fam]["refr_by_sign"]["LR_sign_vs_one_rho"] = out[fam]["refr_by_sign"].pop("LR") - e["LR"]
        if fam == "betat":
            out["_betat_base_x"] = b["x"].tolist()
    out["secs"] = round(time.time() - t0, 1)
    print(f"{name:22s} {era:9s} n={len(r):5d}  next-day after 3SD: sq {se['next_day_after_3sd']:.2f} (LR {se['LR']:.0f})"
          f"  egarch {out['egarch']['refr']['next_day_after_3sd']:.2f} (LR {out['egarch']['refr']['LR']:.0f})"
          f"  betat {out['betat']['refr']['next_day_after_3sd']:.2f} (LR {out['betat']['refr']['LR']:.0f})"
          f"  | base ll betat-sq {out['betat']['base']['ll'] - sb['ll']:+.0f}  [{out['secs']:.0f}s]", flush=True)
    return out


def sim_job(args):
    """Simulate Beta-t-EGARCH (no refractory term), refit squared-return and Beta-t models with the term."""
    x_betat, T, seed = args
    rng = np.random.default_rng(seed)
    omega, phi1, phi2, k1, k2, ks, nu = unpack(np.asarray(x_betat))
    eps = rng.standard_t(nu, T + 1000) * np.sqrt((nu - 2) / nu)
    y = betat_sim(eps, omega, phi1, phi2, k1, k2, ks, nu)[1000:]
    y = y - y.mean()
    _, se = fit_sq(y)
    b = fit_base(y, "betat")
    e = fit_ext(y, "betat", b)
    return {"seed": seed, "sq_next_day": se["next_day_after_3sd"], "sq_LR": se["LR"],
            "betat_next_day": e["next_day_after_3sd"], "betat_LR": e["LR"]}


def main():
    t0 = time.time()
    mkt = load_market(ROOT / "data") * 100
    jobs = [("CRSP market", "2000-2026", mkt.loc["2000":].values, True),
            ("CRSP market", "1926-1999", mkt.loc[:"1999"].values, False)]
    for code in T14.NAMES:
        r, _ = T14.load(code)
        for era, (a, b) in ERAS.items():
            sub = r.loc[a:b]
            if len(sub) >= 2000:
                jobs.append((code, era, sub.values, code == "GSPC" and era == "2000-2012"))
    print(f"{len(jobs)} series", flush=True)
    with ProcessPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(series_job, jobs, chunksize=1))
    print(f"fits done in {time.time() - t0:.0f}s", flush=True)

    # simulation null: Beta-t-EGARCH fitted to CRSP 2000-2026, no refractory term
    crsp = rows[0]
    sims = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for o in ex.map(sim_job, [(crsp["_betat_base_x"], crsp["n"], 1000 + i) for i in range(N_SIM)], chunksize=1):
            sims.append(o)
            if len(sims) % 10 == 0:
                print(f"  sim {len(sims)}/{N_SIM}", flush=True)
    sims = pd.DataFrame(sims)

    flat = []
    for o in rows:
        row = {"name": o["name"], "era": o["era"], "n": o["n"]}
        for fam in ("sq", "egarch", "betat"):
            row[f"{fam}_base_ll"] = o[fam]["base"]["ll"]
            row[f"{fam}_base_absz_lag1"] = o[fam]["base"]["absz_lag1_corr"]
            row[f"{fam}_LR"] = o[fam]["refr"]["LR"]
            row[f"{fam}_next_day"] = o[fam]["refr"]["next_day_after_3sd"]
            row[f"{fam}_refr_absz_lag1"] = o[fam]["refr"]["absz_lag1_corr"]
        flat.append(row)
    df = pd.DataFrame(flat)
    df.to_csv(ROOT / "results" / "test15_score_driven.csv", index=False)
    summary = {"by_era": {}}
    intl = df[df["name"] != "CRSP market"]
    for era in ERAS:
        s = intl[intl["era"] == era]
        summary["by_era"][era] = {"n_markets": len(s), **{
            f"{fam}_{k}": v for fam in ("sq", "egarch", "betat") for k, v in {
                "median_next_day": float(s[f"{fam}_next_day"].median()),
                "n_LR_gt_3.84": int((s[f"{fam}_LR"] > 3.84).sum()),
                "n_LR_gt_10.8": int((s[f"{fam}_LR"] > 10.83).sum()),
                "median_base_absz_lag1": float(s[f"{fam}_base_absz_lag1"].median())}.items()},
            "n_betat_beats_sq": int((s["betat_base_ll"] > s["sq_base_ll"]).sum()),
            "median_ll_gain_betat_vs_sq": float((s["betat_base_ll"] - s["sq_base_ll"]).median())}
    summary["sim_null_betat_dgp"] = {
        "n": len(sims), "dgp": "Beta-t-EGARCH fitted to CRSP 2000-2026, no refractory term",
        **{f"{c}_{q}": float(sims[c].quantile(v)) for c in ("sq_next_day", "betat_next_day", "sq_LR", "betat_LR")
           for q, v in (("p05", 0.05), ("median", 0.5), ("p95", 0.95))},
        "real_sq_next_day": crsp["sq"]["refr"]["next_day_after_3sd"], "real_betat_next_day": crsp["betat"]["refr"]["next_day_after_3sd"],
        "real_sq_LR": crsp["sq"]["refr"]["LR"], "real_betat_LR": crsp["betat"]["refr"]["LR"]}
    summary["elapsed_s"] = round(time.time() - t0)
    for o in rows:
        o.pop("_betat_base_x", None)
    (ROOT / "results" / "test15.json").write_text(json.dumps({"summary": summary, "rows": rows, "sims": sims.to_dict("records")},
                                                            indent=2, default=float))
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
