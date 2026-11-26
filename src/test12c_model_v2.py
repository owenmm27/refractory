"""Test 12c - Market-brain model, version 2: add the two pieces version 1 was missing.

Version 1 (test12) missed two groups of patterns, and direct checks pointed to two missing mechanisms:
  1. Reversal that strengthens in turbulent times: real lag-1 autocorrelation is -0.11 when trailing volatility is
     high and -0.04 when low; v1 has the same reversal in both.
     -> market mean: mu_t = (a1*m_{t-1} + a2*sum m_{t-21..t-2}) * exp(c * (log level_t - mean))
  2. A shared "gain" on stocks' own volatility: the monthly common component of idiosyncratic volatility varies
     about twice as much in real data (sd of log 0.70) as in v1 simulations (0.26-0.40).
     -> stock variance h_i,t = (own state) x G_t, with log G_t = beta*(log market level_t - mean) + x_fast + x_slow + w
        x_fast, x_slow: AR(1) components (fitted to the autocovariance of the measured common component, after
        removing cross-sectional measurement noise estimated by bootstrap); w: white daily part.
        Stocks are refitted on de-gained residuals e / sqrt(G_measured).
Variants simulated (20 paths each): v1 full, v2 reversal only, v2 gain only, v2 full. Scorecard = test12's plus
5 new patterns (reversal by volatility regime; common idiosyncratic volatility size, persistence, market link).

Usage: python3 -I src/test12c_model_v2.py <project_root> [n_paths]
"""
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

import test12_brain_model as B
from common import _unpack_multi, acf, fit_multi
from test9_scheduled_news import t_ll

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
N_PATHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUT = ROOT / "results"
CACHE2 = OUT / "cache" / "test12c_fit.pkl"
rng0 = np.random.default_rng(12)


# ----------------------------------------------------------------------------- market v2
def regime(m):
    """log of the trailing 21-day mean of m^2 (days t-21..t-1): an observable volatility gauge."""
    sq = m ** 2
    cs = np.r_[0.0, np.cumsum(sq)]
    t = np.arange(len(m))
    n = np.minimum(t, 21)
    tr = (cs[t] - cs[np.maximum(t - 21, 0)]) / np.maximum(n, 1)
    tr[0] = sq.mean()
    return np.log(tr)


def mkt2_path(p, m, statedep=True):
    """v1 market unit (state driven by surprises e) + reversal whose strength scales with the volatility gauge."""
    omega, a, b, delta, nu = _unpack_multi(p[:7], 2)
    rho, a1, a2 = p[7], p[8], p[9]
    c = p[10] if statedep else 0.0
    lr = regime(m)
    Rbar = float(lr.mean())
    e = m - B.mkt_mean(m, a1, a2) * np.exp(c * (lr - Rbar))
    v = np.mean(e ** 2)
    sq = e ** 2 * (1 + delta * (e < 0)) / (1 + delta / 2)
    level = B.filt(np.r_[v, sq[:-1]], a, b, v, omega)
    s = np.abs(e) / np.sqrt(level)
    sbar = float(s.mean())
    h = level * np.exp(-rho * (np.r_[sbar, s[:-1]] - sbar))
    return e, h, nu, s, sbar, level, Rbar, lr


def fit_market2(m, statedep=True):
    f0 = fit_multi(m, 2)

    def nll(p):
        e, h, nu, *_ = mkt2_path(p, m, statedep)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        return -t_ll(e, h, nu).sum()
    best = None
    for c0 in ([0.0, 0.5] if statedep else [None]):
        p0 = np.r_[f0["params_vec"], 0.28, -0.04, -0.003, [c0] if statedep else []]
        res = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 30000, "maxfev": 40000, "xatol": 1e-7, "fatol": 1e-8})
        res = minimize(nll, res.x, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    p = best.x
    omega, a, b, delta, nu = _unpack_multi(p[:7], 2)
    e, h, nu, s, sbar, level, Rbar, lr = mkt2_path(p, m, statedep)
    return {"omega": float(omega), "a": a, "b": b, "delta": float(delta), "nu": float(nu), "rho": float(p[7]),
            "a1": float(p[8]), "a2": float(p[9]), "c": float(p[10]) if statedep else 0.0, "sbar": sbar,
            "Lbar": float(np.log(level).mean()), "Rbar": Rbar, "Rmax": float(lr.max() - Rbar),
            "vbar": float(np.mean(e ** 2)), "ll": float(-best.fun), "s": s, "level": level}


# ----------------------------------------------------------------------------- common gain
def measure_gain(e, mask):
    """Daily log of the cross-sectional median of e^2 / own variance, centered; plus bootstrap measurement noise."""
    var = np.nanvar(np.where(mask, e, np.nan), axis=0)
    q = np.where(mask, e ** 2 / var, np.nan)
    g = np.log(np.nanmedian(q, axis=1))
    g = g - g.mean()
    days = rng0.choice(len(g), 300, replace=False)
    noise = []
    for t in days:
        row = q[t][np.isfinite(q[t])]
        bs = [np.log(np.median(rng0.choice(row, len(row)))) for _ in range(200)]
        noise.append(np.var(bs))
    return g, float(np.mean(noise))


def fit_gain_dynamics(g, logL, Lbar, s2_meas, maxlag=250):
    X = logL - Lbar
    beta = float(np.polyfit(X, g, 1)[0])
    r = g - beta * X
    r = r - r.mean()
    ac = np.array([np.mean(r[k:] * r[:len(r) - k]) for k in range(maxlag + 1)])
    lags = np.arange(1, maxlag + 1)

    def resid(th):
        vf, phf, vs, phs = th
        return (vf * phf ** lags + vs * phs ** lags - ac[1:]) / ac[1]
    sol = least_squares(resid, x0=[ac[1] * 0.5, 0.95, ac[1] * 0.5, 0.997],
                        bounds=([0, 0.5, 0, 0.9], [10, 0.995, 10, 0.99995]))
    vf, phf, vs, phs = sol.x
    vw = max(ac[0] - vf - vs - s2_meas, 0.0)
    return {"beta_market": beta, "var_fast": float(vf), "phi_fast": float(phf), "var_slow": float(vs),
            "phi_slow": float(phs), "var_white": float(vw), "var_meas_noise": s2_meas, "var_total_resid": float(ac[0]),
            "half_life_fast_days": float(np.log(0.5) / np.log(phf)), "half_life_slow_days": float(np.log(0.5) / np.log(phs))}


def refit_job(args):
    x, mask, m2, smc, E = args
    earn = E.sum() > 20
    f, _ = B.fit_unit(x, mask, m2, smc, E, True, earn)
    return f


# ----------------------------------------------------------------------------- fit everything new
def fit_v2():
    P1 = pickle.load(open(B.CACHE, "rb"))
    panel, m_s, sectors, E = B.load_panel()
    R, m = panel.values, m_s.values
    mc = m - P1["mbar"]
    T, N = R.shape
    sidx = P1["sidx"]
    mask = np.isfinite(R)
    Q = {"market_v2": fit_market2(mc)}
    mk = Q["market_v2"]
    Q["LR_state_dependent_reversal"] = float(2 * (mk["ll"] - fit_market2(mc, statedep=False)["ll"]))
    print("LR state-dependent reversal", round(Q["LR_state_dependent_reversal"], 2), flush=True)
    print("market v2", {k: round(mk[k], 4) for k in ("rho", "a1", "a2", "c", "nu", "ll")}, "v1 ll", round(P1["market"]["full"]["ll"], 2), flush=True)
    # v1 market level path (for the gain's market term in the 'gain only' variant)
    m1 = P1["market"]["full"]
    e1 = mc - B.mkt_mean(mc, m1["a1"], m1["a2"])
    v1 = np.mean(e1 ** 2)
    lvl1 = B.filt(np.r_[v1, (e1 ** 2 * (1 + m1["delta"] * (e1 < 0)) / (1 + m1["delta"] / 2))[:-1]], m1["a"], m1["b"], v1, m1["omega"])
    Q["v1_Lbar"] = float(np.log(lvl1).mean())
    # stock residuals exactly as in test12
    F = np.empty((T, len(P1["sectors_names"])))
    for k in range(F.shape[1]):
        ew = np.nanmean(R[:, sidx == k], axis=1)
        bb = np.polyfit(mc, ew, 1)
        F[:, k] = ew - np.polyval(bb, mc)
    e = np.empty_like(R)
    for i, st in enumerate(P1["stocks"]):
        X = np.column_stack([np.ones(T), mc, F[:, sidx[i]]])
        e[:, i] = np.where(mask[:, i], R[:, i] - X @ st["coef"], 0.0)
    g, s2m = measure_gain(e, mask)
    Q["gain"] = fit_gain_dynamics(g, np.log(mk["level"]), mk["Lbar"], s2m)
    Q["gain"]["sd_measured_daily"] = float(g.std())
    print("gain", {k: round(v, 4) for k, v in Q["gain"].items()}, flush=True)
    # refit stocks on de-gained residuals
    smc = np.r_[0.0, mk["s"][:-1] - mk["sbar"]]
    m2 = mc ** 2
    de = np.exp(-g / 2)[:, None]
    jobs = [(np.where(mask[:, i], e[:, i] * de[:, 0], 0.0), mask[:, i], m2, smc, E[:, i]) for i in range(N)]
    t0 = time.time()
    Q["stocks_v2"] = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for i, f in enumerate(ex.map(refit_job, jobs, chunksize=4)):
            Q["stocks_v2"].append(f)
            if i % 40 == 0:
                print(f"  refit {i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    with open(CACHE2, "wb") as fh:
        pickle.dump(Q, fh)
    return Q


# ----------------------------------------------------------------------------- simulation
VARIANTS2 = {"v1_full": ("v1", False), "v2_reversal_only": ("v2", False), "v2_gain_only": ("v1", True), "v2_full": ("v2", True)}


def simulate2(P1, Q, variant, seed, mask, E, burn=500, stocks=None):
    """stocks: optional list of stock-unit fits to use instead (e.g. Test 13's two-day earnings units)."""
    mver, gain = VARIANTS2[variant]
    rng = np.random.default_rng(seed)
    T, N = mask.shape
    TT = T + burn
    Ef = np.vstack([np.zeros((burn, N)), E])
    mk = P1["market"]["full"] if mver == "v1" else Q["market_v2"]
    Lbar = Q["v1_Lbar"] if mver == "v1" else mk["Lbar"]
    cst = mk.get("c", 0.0) if mver == "v2" else 0.0
    Rbar, dmax = (mk["Rbar"], mk["Rmax"]) if mver == "v2" else (0.0, 0.0)   # cap: no extrapolation beyond the data
    units = stocks if stocks is not None else (Q["stocks_v2"] if gain else [s["full"] for s in P1["stocks"]])
    st = B.stack_units(units)
    kE1 = np.array([u.get("kE1", 0.0) for u in units])          # second-day earnings bump (0 unless given)
    coef = np.array([s["coef"] for s in P1["stocks"]])
    sidx = P1["sidx"]
    sc = B.stack_units(P1["sectors"]["full"])
    L = np.linalg.cholesky(P1["sector_corr"])
    nuf = P1["sector_nu_common"]
    S = len(sc["omega"])
    g = rng.standard_normal((TT, S)) @ L.T
    w = rng.chisquare(nuf, TT) / nuf
    eps_s = g / np.sqrt(w)[:, None] * np.sqrt((nuf - 2) / nuf)
    eps_m = B.std_t(rng, mk["nu"], TT)
    eps_i = B.std_t(rng, st["nu"][None, :], (TT, N))
    gp = Q["gain"]
    nf = rng.standard_normal(TT) * np.sqrt(gp["var_fast"] * (1 - gp["phi_fast"] ** 2))
    ns = rng.standard_normal(TT) * np.sqrt(gp["var_slow"] * (1 - gp["phi_slow"] ** 2))
    nw = rng.standard_normal(TT) * np.sqrt(gp["var_white"])
    xf = rng.standard_normal() * np.sqrt(gp["var_fast"])
    xs = rng.standard_normal() * np.sqrt(gp["var_slow"])
    am, bm = np.asarray(mk["a"]), np.asarray(mk["b"])
    c_m = am / (1 - bm) * mk["vbar"]
    c_s = sc["a"] / (1 - sc["b"]) * sc["vin"][:, None]
    c_i = st["a"] / (1 - st["b"]) * st["vin"][:, None]
    asym = lambda x, d: (1 + d * (x < 0)) / (1 + d / 2)
    m = np.zeros(TT)
    Fs = np.zeros((TT, S))
    Ei = np.zeros((TT, N))
    s_prev = mk["sbar"]
    for t in range(TT):
        lvl = mk["omega"] + c_m.sum()
        S21 = m[max(t - 21, 0):max(t - 1, 0)].sum()
        gauge = np.log(np.mean(m[max(t - 21, 0):t] ** 2)) if t else Rbar
        mu = (mk["a1"] * (m[t - 1] if t else 0.0) + mk["a2"] * S21) * np.exp(cst * min(gauge - Rbar, dmax))
        smc = s_prev - mk["sbar"]
        e = np.sqrt(lvl * np.exp(-mk["rho"] * smc)) * eps_m[t]
        m[t] = mu + e
        s_prev = abs(e) / np.sqrt(lvl)
        c_m = am * (e ** 2 * asym(e, mk["delta"])) + bm * c_m
        m2 = m[t] ** 2
        lv = sc["omega"] + c_s.sum(1)
        f = np.sqrt(lv * np.exp(sc["kap"] * smc)) * eps_s[t]
        Fs[t] = f
        c_s = sc["a"] * (f ** 2 * asym(f, sc["delta"]) + sc["lam"] * m2)[:, None] + sc["b"] * c_s
        if gain:
            xf = gp["phi_fast"] * xf + nf[t]
            xs = gp["phi_slow"] * xs + ns[t]
            G = np.exp(gp["beta_market"] * (np.log(lvl) - Lbar) + xf + xs + nw[t])
        else:
            G = 1.0
        bump = 1 + st["kE"] * Ef[t] + kE1 * (Ef[t - 1] if t else 0.0)
        lv = st["omega"] + c_i.sum(1)
        x = np.sqrt(lv * bump * np.exp(st["kap"] * smc) * G) * eps_i[t]
        Ei[t] = x
        c_i = st["a"] * (x ** 2 * asym(x, st["delta"]) / (bump * G) + st["lam"] * m2)[:, None] + st["b"] * c_i
    m, Fs, Ei = m[burn:] + P1["mbar"], Fs[burn:], Ei[burn:]
    Rs = coef[:, 0] + coef[:, 1] * (m - P1["mbar"])[:, None] + coef[:, 2] * Fs[:, sidx] + Ei
    Rs[~mask] = np.nan
    return m, Rs


def scorecard2(m, R):
    out = B.scorecard(m, R)
    mc = m - m.mean()
    sig = pd.Series(mc ** 2).rolling(21).mean().shift(1).values
    ok = np.isfinite(sig[1:])
    hi = (sig[1:] > np.nanmedian(sig)) & ok
    lo = (~hi) & ok
    out["index_ac1_high_vol"] = float(np.corrcoef(mc[:-1][hi], mc[1:][hi])[0, 1])
    out["index_ac1_low_vol"] = float(np.corrcoef(mc[:-1][lo], mc[1:][lo])[0, 1])
    Rc = np.where(np.isfinite(R), R - np.nanmean(R, 0), np.nan)
    b = np.nansum(Rc * mc[:, None], 0) / np.nansum(np.where(np.isfinite(R), mc[:, None] ** 2, 0), 0)
    e = Rc - mc[:, None] * b
    civ = np.nanmedian(e ** 2 / np.nanvar(e, 0), axis=1)
    k = len(m) // 21
    civm = np.log(civ[:k * 21].reshape(k, 21).mean(1))
    rvm = np.log((mc[:k * 21] ** 2).reshape(k, 21).mean(1))
    out["common_idio_vol_monthly_sd"] = float(civm.std())
    out["common_idio_vol_corr_market_vol"] = float(np.corrcoef(civm, rvm)[0, 1])
    out["common_idio_vol_acf_6m"] = float(acf(civm, 6)[6])
    return out


_D = {}


def sim_job2(args):
    variant, seed = args
    if not _D:
        _D["P1"] = pickle.load(open(B.CACHE, "rb"))
        _D["Q"] = pickle.load(open(CACHE2, "rb"))
        panel, _, _, E = B.load_panel()
        _D["mask"], _D["E"] = np.isfinite(panel.values), E
    m, R = simulate2(_D["P1"], _D["Q"], variant, seed, _D["mask"], _D["E"])
    return variant, scorecard2(m, R)


def main():
    Q = pickle.load(open(CACHE2, "rb")) if CACHE2.exists() else fit_v2()
    panel, m_s, _, _ = B.load_panel()
    real = scorecard2(m_s.values, panel.values)
    jobs = [(v, 7000 + 131 * k + j) for k, v in enumerate(VARIANTS2) for j in range(N_PATHS)]
    sims = {v: [] for v in VARIANTS2}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for i, (v, sc) in enumerate(ex.map(sim_job2, jobs, chunksize=1)):
            sims[v].append(sc)
            if i % 10 == 0:
                print(f"  sim {i + 1}/{len(jobs)} {v} {time.time() - t0:.0f}s", flush=True)
    res = {"real": real, "variants": {}, "n_paths": N_PATHS,
           "market_v2": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in Q["market_v2"].items() if k not in ("s", "level")},
           "gain": Q["gain"], "LR_state_dependent_reversal": Q["LR_state_dependent_reversal"],
           "stocks_v2_median": {k: float(np.median([s[k] for s in Q["stocks_v2"]])) for k in ("lam", "kap", "kE", "nu")}}
    for v, rows in sims.items():
        df = pd.DataFrame(rows)
        res["variants"][v] = {k: {"mean": float(df[k].mean()), "p5": float(df[k].quantile(0.05)), "p95": float(df[k].quantile(0.95)),
                                  "sd": float(df[k].std()), "real_pctile": float((df[k] < real[k]).mean())} for k in df.columns}
        inside = [k for k in df.columns if res["variants"][v][k]["p5"] <= real[k] <= res["variants"][v][k]["p95"]]
        res["variants"][v]["_n_inside_5_95"] = len(inside)
        res["variants"][v]["_n_patterns"] = len(df.columns)
        print(v, "inside 5-95%:", len(inside), "of", len(df.columns), flush=True)
    (OUT / "test12c.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
