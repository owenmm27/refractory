"""Test 12 - One model of the "market brain": assemble the pieces from Tests 1-11 and check what they reproduce.

Generative model (daily, 2000-2026; 237 stocks in 11 GICS sectors):
  MARKET unit (S&P 500 return m):
      m_t   = a1*m_{t-1} + a2*sum(m_{t-21..t-2}) + e_t                    (short and medium reversal)
      e_t   = sqrt(h_t) * t-shock,  h_t = level_t * exp(-rho*(s_{t-1} - mean s))   (refractory period)
      level = omega + two leaky integrators of e^2 (fast + slow), asymmetric for down moves
  SECTOR units (11 sector factors f = sector equal-weight return beyond the market), correlated t-shocks:
      h_s,t = level_s,t * exp(kap_s*(s_m,t-1 - mean))   (one-day response to yesterday's market surprise)
      level_s = omega + two integrators of [own f^2 (asymmetric) + lam_s * m^2]   (persistent market spillover)
  STOCK units (idiosyncratic e_i = r_i - alpha - b*m - c*f_sector), independent t-shocks:
      as sectors, plus an earnings-day variance bump (1 + kE) kept out of the persistent state (2004+ calendar)
  Stock return = alpha_i + b_i*m + c_i*f_sector(i) + e_i, with the real data's missing-day pattern.

Each layer is fitted by maximum likelihood (Student-t) on the real data. The model is then simulated (N paths) and
a scorecard of ~25 patterns from earlier tests is computed identically on real and simulated data. Ablations
remove one piece at a time (refitted without it) to show which pieces each pattern needs.

Usage: python3 -I src/test12_brain_model.py <project_root> [n_paths]
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
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import kurtosis

import test3_avalanches as t3
from common import _unpack_multi, acf, fit_multi, load_stocks
from test4_balance import ac1, variance_ratio
from test8_earnings import reaction_days
from test9_scheduled_news import t_ll

t3.K = 4.0          # firing threshold (test3 reads argv[2] at import; reset it)
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
N_PATHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUT = ROOT / "results"
CACHE = OUT / "cache" / "test12_fit.pkl"
START = "2000-01-01"


def nbase(K):
    return 2 * K + 3


def filt(u, a, b, vin, omega):
    level = np.full(len(u), omega)
    for ak, bk in zip(a, b):
        level = level + lfilter([ak], [1.0, -bk], u, zi=[bk * ak / (1 - bk) * vin])[0]
    return level


# ----------------------------------------------------------------------------- market unit
def mkt_mean(m, a1, a2):
    t = np.arange(len(m))
    cs = np.r_[0.0, np.cumsum(m)]
    S = cs[np.maximum(t - 1, 0)] - cs[np.maximum(t - 21, 0)]          # sum of m[t-21 .. t-2]
    return a1 * np.r_[0.0, m[:-1]] + a2 * S


def mkt_path(p, m, K, refr, mean):
    nb = nbase(K)
    omega, a, b, delta, nu = _unpack_multi(p[:nb], K)
    j = nb
    rho = p[j] if refr else 0.0
    j += int(refr)
    a1, a2 = (p[j], p[j + 1]) if mean else (0.0, 0.0)
    e = m - mkt_mean(m, a1, a2)
    v = np.mean(e ** 2)
    sq = e ** 2 * (1 + delta * (e < 0)) / (1 + delta / 2)
    level = filt(np.r_[v, sq[:-1]], a, b, v, omega)
    s = np.abs(e) / np.sqrt(level)
    sbar = float(s.mean())
    h = level * np.exp(-rho * (np.r_[sbar, s[:-1]] - sbar))
    return e, h, nu, s, sbar


def fit_market(m, K=2, refr=True, mean=True):
    f0 = fit_multi(m, K)

    def nll(p):
        e, h, nu, _, _ = mkt_path(p, m, K, refr, mean)
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
    p = best.x
    nb = nbase(K)
    omega, a, b, delta, nu = _unpack_multi(p[:nb], K)
    e, h, nu, s, sbar = mkt_path(p, m, K, refr, mean)
    j = nb
    rho = float(p[j]) if refr else 0.0
    j += int(refr)
    a1, a2 = (float(p[j]), float(p[j + 1])) if mean else (0.0, 0.0)
    return {"K": K, "omega": float(omega), "a": a, "b": b, "delta": float(delta), "nu": float(nu), "rho": rho,
            "a1": a1, "a2": a2, "sbar": sbar, "ll": float(-best.fun), "s": s, "vbar": float(np.mean(e ** 2))}


# ----------------------------------------------------------------------------- sector / stock units
def unit_path(p, x, mask, m2, smc, E, spill, earn, K=2):
    nb = nbase(K)
    omega, a, b, delta, nu = _unpack_multi(p[:nb], K)
    j = nb
    lam = kap = 0.0
    if spill:
        lam, kap = p[j], p[j + 1]
        j += 2
    bump = 1.0 + np.exp(p[j]) * E if earn else np.ones(len(x))
    sq = x ** 2 * (1 + delta * (x < 0)) / (1 + delta / 2) / bump + lam * m2
    vin = float(np.mean(sq[mask]))
    level = filt(np.r_[vin, sq[:-1]], a, b, vin, omega)
    h = level * bump * np.exp(kap * smc)
    return h, nu, vin


def fit_unit(x, mask, m2, smc, E, spill, earn):
    f0 = fit_multi(x[mask], 2)
    p0 = np.r_[f0["params_vec"], [0.0, 0.0] if spill else [], [0.0] if earn else []]
    bounds = [(None, None)] * 7 + ([(0.0, 5.0), (-1.5, 1.5)] if spill else []) + ([(-6.0, 4.0)] if earn else [])

    def nll(p):
        h, nu, _ = unit_path(p, x, mask, m2, smc, E, spill, earn)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        return -t_ll(x[mask], h[mask], nu).sum()
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    p = res.x
    omega, a, b, delta, nu = _unpack_multi(p[:7], 2)
    h, nu, vin = unit_path(p, x, mask, m2, smc, E, spill, earn)
    out = {"omega": float(omega), "a": a, "b": b, "delta": float(delta), "nu": float(nu), "vin": vin,
           "lam": float(p[7]) if spill else 0.0, "kap": float(p[8]) if spill else 0.0,
           "kE": float(np.exp(p[-1])) if earn else 0.0, "ll": float(-res.fun)}
    return out, np.where(mask, x / np.sqrt(h), np.nan)


def stock_job(args):
    sym, r, m, fs, m2, smc, E = args
    mask = np.isfinite(r)
    earn = E.sum() > 20
    X = np.column_stack([np.ones(len(m)), m, fs])
    coef = np.linalg.lstsq(X[mask], r[mask], rcond=None)[0]
    e = np.where(mask, r - X @ coef, 0.0)
    full, _ = fit_unit(e, mask, m2, smc, E, True, earn)
    nosp, _ = fit_unit(e, mask, m2, smc, E, False, earn)
    coef2 = np.linalg.lstsq(X[mask][:, :2], r[mask], rcond=None)[0]
    e2 = np.where(mask, r - X[:, :2] @ coef2, 0.0)
    nosec, _ = fit_unit(e2, mask, m2, smc, E, True, earn)
    return {"sym": sym, "coef": coef, "coef_nosec": coef2, "full": full, "nospill": nosp, "nosec": nosec}


# ----------------------------------------------------------------------------- data + fitting
def load_panel():
    st, spx = load_stocks(ROOT / "data")
    panel = st.loc[:, st.notna().mean() >= 0.98]
    panel = panel.loc[panel.notna().mean(axis=1) >= 0.95]
    panel = panel.loc[START:] * 100
    m = spx.reindex(panel.index) * 100
    keep = m.notna()
    panel, m = panel.loc[keep], m.loc[keep]
    meta = pd.read_csv(ROOT / "data" / "meta" / "constituents.csv").set_index("Symbol")
    sectors = np.array([meta.loc[s, "GICS Sector"] for s in panel.columns])
    ev = pd.read_csv(ROOT / "data" / "earnings" / "earnings_8k_202.csv")
    ev = ev[ev["symbol"].isin(panel.columns)].copy()
    ev["pos"] = reaction_days(ev, pd.Series(panel.index))
    ev = ev[ev["pos"] >= 0]
    E = np.zeros(panel.shape)
    col = {s: i for i, s in enumerate(panel.columns)}
    for sym, g in ev.groupby("symbol"):
        E[g["pos"].values, col[sym]] = 1.0
    return panel, m, sectors, E


def fit_all():
    panel, m_s, sectors, E = load_panel()
    R, m = panel.values, m_s.values
    mbar = float(m.mean())
    mc = m - mbar
    T, N = R.shape
    secs = sorted(set(sectors))
    sidx = np.array([secs.index(s) for s in sectors])
    print(f"panel {T} x {N}, {len(secs)} sectors; fitting market...", flush=True)
    P = {"mbar": mbar, "sectors_names": secs, "sidx": sidx, "symbols": list(panel.columns), "dates": panel.index}
    P["market"] = {"full": fit_market(mc), "no_refractory": fit_market(mc, refr=False),
                   "no_reversal": fit_market(mc, mean=False), "one_timescale": fit_market(mc, K=1)}
    for k, v in P["market"].items():
        print(" market", k, {kk: (round(vv, 4) if isinstance(vv, float) else None) for kk, vv in v.items() if kk in ("ll", "rho", "a1", "a2", "nu")}, flush=True)
    mk = P["market"]["full"]
    smc = np.r_[0.0, mk["s"][:-1] - mk["sbar"]]
    m2 = mc ** 2
    # sector factors
    F = np.empty((T, len(secs)))
    for k in range(len(secs)):
        ew = np.nanmean(R[:, sidx == k], axis=1)
        bb = np.polyfit(mc, ew, 1)
        F[:, k] = ew - np.polyval(bb, mc)
    allm = np.ones(T, bool)
    P["sectors"] = {"full": [], "nospill": []}
    Zs = np.empty_like(F)
    for k in range(len(secs)):
        f, z = fit_unit(F[:, k], allm, m2, smc, np.zeros(T), True, False)
        P["sectors"]["full"].append(f)
        Zs[:, k] = z
        P["sectors"]["nospill"].append(fit_unit(F[:, k], allm, m2, smc, np.zeros(T), False, False)[0])
    P["sector_corr"] = np.corrcoef(Zs, rowvar=False)
    P["sector_nu_common"] = float(np.median([f["nu"] for f in P["sectors"]["full"]]))
    print(" sectors done", {"lam_med": np.median([f["lam"] for f in P["sectors"]["full"]]),
                            "kap_med": np.median([f["kap"] for f in P["sectors"]["full"]])}, flush=True)
    jobs = [(panel.columns[i], R[:, i], mc, F[:, sidx[i]], m2, smc, E[:, i]) for i in range(N)]
    t0 = time.time()
    P["stocks"] = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for i, o in enumerate(ex.map(stock_job, jobs, chunksize=4)):
            P["stocks"].append(o)
            if i % 20 == 0:
                print(f"  stock {i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    print(f" stocks fitted in {time.time() - t0:.0f}s", flush=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(P, fh)
    return P


# ----------------------------------------------------------------------------- simulation
def stack_units(units):
    return {k: np.array([u[k] for u in units]) for k in ("omega", "a", "b", "delta", "nu", "vin", "lam", "kap", "kE")}


def std_t(rng, nu, size):
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2) / nu)


VARIANTS = {   # name: (market fit, sector fits or None, stock fit key)
    "full": ("full", "full", "full"),
    "no_refractory": ("no_refractory", "full", "full"),
    "no_reversal": ("no_reversal", "full", "full"),
    "one_timescale_market": ("one_timescale", "full", "full"),
    "no_spillover": ("full", "nospill", "nospill"),
    "no_sector_layer": ("full", None, "nosec"),
}


def simulate(P, variant, seed, mask, E, burn=500):
    mkey, skey, ikey = VARIANTS[variant]
    rng = np.random.default_rng(seed)
    mk = P["market"][mkey]
    T, N = mask.shape
    TT = T + burn
    Ef = np.vstack([np.zeros((burn, N)), E])
    st = stack_units([s[ikey] for s in P["stocks"]])
    coef = np.array([s["coef"] if ikey != "nosec" else np.r_[s["coef_nosec"], 0.0] for s in P["stocks"]])
    use_sec = skey is not None
    sidx = P["sidx"]
    if use_sec:
        sc = stack_units(P["sectors"][skey])
        L = np.linalg.cholesky(P["sector_corr"])
        nuf = P["sector_nu_common"]
        S = len(sc["omega"])
        g = rng.standard_normal((TT, S)) @ L.T
        w = rng.chisquare(nuf, TT) / nuf
        eps_s = g / np.sqrt(w)[:, None] * np.sqrt((nuf - 2) / nuf)
        c_s = sc["a"] / (1 - sc["b"]) * sc["vin"][:, None]
        asym_s = lambda x: (1 + sc["delta"] * (x < 0)) / (1 + sc["delta"] / 2)
    eps_m = std_t(rng, mk["nu"], TT)
    eps_i = std_t(rng, st["nu"][None, :], (TT, N))
    am, bm = np.asarray(mk["a"]), np.asarray(mk["b"])
    c_m = am / (1 - bm) * mk["vbar"]
    c_i = st["a"] / (1 - st["b"]) * st["vin"][:, None]
    asym_i = lambda x: (1 + st["delta"] * (x < 0)) / (1 + st["delta"] / 2)
    m = np.zeros(TT)
    Fs = np.zeros((TT, len(P["sectors_names"])))
    Ei = np.zeros((TT, N))
    s_prev = mk["sbar"]
    for t in range(TT):
        S21 = m[max(t - 21, 0):max(t - 1, 0)].sum()
        mu = mk["a1"] * (m[t - 1] if t else 0.0) + mk["a2"] * S21
        lvl = mk["omega"] + c_m.sum()
        smc = s_prev - mk["sbar"]
        e = np.sqrt(lvl * np.exp(-mk["rho"] * smc)) * eps_m[t]
        m[t] = mu + e
        s_prev = abs(e) / np.sqrt(lvl)
        c_m = am * (e ** 2 * (1 + mk["delta"] * (e < 0)) / (1 + mk["delta"] / 2)) + bm * c_m
        m2 = m[t] ** 2
        if use_sec:
            lv = sc["omega"] + c_s.sum(1)
            f = np.sqrt(lv * np.exp(sc["kap"] * smc)) * eps_s[t]
            Fs[t] = f
            c_s = sc["a"] * (f ** 2 * asym_s(f) + sc["lam"] * m2)[:, None] + sc["b"] * c_s
        bump = 1 + st["kE"] * Ef[t]
        lv = st["omega"] + c_i.sum(1)
        x = np.sqrt(lv * bump * np.exp(st["kap"] * smc)) * eps_i[t]
        Ei[t] = x
        c_i = st["a"] * (x ** 2 * asym_i(x) / bump + st["lam"] * m2)[:, None] + st["b"] * c_i
    m, Fs, Ei = m[burn:] + P["mbar"], Fs[burn:], Ei[burn:]
    R = coef[:, 0] + coef[:, 1] * (m - P["mbar"])[:, None] + coef[:, 2] * Fs[:, sidx] + Ei
    R[~mask] = np.nan
    return m, R


# ----------------------------------------------------------------------------- scorecard
def dip_ratio(x2, big, lead=6):
    idx = np.where(big)[0]
    idx = idx[idx < len(x2) - lead]
    nxt = np.nanmean(x2[idx + 1])
    later = np.nanmean(np.concatenate([x2[idx + k] for k in range(2, lead + 1)]))
    return float(nxt / later)


def scorecard(m, R):
    out = {}
    mc = m - m.mean()
    a = acf(np.abs(mc), 252)
    for L in (1, 5, 21, 63, 126, 252):
        out[f"index_absacf_lag{L}"] = float(a[L])
    out["index_excess_kurtosis"] = float(kurtosis(mc))
    out["index_ac1"] = ac1(mc)
    out["index_VR5"] = variance_ratio(mc, 5)
    out["index_VR21"] = variance_ratio(mc, 21)
    out["index_leverage_corr"] = float(np.corrcoef(mc[:-1], np.abs(mc[1:]))[0, 1])
    f = fit_multi(mc, 2)
    z = f["z"]
    out["index_absz_lag1_corr_vs_plain_model"] = float(np.corrcoef(np.abs(z[:-1]), np.abs(z[1:]))[0, 1])
    sig = np.sqrt(pd.Series(mc ** 2).rolling(21).mean().shift(1).values)
    big = np.abs(mc) > 2.5 * sig
    out["index_dip_after_big_day"] = dip_ratio(mc ** 2, big)
    # idiosyncratic (market-only residual), identical on real and simulated data
    mask = np.isfinite(R)
    Rc = np.where(mask, R - np.nanmean(R, 0), 0.0)
    b = (Rc * mc[:, None]).sum(0) / ((mc ** 2)[:, None] * mask).sum(0)
    Eid = np.where(mask, Rc - mc[:, None] * b, np.nan)
    e2 = Eid ** 2
    ebar2 = np.nanmean(e2, axis=1)
    out["idio_dip_after_big_market_day"] = dip_ratio(ebar2, big)
    sig_i = np.sqrt(pd.DataFrame(e2).rolling(21, min_periods=15).mean().shift(1).values)
    bigi = np.abs(Eid) > 4 * sig_i
    T = len(m)
    idx_t, idx_n = np.where(bigi[:T - 6])
    nxt = np.nanmean(e2[idx_t + 1, idx_n])
    later = np.nanmean(np.concatenate([e2[idx_t + k, idx_n] for k in range(2, 7)]))
    out["stock_dip_after_own_big_day"] = float(nxt / later)
    A = np.abs(Eid)
    for L in (1, 21, 126):
        x0, x1 = A[:-L], A[L:]
        ok = np.isfinite(x0) & np.isfinite(x1)
        mu = np.nanmean(A, 0)
        sd = np.nanstd(A, 0)
        cc = np.nansum(np.where(ok, (x0 - mu) * (x1 - mu), 0), 0) / ok.sum(0) / sd ** 2
        out[f"idio_absacf_lag{L}_mean"] = float(np.mean(cc))
    out["idio_excess_kurtosis_median"] = float(np.median([kurtosis(Eid[mask[:, i], i]) for i in range(R.shape[1])]))
    # synchrony in non-overlapping 21-day windows
    sync, lrv = [], []
    for s in range(0, T - 20, 21):
        w = R[s:s + 21]
        keep = np.isfinite(w).all(0)
        c = np.corrcoef(w[:, keep], rowvar=False)
        k = c.shape[0]
        sync.append((c.sum() - k) / (k * (k - 1)))
        lrv.append(np.log((mc[s:s + 21] ** 2).sum()))
    sync, lrv = np.array(sync), np.array(lrv)
    out["sync_mean_pairwise_corr"] = float(sync.mean())
    out["sync_vs_same_month_vol_corr"] = float(np.corrcoef(sync, lrv)[0, 1])
    out["sync_after_minus_before_vol"] = float(np.corrcoef(sync[1:], lrv[:-1])[0, 1] - np.corrcoef(sync[:-1], lrv[1:])[0, 1])
    # collective firing / avalanches (threshold 4 robust SDs)
    counts = t3.firing_counts(R)
    sm, _, _ = t3.summarize(counts)
    for k in ("silent_share", "fano_factor", "lag1_slope", "size_p99", "dur_p99", "tau_size"):
        out[f"firing_{k}"] = float(sm[k])
    return out


_SIM_DATA = {}


def sim_job(args):
    variant, seed = args
    if not _SIM_DATA:
        with open(CACHE, "rb") as fh:
            _SIM_DATA["P"] = pickle.load(fh)
        panel, _, _, E = load_panel()
        _SIM_DATA["mask"] = np.isfinite(panel.values)
        _SIM_DATA["E"] = E
    P = _SIM_DATA["P"]
    m, R = simulate(P, variant, seed, _SIM_DATA["mask"], _SIM_DATA["E"])
    return variant, seed, scorecard(m, R)


def main():
    P = pickle.load(open(CACHE, "rb")) if CACHE.exists() else fit_all()
    panel, m_s, _, _ = load_panel()
    real = scorecard(m_s.values, panel.values)
    print("real", json.dumps({k: round(v, 3) for k, v in real.items()}), flush=True)
    jobs = [(v, 5000 + 97 * k + j) for k, v in enumerate(VARIANTS) for j in range(N_PATHS)]
    sims = {v: [] for v in VARIANTS}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for i, (v, seed, sc) in enumerate(ex.map(sim_job, jobs, chunksize=1)):
            sims[v].append(sc)
            if i % 10 == 0:
                print(f"  sim {i + 1}/{len(jobs)} {v} {time.time() - t0:.0f}s", flush=True)
    res = {"real": real, "variants": {}, "n_paths": N_PATHS,
           "params": {"market": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in f.items() if kk != "s"}
                                 for k, f in P["market"].items()},
                      "sector_median": {k: float(np.median([f[k] for f in P["sectors"]["full"]])) for k in ("lam", "kap", "nu")},
                      "stock_median": {k: float(np.median([s["full"][k] for s in P["stocks"]])) for k in ("lam", "kap", "kE", "nu")},
                      "stock_share_kap_positive": float(np.mean([s["full"]["kap"] > 0 for s in P["stocks"]])),
                      "stock_LR_spillover_sum": float(sum(2 * (s["full"]["ll"] - s["nospill"]["ll"]) for s in P["stocks"]))}}
    for v, rows in sims.items():
        df = pd.DataFrame(rows)
        res["variants"][v] = {k: {"mean": float(df[k].mean()), "p5": float(df[k].quantile(0.05)), "p95": float(df[k].quantile(0.95)),
                                  "real_pctile": float((df[k] < real[k]).mean())} for k in df.columns}
        inside = np.mean([(res["variants"][v][k]["p5"] <= real[k] <= res["variants"][v][k]["p95"]) for k in df.columns])
        res["variants"][v]["_share_inside_5_95"] = float(inside)
        print(v, "share of patterns inside the model's 5-95% range:", round(inside, 2), flush=True)
    (OUT / "test12.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame({"real": real} | {v: {k: res["variants"][v][k]["mean"] for k in real} for v in VARIANTS}).to_csv(OUT / "test12_scorecard.csv")


if __name__ == "__main__":
    main()
