"""Test 13b - Re-score the market-brain model on robust burst-persistence measures; add two-day earnings reactions.

Test 13a found:
  * the raw persistence slope of collective firing (0.54 real) is dominated by the 2008-09 and 2020 crises; on robust
    measures (rank and log correlations; slope outside those crises) version 2's shared volatility gain already
    matches the real data, version 1 does not;
  * a real miss in the stock units: the day after a big earnings move has 3.8x normal variance in real data, 1.9x in
    the model, because the model treats earnings as a one-day jump.
Version 3 = version 2 + a second-day earnings bump (also kept out of the persistent state). Stocks are refitted on
the de-gained residuals exactly as in version 2. 20 simulated histories each of v1, v2 and v3 are scored on the
Test 12 scorecard plus robust burst measures and aftershock profiles.

Usage: python3 -I src/test13b_model_v3.py <project_root> [n_paths]
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
from scipy.stats import spearmanr

import test12_brain_model as B
import test12c_model_v2 as C
import test13_bursts as TB
from common import _unpack_multi, fit_multi
from test9_scheduled_news import t_ll

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
N_PATHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUT = ROOT / "results"
CACHE3 = OUT / "cache" / "test13_fit.pkl"


# ----------------------------------------------------------------------------- stock unit with two-day earnings bump
def unit_path3(p, x, mask, m2, smc, E0, E1, earn):
    omega, a, b, delta, nu = _unpack_multi(p[:7], 2)
    lam, kap = p[7], p[8]
    bump = 1.0 + np.exp(p[9]) * E0 + np.exp(p[10]) * E1 if earn else np.ones(len(x))
    sq = x ** 2 * (1 + delta * (x < 0)) / (1 + delta / 2) / bump + lam * m2
    vin = float(np.mean(sq[mask]))
    level = B.filt(np.r_[vin, sq[:-1]], a, b, vin, omega)
    return level * bump * np.exp(kap * smc), nu, vin


def fit_unit3(args):
    x, mask, m2, smc, E0 = args
    E1 = np.r_[0.0, E0[:-1]]
    earn = E0.sum() > 20
    f0 = fit_multi(x[mask], 2)
    p0 = np.r_[f0["params_vec"], 0.0, 0.0, [np.log(10.0), np.log(1.0)] if earn else []]
    bounds = [(None, None)] * 7 + [(0.0, 5.0), (-1.5, 1.5)] + ([(-6.0, 4.0)] * 2 if earn else [])

    def nll(p):
        h, nu, _ = unit_path3(p, x, mask, m2, smc, E0, E1, earn)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        return -t_ll(x[mask], h[mask], nu).sum()
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    p = res.x
    omega, a, b, delta, nu = _unpack_multi(p[:7], 2)
    _, nu, vin = unit_path3(p, x, mask, m2, smc, E0, E1, earn)
    return {"omega": float(omega), "a": a, "b": b, "delta": float(delta), "nu": float(nu), "vin": vin,
            "lam": float(p[7]), "kap": float(p[8]), "kE": float(np.exp(p[9])) if earn else 0.0,
            "kE1": float(np.exp(p[10])) if earn else 0.0, "ll": float(-res.fun)}


def fit_v3():
    P1 = pickle.load(open(B.CACHE, "rb"))
    Q = pickle.load(open(C.CACHE2, "rb"))
    panel, m_s, _, E = B.load_panel()
    R, m = panel.values, m_s.values
    mc = m - P1["mbar"]
    T, N = R.shape
    sidx = P1["sidx"]
    mask = np.isfinite(R)
    F = np.empty((T, len(P1["sectors_names"])))
    for k in range(F.shape[1]):
        ew = np.nanmean(R[:, sidx == k], axis=1)
        bb = np.polyfit(mc, ew, 1)
        F[:, k] = ew - np.polyval(bb, mc)
    e = np.empty_like(R)
    for i, st in enumerate(P1["stocks"]):
        X = np.column_stack([np.ones(T), mc, F[:, sidx[i]]])
        e[:, i] = np.where(mask[:, i], R[:, i] - X @ st["coef"], 0.0)
    g, _ = C.measure_gain(e, mask)
    mk = Q["market_v2"]
    smc = np.r_[0.0, mk["s"][:-1] - mk["sbar"]]
    de = np.exp(-g / 2)
    jobs = [(np.where(mask[:, i], e[:, i] * de, 0.0), mask[:, i], mc ** 2, smc, E[:, i]) for i in range(N)]
    t0 = time.time()
    fits = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for i, f in enumerate(ex.map(fit_unit3, jobs, chunksize=4)):
            fits.append(f)
            if i % 40 == 0:
                print(f"  refit {i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    LR = [2 * (f3["ll"] - f2["ll"]) for f3, f2 in zip(fits, Q["stocks_v2"])]
    Q3 = {"stocks_v3": fits, "LR_second_day_vs_v2": LR}
    with open(CACHE3, "wb") as fh:
        pickle.dump(Q3, fh)
    return Q3


# ----------------------------------------------------------------------------- robust burst measures
def burst_robust(m, R, E, dates):
    fire, valid, good = TB.fire_matrix(R)
    Cn = fire.sum(1).astype(float)
    crisis = ((dates >= np.datetime64("2008-09-01")) & (dates <= np.datetime64("2009-06-30"))) | \
             ((dates >= np.datetime64("2020-02-15")) & (dates <= np.datetime64("2020-06-30")))
    ok = good[1:] & good[:-1]
    a, b = Cn[:-1][ok], Cn[1:][ok]
    okx = ok & ~crisis[1:] & ~crisis[:-1]
    out = {"burst_slope_all": float(np.polyfit(a, b, 1)[0]),
           "burst_slope_ex_crisis": float(np.polyfit(Cn[:-1][okx], Cn[1:][okx], 1)[0]),
           "burst_rank_corr": float(spearmanr(a, b).statistic),
           "burst_log_corr": float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]),
           "firing_per_day": float(Cn[good].mean())}
    # outside the crises: repeat / new split and repeat probability; crisis intensity
    x = Cn[:-1][okx]
    vx = x.var()
    rep = (fire[1:] & fire[:-1]).sum(1)[okx]
    new = (fire[1:] & ~fire[:-1]).sum(1)[okx]
    v = valid[:-1] & valid[1:] & okx[:, None]
    f0, f1 = fire[:-1], fire[1:]
    out["burst_slope_repeat_ex_crisis"] = float(np.cov(rep, x)[0, 1] / vx)
    out["burst_slope_new_ex_crisis"] = float(np.cov(new, x)[0, 1] / vx)
    out["p_fire_again_ex_crisis"] = float((f1 & f0 & v).sum() / max((f0 & v).sum(), 1))
    out["firing_per_day_ex_crisis"] = float(Cn[good & ~crisis].mean())
    c21 = np.convolve(np.where(good, Cn, 0.0), np.ones(21) / 21, "valid")
    out["peak_month_firing"] = float(c21.max())
    out["days_half_of_stocks_firing"] = float((Cn >= 0.5 * R.shape[1]).sum())
    st = TB.burst_stats(m, R, E)
    out["burst_slope_repeat"] = st["slope_repeat"]
    out["burst_slope_new"] = st["slope_new"]
    out["p_fire_again"] = st["p_fire_given_fired"]
    for lab in ("earnings", "other"):
        out[f"aftershock_{lab}_day1"] = st[f"aftershock_{lab}"][0]
        out[f"aftershock_{lab}_day2"] = st[f"aftershock_{lab}"][1]
    return out


_D = {}


def sim_job(args):
    variant, seed = args
    if not _D:
        _D["P1"] = pickle.load(open(B.CACHE, "rb"))
        _D["Q"] = pickle.load(open(C.CACHE2, "rb"))
        _D["Q3"] = pickle.load(open(CACHE3, "rb"))
        panel, _, _, E = B.load_panel()
        _D["mask"], _D["E"], _D["dates"] = np.isfinite(panel.values), E, panel.index.values
    P1, Q = _D["P1"], _D["Q"]
    if variant == "v1":
        m, R = C.simulate2(P1, Q, "v1_full", seed, _D["mask"], _D["E"])
    elif variant == "v2":
        m, R = C.simulate2(P1, Q, "v2_full", seed, _D["mask"], _D["E"])
    else:
        m, R = C.simulate2(P1, Q, "v2_full", seed, _D["mask"], _D["E"], stocks=_D["Q3"]["stocks_v3"])
    return variant, C.scorecard2(m, R) | burst_robust(m, R, _D["E"], _D["dates"])


def main():
    Q3 = pickle.load(open(CACHE3, "rb")) if CACHE3.exists() else fit_v3()
    LR = np.array(Q3["LR_second_day_vs_v2"])
    k1 = np.array([f["kE1"] for f in Q3["stocks_v3"]])
    print("second-day earnings bump: median kE1 %.2f, LR median %.1f, share LR>3.84 %.2f, LR sum %.0f"
          % (np.median(k1[k1 > 0]), np.median(LR), np.mean(LR > 3.84), LR.sum()), flush=True)
    panel, m_s, _, E = B.load_panel()
    real = C.scorecard2(m_s.values, panel.values) | burst_robust(m_s.values, panel.values, E, panel.index.values)
    variants = ["v1", "v2", "v3"]
    jobs = [(v, 9100 + 211 * k + j) for k, v in enumerate(variants) for j in range(N_PATHS)]
    sims = {v: [] for v in variants}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for i, (v, sc) in enumerate(ex.map(sim_job, jobs, chunksize=1)):
            sims[v].append(sc)
            if i % 15 == 0:
                print(f"  sim {i + 1}/{len(jobs)} {v} {time.time() - t0:.0f}s", flush=True)
    res = {"real": real, "n_paths": N_PATHS, "variants": {},
           "second_day_bump": {"kE1_median": float(np.median(k1[k1 > 0])), "kE_median": float(np.median([f["kE"] for f in Q3["stocks_v3"] if f["kE"] > 0])),
                               "LR_median": float(np.median(LR)), "share_LR_gt_3.84": float(np.mean(LR > 3.84)), "LR_sum": float(LR.sum())}}
    for v, rows in sims.items():
        df = pd.DataFrame(rows)
        res["variants"][v] = {k: {"mean": float(df[k].mean()), "p5": float(df[k].quantile(0.05)), "p95": float(df[k].quantile(0.95)),
                                  "real_pctile": float((df[k] < real[k]).mean())} for k in df.columns}
    res["paths"] = {v: {k: [float(r[k]) for r in rows] for k in ("peak_month_firing", "days_half_of_stocks_firing")}
                    for v, rows in sims.items()}
    (OUT / "test13b.json").write_text(json.dumps(res, indent=2, default=float))
    keys = ["burst_slope_all", "burst_slope_ex_crisis", "burst_rank_corr", "burst_log_corr", "burst_slope_repeat", "burst_slope_new",
            "burst_slope_repeat_ex_crisis", "burst_slope_new_ex_crisis", "p_fire_again_ex_crisis", "firing_per_day_ex_crisis",
            "peak_month_firing", "days_half_of_stocks_firing",
            "p_fire_again", "firing_per_day", "aftershock_earnings_day1", "aftershock_earnings_day2", "aftershock_other_day1",
            "aftershock_other_day2", "stock_dip_after_own_big_day", "idio_absacf_lag1_mean", "idio_excess_kurtosis_median"]
    for k in keys:
        print("%-30s real %7.3f | " % (k, real[k]) + " | ".join("%s %7.3f [%6.3f,%6.3f]%s" % (
            v, res["variants"][v][k]["mean"], res["variants"][v][k]["p5"], res["variants"][v][k]["p95"],
            " " if res["variants"][v][k]["p5"] <= real[k] <= res["variants"][v][k]["p95"] else "*") for v in variants), flush=True)


if __name__ == "__main__":
    main()
