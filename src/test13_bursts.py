"""Test 13 - Why do collective bursts persist from day to day?

The market-brain model (Test 12) reproduces the index but not the day-to-day persistence of collective firing
(slope of today's firing count on yesterday's: 0.54 real vs 0.21 v1 / 0.29 v2). Part A locates the gap:

  A1. Split the persistence slope into REPEAT firers (stocks that fired yesterday and fire again today: their own
      aftershocks) and NEW firers (stocks that didn't fire yesterday: spread, or a shared driver).
  A2. Aftershock profile: after a stock's own big stock-specific move (|e| > 4 trailing SDs), its variance on each of
      the next 10 days relative to before; split by earnings vs other days.
  A3. Does yesterday's population firing predict a stock's firing today, beyond the stock's own firing yesterday?
Computed identically on real data and on simulated histories of the v1 and v2 models.

Usage: python3 -I src/test13_bursts.py <project_root> [n_paths]
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

import test12_brain_model as B
import test12c_model_v2 as C

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
N_PATHS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = ROOT / "results"
KF = 4.0


def fire_matrix(R):
    df = pd.DataFrame(R)
    scale = df.abs().rolling(252, min_periods=200).median().shift(1).values * 1.4826
    valid = np.isfinite(R) & np.isfinite(scale)
    fire = (np.abs(np.nan_to_num(R)) > KF * np.nan_to_num(scale, nan=np.inf)) & valid
    good = valid.sum(1) >= 0.9 * R.shape[1]
    return fire, valid, good


def idio(m, R):
    mc = m - m.mean()
    mask = np.isfinite(R)
    Rc = np.where(mask, R - np.nanmean(R, 0), 0.0)
    b = (Rc * mc[:, None]).sum(0) / ((mc ** 2)[:, None] * mask).sum(0)
    return np.where(mask, Rc - mc[:, None] * b, np.nan)


def burst_stats(m, R, E):
    out = {}
    fire, valid, good = fire_matrix(R)
    Cn = fire.sum(1).astype(float)
    ok = good[1:] & good[:-1]
    x = Cn[:-1][ok]
    vx = x.var()
    rep = (fire[1:] & fire[:-1]).sum(1)[ok].astype(float)
    new = (fire[1:] & ~fire[:-1]).sum(1)[ok].astype(float)
    cov = lambda a: float(np.cov(a, x)[0, 1] / vx)
    out["slope_total"] = cov(Cn[1:][ok])
    out["slope_repeat"] = cov(rep)
    out["slope_new"] = cov(new)
    out["mean_firing"] = float(Cn[good].mean())
    # own persistence: P(fire today | fired yesterday) and | not fired yesterday
    f0, f1 = fire[:-1], fire[1:]
    v = valid[:-1] & valid[1:] & ok_rows(ok, fire.shape[1])
    out["p_fire_given_fired"] = float((f1 & f0 & v).sum() / max((f0 & v).sum(), 1))
    out["p_fire_given_not"] = float((f1 & ~f0 & v).sum() / max((~f0 & v).sum(), 1))
    # A3: among stocks that did NOT fire yesterday, P(fire today) by yesterday's population firing (quintiles)
    pop = np.r_[np.nan, Cn[:-1]]
    q = np.nanquantile(pop[1:][ok], [0.5, 0.9, 0.98])
    rows = {}
    for lab, lo, hi in [("pop_below_median", -1, q[0]), ("pop_50_90", q[0], q[1]), ("pop_90_98", q[1], q[2]), ("pop_top2", q[2], 1e9)]:
        day = ok & (pop[1:] > lo) & (pop[1:] <= hi)
        sel = (~f0) & v & day[:, None]
        rows[lab] = float((f1 & sel).sum() / max(sel.sum(), 1))
    out["p_new_fire_by_pop_yesterday"] = rows
    # A2: aftershock profile after own big idiosyncratic day
    e = idio(m, R)
    e2 = e ** 2
    sig2 = pd.DataFrame(e2).rolling(21, min_periods=15).mean().shift(1).values
    big = np.abs(e) > 4 * np.sqrt(sig2)
    T = len(m)
    for lab, cond in [("all", np.ones_like(big)), ("earnings", E.astype(bool)), ("other", ~E.astype(bool))]:
        ti, ni = np.where(big & cond)
        keep = ti < T - 11
        ti, ni = ti[keep], ni[keep]
        prof = [float(np.nanmean(e2[ti + k, ni] / sig2[ti, ni])) for k in range(1, 11)]
        out[f"aftershock_{lab}"] = prof
        out[f"aftershock_{lab}_n"] = int(len(ti))
    return out


def ok_rows(ok, N):
    return np.repeat(ok[:, None], N, axis=1)


_D = {}


def job(args):
    kind, seed = args
    if not _D:
        _D["P1"] = pickle.load(open(B.CACHE, "rb"))
        _D["Q"] = pickle.load(open(C.CACHE2, "rb"))
        panel, m_s, _, E = B.load_panel()
        _D["mask"], _D["E"] = np.isfinite(panel.values), E
    if kind == "v1":
        m, R = B.simulate(_D["P1"], "full", seed, _D["mask"], _D["E"])
    else:
        m, R = C.simulate2(_D["P1"], _D["Q"], "v2_full", seed, _D["mask"], _D["E"])
    return kind, burst_stats(m, R, _D["E"])


def summarize(rows):
    out = {}
    for k in rows[0]:
        v = rows[0][k]
        if isinstance(v, dict):
            out[k] = {kk: float(np.mean([r[k][kk] for r in rows])) for kk in v}
        elif isinstance(v, list):
            out[k] = np.mean([r[k] for r in rows], axis=0).round(3).tolist()
        else:
            arr = np.array([r[k] for r in rows])
            out[k] = {"mean": float(arr.mean()), "p5": float(np.quantile(arr, 0.05)), "p95": float(np.quantile(arr, 0.95))}
    return out


def main():
    panel, m_s, _, E = B.load_panel()
    real = burst_stats(m_s.values, panel.values, E)
    print("REAL", json.dumps(real), flush=True)
    jobs = [(k, 8000 + 37 * i + (0 if k == "v1" else 5000)) for k in ("v1", "v2") for i in range(N_PATHS)]
    res = {"v1": [], "v2": []}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for kind, st in ex.map(job, jobs):
            res[kind].append(st)
    print(f"sims {time.time() - t0:.0f}s", flush=True)
    out = {"real": real, "v1": summarize(res["v1"]), "v2": summarize(res["v2"]), "n_paths": N_PATHS}
    for k in ("v1", "v2"):
        print(k, json.dumps(out[k]), flush=True)
    (OUT / "test13a.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
