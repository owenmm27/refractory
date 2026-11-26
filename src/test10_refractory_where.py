"""Test 10 - Where does the refractory period live?

Test 9 found that since ~2000 the day after a big INDEX surprise is damped (next day ~53% of the variance the
volatility state implies after a 3-SD day). Two broad explanations make different predictions:

  * index-level flows (futures / ETF arbitrage, dealer hedging, volatility-targeting funds) damp the COMMON
    component only: individual stocks' idiosyncratic returns show no refractory period, and a big market day
    does not damp the next day's idiosyncratic variance;
  * general liquidity provision (market makers, short-term reversal traders working every name) damps any
    price after its own big surprise: idiosyncratic returns show the refractory period too, driven by each
    stock's OWN surprise.

Model (2-unit volatility state, Student-t), generalizing Test 9b:
    h_t = level_t * (1 + kappa*E_t) * exp(-rho_own*(s_own,t-1 - mean) - rho_mkt*(s_mkt,t-1 - mean))
    s_own = |r| / sqrt(level*(1+kappa*E)),  s_mkt = |r_m| / sqrt(h_m) from the index's own base fit,
    E = earnings reaction day (2005+; its variance bump is stripped from the persistent state).
Fitted to: the index, GICS sector portfolios, French 49 industries, and 237 stocks (raw and idiosyncratic =
residual on the S&P 500), in eras 1990-1999 / 2000-2012 / 2013-2026 and 2000-2026. Null: each stock's base
fit simulated (no refractory term) and re-estimated, to measure estimator bias.

Usage: python3 -I src/test10_refractory_where.py <project_root> [n_stocks]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from common import _unpack_multi, fit_multi, load_industries, load_market, load_stocks, simulate_multi
from test8_earnings import reaction_days
from test9_scheduled_news import t_ll

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
N_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
ERAS = {"1990-1999": ("1990", "1999"), "2000-2012": ("2000", "2012"), "2013-2026": ("2013", "2026"),
        "2000-2026": ("2000", "2026")}
EARN_ERA = ("2005", "2026")


# ----------------------------------------------------------------------------- model
def gpath(p, r, mask, own=False, S=None, E=None):
    omega, a, b, delta, nu = _unpack_multi(p[:7], 2)
    j = 7
    rho_own = 0.0
    if own:
        rho_own = p[j]
        j += 1
    k = 0 if S is None else S.shape[1]
    rho_exo = p[j:j + k]
    j += k
    bump = 1.0 if E is None else 1 + np.exp(p[j]) * E
    v = np.mean(r[mask] ** 2)
    sq = r ** 2 * (1 + delta * (r < 0)) / (1 + delta / 2) / bump
    u = np.r_[v, sq[:-1]]
    level = np.full(len(r), omega)
    for ak, bk in zip(a, b):
        level = level + lfilter([ak], [1.0, -bk], u, zi=[bk * ak / (1 - bk) * v])[0]
    lb = level * bump
    s = np.abs(r) / np.sqrt(lb)
    sm = float(np.mean(s[mask]))
    x = np.zeros(len(r))
    if own:
        s = np.where(mask, s, sm)
        x -= rho_own * (np.r_[sm, s[:-1]] - sm)
    if k:
        x -= S @ rho_exo
    return lb * np.exp(x), nu, sm


def _fit(r, mask, p0_list, **kw):
    def nll(p):
        h, nu, _ = gpath(p, r, mask, **kw)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        return -t_ll(r[mask], h[mask], nu).sum()
    best = None
    for p0 in p0_list:
        res = minimize(nll, p0, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    h, nu, sm = gpath(best.x, r, mask, **kw)
    return {"ll": float(-best.fun), "x": best.x, "z": r / np.sqrt(h), "s_mean": sm, "nu": float(nu)}


def base_fit(r, mask, E=None):
    f0 = fit_multi(r[mask], 2)
    p = np.asarray(f0["params_vec"], float)
    if E is None:
        return _fit(r, mask, [p])
    return _fit(r, mask, [np.r_[p, np.log(0.5)], np.r_[p, np.log(2.0)]], E=E)


def extend(r, mask, base, own=False, S=None, E=None):
    """Nested extension of `base`: one start exactly at the base solution, so LR >= 0 by construction."""
    k = 0 if S is None else S.shape[1]
    p_core, p_tail = base["x"][:7], base["x"][7:]          # p_tail = log kappa when E is given
    starts = [np.r_[p_core, [0.0] * (int(own) + k), p_tail]]
    if own:
        starts.append(np.r_[p_core, [0.3] + [0.0] * k, p_tail])
    f = _fit(r, mask, starts, own=own, S=S, E=E)
    j = 7
    if own:
        f["rho_own"] = float(f["x"][j])
        f["next_day_after_3sd_own"] = float(np.exp(-f["rho_own"] * (3 - f["s_mean"])))
        j += 1
    if k:
        f["rho_mkt"] = float(f["x"][j])
    if E is not None:
        f["kappa"] = float(np.exp(f["x"][-1]))
    f["LR"] = float(2 * (f["ll"] - base["ll"]))
    return f


def slim(f, keys=("ll", "LR", "rho_own", "rho_mkt", "next_day_after_3sd_own", "kappa", "nu", "s_mean")):
    return {k: f[k] for k in keys if k in f}


# ----------------------------------------------------------------------------- one series
def series_job(args):
    """args: name, x (pct returns Series, NaN allowed), m (market pct Series, same index), smkt {era: T x 1},
    smkt_mean {era: float}, E (float Series or None), do_idio, do_null, seed."""
    name, x, m, smkt, smkt_mean, E, do_idio, do_null, seed = args
    t0 = time.time()
    out = {"name": name, "eras": {}}
    zs = {}
    for era, (a, b) in ERAS.items():
        xe, me = x.loc[a:b], m.loc[a:b]
        mask = (xe.notna() & me.notna()).values
        if mask.sum() < 1500:
            continue
        r = np.where(mask, xe.values - np.nanmean(xe.values[mask]), 0.0)
        row = {"n": int(mask.sum())}
        base = base_fit(r, mask)
        own = extend(r, mask, base, own=True)
        row["raw"] = {"base_ll": base["ll"], "own": slim(own)}
        if era == "2000-2026":
            zs["raw"] = base["z"].astype(np.float32)
        if do_idio:
            beta, alpha = np.polyfit(me.values[mask], xe.values[mask], 1)
            e = np.where(mask, xe.values - alpha - beta * me.values, 0.0)
            S = smkt[era]
            ib = base_fit(e, mask)
            both = extend(e, mask, ib, own=True, S=S)
            row["idio"] = {"beta": float(beta), "base_ll": ib["ll"], "own_and_mkt": slim(both),
                           "next_day_after_3sd_mkt": float(np.exp(-both["rho_mkt"] * (3 - smkt_mean[era])))}
            if era == "2000-2026":
                row["idio"]["own_only"] = slim(extend(e, mask, ib, own=True))
                row["idio"]["mkt_only"] = slim(extend(e, mask, ib, S=S))
                zs["idio"] = ib["z"].astype(np.float32)
                if do_null:   # simulate the base fit (no refractory) and re-estimate
                    rng = np.random.default_rng(seed)
                    om, aa, bb, dd, nu = _unpack_multi(ib["x"][:7], 2)
                    y = simulate_multi({"omega": om, "a": aa, "b": bb, "delta": dd, "nu": nu}, len(e), 1, rng)[:, 0]
                    full = np.ones(len(y), bool)
                    nb = base_fit(y, full)
                    row["idio"]["null"] = slim(extend(y, full, nb, own=True, S=S))
        out["eras"][era] = row
    if do_idio and E is not None:     # earnings-controlled idiosyncratic fit
        a, b = EARN_ERA
        xe, me, Ee = x.loc[a:b], m.loc[a:b], E.loc[a:b].values
        mask = (xe.notna() & me.notna()).values
        beta, alpha = np.polyfit(me.values[mask], xe.values[mask], 1)
        e = np.where(mask, xe.values - alpha - beta * me.values, 0.0)
        S = smkt["earn"]
        ib = base_fit(e, mask)
        ibk = base_fit(e, mask, E=Ee)
        out["earnings"] = {"n_events": int(Ee.sum()), "no_control": slim(extend(e, mask, ib, own=True, S=S)),
                           "earnings_bump": {"base_ll": ibk["ll"], "kappa": float(np.exp(ibk["x"][-1]))},
                           "earnings_controlled": slim(extend(e, mask, ibk, own=True, S=S, E=Ee))}
    out["secs"] = round(time.time() - t0, 1)
    return out, zs


# ----------------------------------------------------------------------------- market surprise inputs
def market_surprise(m):
    """Lagged, centered s_mkt per era from the index's own base fit (T x 1 arrays), plus the means."""
    S, means = {}, {}
    for era, (a, b) in list(ERAS.items()) + [("earn", EARN_ERA)]:
        me = m.loc[a:b]
        mask = me.notna().values
        r = np.where(mask, me.values - np.nanmean(me.values), 0.0)
        f = base_fit(r, mask)
        s = np.where(mask, np.abs(f["z"]), np.nan)
        mu = float(np.nanmean(s))
        s = np.nan_to_num(s - mu)
        S[era] = np.r_[0.0, s[:-1]][:, None]
        means[era] = mu
    return S, means


def summarize(rows, path_fn):
    vals = [path_fn(r) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    rho = np.array([v[0] for v in vals])
    lr = np.array([v[1] for v in vals])
    nd = np.array([v[2] for v in vals]) if vals[0][2] is not None else None
    se = rho.std(ddof=1) / np.sqrt(len(rho))
    o = {"n": len(rho), "rho_mean": float(rho.mean()), "rho_se": float(se), "rho_median": float(np.median(rho)),
         "share_rho_positive": float(np.mean(rho > 0)), "share_LR_gt_3.84": float(np.mean(lr > 3.84)),
         "LR_sum": float(lr.sum())}
    if nd is not None:
        o["next_day_after_3sd_median"] = float(np.median(nd))
    return o


def get(d, *ks):
    for k in ks:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def main():
    OUT = ROOT / "results"
    st, spx = load_stocks(ROOT / "data")
    panel = st.loc[:, st.notna().mean() >= 0.98]
    panel = panel.loc[panel.notna().mean(axis=1) >= 0.95] * 100
    if N_LIMIT:
        panel = panel.iloc[:, :N_LIMIT]
    dates = panel.index
    m = (spx.reindex(dates) * 100)
    meta = pd.read_csv(ROOT / "data" / "meta" / "constituents.csv").set_index("Symbol")
    sectors = pd.Series({s: meta.loc[s, "GICS Sector"] for s in panel.columns})

    # earnings reaction days (Item 2.02 8-Ks)
    ev = pd.read_csv(ROOT / "data" / "earnings" / "earnings_8k_202.csv")
    ev = ev[ev["symbol"].isin(panel.columns)].copy()
    ev["pos"] = reaction_days(ev, pd.Series(dates))
    ev = ev[ev["pos"] >= 0]
    E = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    for sym, g in ev.groupby("symbol"):
        E.iloc[g["pos"].values, E.columns.get_loc(sym)] = 1.0

    print("market surprise inputs...", flush=True)
    S_spx, mu_spx = market_surprise(m)
    crsp = load_market(ROOT / "data") * 100
    S_crsp, mu_crsp = market_surprise(crsp)
    ind = load_industries(ROOT / "data") * 100
    ind = ind.where(ind > -99)

    jobs = []
    # index level
    jobs.append(("INDEX ^GSPC", m, m, S_spx, mu_spx, None, False, False, 0))
    jobs.append(("INDEX CRSP", crsp, crsp, S_crsp, mu_crsp, None, False, False, 0))
    # GICS sector equal-weight portfolios of the panel stocks
    for sec in sorted(sectors.unique()):
        cols = sectors.index[sectors == sec]
        jobs.append((f"GICS {sec}", panel[cols].mean(axis=1, skipna=True), m, S_spx, mu_spx, None, True, False, 0))
    # French 49 industries (value-weighted CRSP)
    for c in ind.columns:
        x = ind[c].reindex(crsp.index)
        jobs.append((f"FF49 {c}", x, crsp, S_crsp, mu_crsp, None, True, False, 0))
    # stocks
    for i, sym in enumerate(panel.columns):
        has_e = E[sym].sum() > 20
        jobs.append((f"STOCK {sym}", panel[sym], m, S_spx, mu_spx, E[sym] if has_e else None, True, True, 1000 + i))

    print(f"{len(jobs)} series ({panel.shape[1]} stocks); fitting...", flush=True)
    results, Z = [], {"raw": {}, "idio": {}}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for k, (o, zs) in enumerate(ex.map(series_job, jobs, chunksize=1)):
            results.append(o)
            for kind, z in zs.items():
                Z[kind][o["name"]] = z
            if k % 20 == 0 or k == len(jobs) - 1:
                print(f"  {k + 1}/{len(jobs)} {o['name']} {o['secs']}s elapsed {time.time() - t0:.0f}s", flush=True)
    (OUT / "test10_raw.json").write_text(json.dumps(results, default=float))
    d0 = dates[dates >= "2000-01-01"]
    np.savez_compressed(OUT / "cache" / "test10_z.npz",
                        **{f"{kind}|{n}": z for kind, dd in Z.items() for n, z in dd.items()},
                        dates_2000=np.array(d0.strftime("%Y-%m-%d"), dtype="U10"), earnings_2000=E.loc[d0].values.astype(np.int8),
                        symbols=np.array(list(panel.columns)))

    # ---------------- summaries by level and era
    groups = {"index": [r for r in results if r["name"].startswith("INDEX")],
              "gics_sector": [r for r in results if r["name"].startswith("GICS")],
              "ff49_industry": [r for r in results if r["name"].startswith("FF49")],
              "stock": [r for r in results if r["name"].startswith("STOCK")]}
    summ = {}
    for g, rows in groups.items():
        summ[g] = {}
        for era in ERAS:
            o = {"raw_own": summarize(rows, lambda r: (lambda f: None if f is None else (f["rho_own"], f["LR"], f["next_day_after_3sd_own"]))(get(r, "eras", era, "raw", "own")))}
            if g != "index":
                o["idio_own_given_mkt"] = summarize(rows, lambda r: (lambda f: None if f is None else (f["rho_own"], f["LR"], f["next_day_after_3sd_own"]))(get(r, "eras", era, "idio", "own_and_mkt")))
                o["idio_mkt"] = summarize(rows, lambda r: (lambda f: None if f is None else (f["rho_mkt"], f["LR"], None))(get(r, "eras", era, "idio", "own_and_mkt")))
            summ[g][era] = o
        if g != "index":
            summ[g]["2000-2026_extra"] = {
                "idio_own_only": summarize(rows, lambda r: (lambda f: None if f is None else (f["rho_own"], f["LR"], f["next_day_after_3sd_own"]))(get(r, "eras", "2000-2026", "idio", "own_only"))),
                "idio_mkt_only": summarize(rows, lambda r: (lambda f: None if f is None else (f["rho_mkt"], f["LR"], None))(get(r, "eras", "2000-2026", "idio", "mkt_only")))}
    stocks = groups["stock"]
    summ["stock_null"] = {
        "own": summarize(stocks, lambda r: (lambda f: None if f is None else (f["rho_own"], f["LR"], f["next_day_after_3sd_own"]))(get(r, "eras", "2000-2026", "idio", "null"))),
        "mkt": summarize(stocks, lambda r: (lambda f: None if f is None else (f["rho_mkt"], f["LR"], None))(get(r, "eras", "2000-2026", "idio", "null")))}
    summ["stock_earnings_2005_2026"] = {
        k: summarize(stocks, lambda r, k=k: (lambda f: None if f is None else (f["rho_own"], f["LR"], f["next_day_after_3sd_own"]))(get(r, "earnings", k)))
        for k in ("no_control", "earnings_controlled")}
    summ["stock_earnings_2005_2026"]["kappa_median"] = float(np.median([r["earnings"]["earnings_bump"]["kappa"] for r in stocks if "earnings" in r]))
    summ["index_by_series"] = {r["name"]: {era: get(r, "eras", era, "raw", "own") for era in ERAS} for r in groups["index"]}
    (OUT / "test10.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
