"""Test 10b - Model-light check of where the refractory period lives (2000-2026).

Uses the base 2-unit residuals z saved by test10 (no refractory term in the model). If the model were right,
today's z^2 would average 1 whatever yesterday's surprise was. A refractory period shows up as today's z^2
falling below 1 after a big surprise yesterday.

  A. Today's z^2 by yesterday's |z| bin: index, GICS sectors (pooled), stocks raw and idiosyncratic (pooled).
  B. Stocks' idiosyncratic surprises split by whether yesterday was an earnings reaction day (2005+).
  C. Cross-level: today's idiosyncratic z^2 (all stocks) by yesterday's MARKET surprise.
Confidence intervals: day-block bootstrap (21-day blocks), so cross-stock correlation on a day is respected.

Usage: python3 -I src/test10b_modelfree.py <project_root>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
BINS = [0, 1, 2, 3, np.inf]
LABELS = ["0-1", "1-2", "2-3", "3+"]
rng = np.random.default_rng(7)


def daily_sums(prev_abs, today_sq, valid):
    """prev_abs, today_sq, valid: (T-1) x N. Returns per-day sums and counts per bin, shape (T-1, nbins)."""
    k = np.digitize(prev_abs, BINS[1:-1])
    S = np.zeros((prev_abs.shape[0], len(LABELS)))
    C = np.zeros_like(S)
    for b in range(len(LABELS)):
        sel = valid & (k == b)
        S[:, b] = np.where(sel, today_sq, 0).sum(1)
        C[:, b] = sel.sum(1)
    return S, C


def ratios(S, C, n_boot=500, block=21):
    point = S.sum(0) / np.maximum(C.sum(0), 1)
    T = S.shape[0]
    nb = int(np.ceil(T / block))
    starts = np.arange(0, T - block + 1)
    boot = np.empty((n_boot, S.shape[1]))
    for i in range(n_boot):
        idx = (rng.choice(starts, nb)[:, None] + np.arange(block)[None, :]).ravel()[:T]
        boot[i] = S[idx].sum(0) / np.maximum(C[idx].sum(0), 1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {lab: {"z2_today": float(p), "ci95": [float(a), float(b)], "n": int(c)}
            for lab, p, a, b, c in zip(LABELS, point, lo, hi, C.sum(0))}


def panel(Z):
    """Z: T x N residuals (0 = missing). Returns prev |z|, today z^2, validity mask."""
    v = Z != 0
    return np.abs(Z[:-1]), Z[1:] ** 2, v[:-1] & v[1:]


def main():
    d = np.load(ROOT / "results" / "cache" / "test10_z.npz", allow_pickle=True)   # our own cache file
    syms = list(d["symbols"])
    E = d["earnings_2000"].astype(bool)
    dates = d["dates_2000"]
    T = len(dates)
    stock_raw = np.column_stack([d[f"raw|STOCK {s}"] for s in syms]).astype(float)
    stock_idio = np.column_stack([d[f"idio|STOCK {s}"] for s in syms]).astype(float)
    gics = np.column_stack([d[k] for k in d.files if k.startswith("raw|GICS")]).astype(float)
    idx = d["raw|INDEX ^GSPC"].astype(float)[:, None]
    assert stock_raw.shape[0] == T == idx.shape[0]
    out = {"sample": [str(dates[0]), str(dates[-1])], "n_stocks": len(syms)}

    # A. by level
    out["A_by_level"] = {}
    for name, Z in [("index_sp500", idx), ("gics_sectors_pooled", gics), ("stocks_raw_pooled", stock_raw),
                    ("stocks_idio_pooled", stock_idio)]:
        out["A_by_level"][name] = ratios(*daily_sums(*panel(Z)))
        print(name, {k: round(v["z2_today"], 3) for k, v in out["A_by_level"][name].items()}, flush=True)

    # B. earnings vs other days (2005+, idiosyncratic)
    k0 = int(np.searchsorted(dates, "2005-01-01"))
    pa, tq, ok = panel(stock_idio[k0:])
    Ey, Et = E[k0:][:-1], E[k0:][1:]
    out["B_idio_2005_2026"] = {
        "yesterday_not_earnings_today_not_earnings": ratios(*daily_sums(pa, tq, ok & ~Ey & ~Et)),
        "yesterday_earnings_day": ratios(*daily_sums(pa, tq, ok & Ey)),
        "n_earnings_days": int(E[k0:].sum())}
    for k in ("yesterday_not_earnings_today_not_earnings", "yesterday_earnings_day"):
        print(k, {b: round(v["z2_today"], 3) for b, v in out["B_idio_2005_2026"][k].items()}, flush=True)

    # C. yesterday's market surprise -> today's idiosyncratic variance (all stocks)
    m_prev = np.repeat(np.abs(idx[:-1]), stock_idio.shape[1], axis=1)
    _, tq, ok = panel(stock_idio)
    out["C_idio_today_by_market_yesterday"] = ratios(*daily_sums(m_prev, tq, ok))
    # and the same with stocks whose own surprise yesterday was small (|z_own| < 1), isolating the market channel
    pa, tq, ok = panel(stock_idio)
    out["C_idio_today_by_market_yesterday_own_small"] = ratios(*daily_sums(m_prev, tq, ok & (pa < 1)))
    for k in ("C_idio_today_by_market_yesterday", "C_idio_today_by_market_yesterday_own_small"):
        print(k, {b: round(v["z2_today"], 3) for b, v in out[k].items()}, flush=True)
    # D. dilution: refractory strength (own-surprise fit on raw returns) vs the share of variance the market explains
    out["D_dilution_spearman_R2_vs_rho"] = dilution()
    (ROOT / "results" / "test10b.json").write_text(json.dumps(out, indent=2))


def dilution():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd
    from scipy.stats import spearmanr
    from common import load_industries, load_market, load_stocks
    raw = json.loads((ROOT / "results" / "test10_raw.json").read_text())
    st, spx = load_stocks(ROOT / "data")
    pnl = st.loc[:, st.notna().mean() >= 0.98]
    pnl = pnl.loc[pnl.notna().mean(axis=1) >= 0.95] * 100
    m = spx.reindex(pnl.index) * 100
    meta = pd.read_csv(ROOT / "data" / "meta" / "constituents.csv").set_index("Symbol")
    sec = pd.Series({s: meta.loc[s, "GICS Sector"] for s in pnl.columns})
    ser = {f"STOCK {s}": (pnl[s], m) for s in pnl.columns}
    for g in sec.unique():
        ser[f"GICS {g}"] = (pnl[sec.index[sec == g]].mean(axis=1), m)
    ind, crsp = load_industries(ROOT / "data") * 100, load_market(ROOT / "data") * 100
    for c in ind.columns:
        ser[f"FF49 {c}"] = (ind[c].where(ind[c] > -99).reindex(crsp.index), crsp)
    ser["INDEX ^GSPC"] = (m, m)
    rows = []
    for r in raw:
        if r["name"] not in ser:
            continue
        for era, row in r["eras"].items():
            a, b = era.split("-")
            x, mm = ser[r["name"]]
            x, mm = x.loc[a:b], mm.loc[a:b]
            ok = x.notna() & mm.notna()
            own = row["raw"]["own"]
            rows.append({"name": r["name"], "level": r["name"].split()[0], "era": era,
                         "R2": float(np.corrcoef(x[ok], mm[ok])[0, 1] ** 2), "rho": own["rho_own"],
                         "next_day_after_3sd": own["next_day_after_3sd_own"]})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "test10_dilution.csv", index=False)
    res = {}
    for era, g in df[df["level"] != "INDEX"].groupby("era"):
        res[era] = {"spearman": float(spearmanr(g["R2"], g["rho"]).statistic), "n": int(len(g)),
                    "rho_mean_by_R2_bin": {str(k): float(v) for k, v in
                                           g.groupby(pd.cut(g["R2"], [0, .2, .3, .4, .5, .7, .9]), observed=True)["rho"].mean().items()}}
    print("D", json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    main()
