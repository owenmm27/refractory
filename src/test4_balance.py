"""Test 4 - Feedback balance: when does the market trend (positive feedback) vs revert (negative feedback)?

Measures, all on daily index returns:
  - lag-1 autocorrelation and variance ratios VR(q) = Var(q-day) / (q * Var(1-day)) by volatility regime
    (VR > 1: positive feedback / trending; VR < 1: negative feedback / mean-reverting)
  - drift of the balance over a century (rolling 1-year lag-1 autocorrelation)
  - continuation vs reversal after large moves, by regime
  - the same autocorrelations on 49 industries and ~500 stocks (is reversal a unit-level or index-level property?)

Usage: python3 -I src/test4_balance.py <project_root>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import load_industries, load_market, load_stocks, load_vix

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"
ERAS = [("1926-1969", "1926", "1969"), ("1970-1999", "1970", "1999"), ("2000-2026", "2000", "2026")]


def ac1(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    return float((x[1:] * x[:-1]).sum() / (x ** 2).sum())


def pair_ac1(r, mask):
    """Lag-1 autocorrelation using only pairs (r[t-1], r[t]) where mask[t] (regime known at t-1) holds."""
    x = r - r.mean()
    prev, cur = x[:-1], x[1:]
    m = mask[1:]
    return float((prev[m] * cur[m]).sum() / np.sqrt((prev[m] ** 2).sum() * (cur[m] ** 2).sum())), int(m.sum())


def variance_ratio(r, q):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)] - np.nanmean(r)
    rq = np.convolve(r, np.ones(q), "valid")
    return float(rq.var() / (q * r.var()))


def block_bootstrap_se(stat_fn, r, block=21, B=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(r)
    nb = n // block
    vals = []
    for _ in range(B):
        starts = rng.integers(0, n - block, nb)
        vals.append(stat_fn(np.concatenate([r[s:s + block] for s in starts])))
    return float(np.std(vals))


def regime_table(r, dates, vol_state, label):
    """Split days into volatility terciles (state known the day BEFORE) and measure feedback in each."""
    out = {}
    valid = np.isfinite(vol_state)
    terc = pd.qcut(pd.Series(vol_state[valid]), 3, labels=False).values
    t_full = np.full(len(r), -1)
    t_full[valid] = terc
    for k, name in enumerate(["calm", "normal", "turbulent"]):
        mask = t_full == k
        rho, n = pair_ac1(r, mask)
        # variance ratios computed on contiguous runs would be messy; use regime of the block start instead
        out[name] = {"ac1": rho, "n_pairs": n, "se_iid": float(1 / np.sqrt(n))}
    # VR by regime on 21-day blocks: regime at block start, VR(5) within non-overlapping 21-day blocks pooled
    blocks = []
    for s in range(0, len(r) - 21, 21):
        if t_full[s] >= 0:
            blocks.append((t_full[s], r[s:s + 21]))
    for k, name in enumerate(["calm", "normal", "turbulent"]):
        segs = [b for t, b in blocks if t == k]
        x = np.concatenate([b - b.mean() for b in segs])
        r5 = np.concatenate([np.convolve(b - b.mean(), np.ones(5), "valid")[::5] for b in segs])
        out[name]["VR5_within_blocks"] = float(r5.var() / (5 * x.var()))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    mkt = load_market(ROOT / "data")
    vix = load_vix(ROOT / "data")
    ind = load_industries(ROOT / "data")
    st, spx = load_stocks(ROOT / "data")
    res = {}

    r = mkt.values
    dates = mkt.index
    # volatility state known at the close of t-1: trailing 63-day realized vol, shifted so row t uses data to t-1
    vol63 = pd.Series(r, index=dates).rolling(63).std().shift(1).values

    # 1. full-history balance by regime
    res["market_by_vol_regime_full"] = regime_table(r, dates, vol63, "mkt")
    res["market_by_vol_regime_era"] = {}
    for name, a, b in ERAS:
        m = np.asarray((dates >= a) & (dates <= b + "-12-31"))
        res["market_by_vol_regime_era"][name] = regime_table(r[m], dates[m], vol63[m], name)

    # 2. variance ratios by era with block-bootstrap SEs
    res["variance_ratios_by_era"] = {}
    for name, a, b in ERAS:
        x = r[np.asarray((dates >= a) & (dates <= b + "-12-31"))]
        row = {}
        for q in (2, 5, 21, 63):
            row[f"VR{q}"] = variance_ratio(x, q)
            row[f"VR{q}_se"] = block_bootstrap_se(lambda y, q=q: variance_ratio(y, q), x, block=126, B=200)
        res["variance_ratios_by_era"][name] = row

    # 3. rolling 1-year lag-1 autocorrelation (the drifting balance) + rolling VR(5)
    s = pd.Series(r, index=dates)
    roll_ac1 = s.rolling(252).apply(ac1, raw=True)
    roll_vr5 = s.rolling(504).apply(lambda x: variance_ratio(x, 5), raw=True)
    pd.DataFrame({"ac1_252": roll_ac1, "vr5_504": roll_vr5}).to_csv(OUT / "test4_rolling.csv")
    yr = roll_ac1.resample("YE").last()
    res["rolling_ac1_decade_means"] = {f"{d}s": float(roll_ac1.loc[str(d):str(d + 9)].mean()) for d in range(1930, 2030, 10)}
    res["ac1_first_negative_sustained_year"] = next((int(y.year) for y, v in yr.items()
                                                     if y.year > 1950 and (yr.loc[y:].iloc[:5] < 0).all()), None)

    # 4. after large moves: next-day return conditional on today's move size, by regime
    z = pd.Series(r, index=dates) / pd.Series(vol63, index=dates)
    nxt = pd.Series(r, index=dates).shift(-1)
    df = pd.DataFrame({"z": z, "next": nxt, "vol": vol63}).dropna()
    df["regime"] = pd.qcut(df["vol"], 3, labels=["calm", "normal", "turbulent"])
    df["era"] = pd.cut(df.index.year, [1925, 1969, 1999, 2026], labels=["1926-1969", "1970-1999", "2000-2026"])
    rows = {}
    for (era, reg), g in df.groupby(["era", "regime"], observed=True):
        big_up, big_dn = g[g["z"] > 2], g[g["z"] < -2]
        rows[f"{era} / {reg}"] = {
            "n_big_up": int(len(big_up)), "next_after_big_up_bp": float(big_up["next"].mean() * 1e4),
            "n_big_down": int(len(big_dn)), "next_after_big_down_bp": float(big_dn["next"].mean() * 1e4),
            "uncond_next_bp": float(g["next"].mean() * 1e4)}
    res["after_large_moves"] = rows

    # 5. VIX regime (1990+): does option-implied fear separate trending vs reverting days?
    v = np.log(vix.reindex(dates).shift(1)).values
    m90 = np.asarray(dates >= "1990-01-03")
    res["market_by_vix_regime_1990on"] = regime_table(r[m90], dates[m90], v[m90], "vix")

    # 6. units vs index: lag-1 autocorrelation of industries and individual stocks, by era
    res["units_vs_index"] = {}
    for name, a, b in ERAS:
        sl = slice(a, b + "-12-31")
        ia = [ac1(ind.loc[sl, c].values) for c in ind.columns if ind.loc[sl, c].notna().sum() > 1000]
        row = {"index_ac1": ac1(mkt.loc[sl].values), "industry_ac1_median": float(np.median(ia)),
               "industry_ac1_share_negative": float(np.mean(np.array(ia) < 0)),
               "equal_weight_industry_index_ac1": ac1(ind.loc[sl].mean(axis=1).values)}
        if a >= "1990" or b >= "1999":
            ss = st.loc[max(a, "1990"):b + "-12-31"]
            sa = [ac1(ss[c].values) for c in ss.columns if ss[c].notna().sum() > 1000]
            row.update({"stock_ac1_median": float(np.median(sa)), "stock_ac1_share_negative": float(np.mean(np.array(sa) < 0)),
                        "spx_ac1": ac1(spx.loc[max(a, "1990"):b + "-12-31"].values)})
        res["units_vs_index"][name] = row

    # 7. cross-autocorrelation: does yesterday's market move predict today's average industry move (lead-lag)?
    ind_ew = ind.mean(axis=1)
    res["lead_lag"] = {}
    for name, a, b in ERAS:
        sl = slice(a, b + "-12-31")
        x, y = mkt.loc[sl], ind_ew.loc[sl]
        res["lead_lag"][name] = {"corr(mkt_t-1, ewind_t)": float(np.corrcoef(x.values[:-1], y.values[1:])[0, 1]),
                                 "corr(ewind_t-1, mkt_t)": float(np.corrcoef(y.values[:-1], x.values[1:])[0, 1])}

    (OUT / "test4.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
