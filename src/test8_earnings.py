"""Test 8 - Is the sector volatility leadership (Test 7) and the leftover firing persistence (Tests 3/6)
just the earnings calendar?

Data: earnings-release dates from SEC 8-K Item 2.02 filings (src/download_earnings.py), 2005-2026.
Reaction day: the filing day if accepted before 16:00 New York time, else the next trading day.

A. Validation: are reaction days where the big idiosyncratic moves are?
B. Reporting order: do early reporters (days after quarter end) lead in volatility?
C. Remove earnings windows (each stock's reaction day + next day) and re-measure directed structure;
   also split earnings-season vs off-season days.
D. Leftover firing persistence: real vs a real-volatility null (shocks bootstrapped from each stock's own
   surprises, so tails match), with and without earnings windows.
E. Event study: after a big earnings surprise at stock j, is the next day more volatile for peers that
   haven't reported yet this quarter (information transfer) vs those that already have?

Usage: python3 -I src/test8_earnings.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import load_stocks
from test7_contagion import antisym, group_flow, lagcorr, standardize, upper

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"
START = "2005-01-01"
SPLIT = "2016-01-01"
K = 4.0


def reaction_days(ev, trade_days):
    acc = pd.to_datetime(ev["accepted_utc"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    fd = pd.to_datetime(ev["filing_date"])
    after_close = (acc.dt.hour >= 16).fillna(False).values
    pos = np.searchsorted(trade_days.values, fd.values)            # first trading day >= filing date
    on_trading_day = (pos < len(trade_days)) & (trade_days.values[np.minimum(pos, len(trade_days) - 1)] == fd.values)
    pos = pos + (after_close & on_trading_day)
    ok = pos < len(trade_days)
    return pd.Series(np.where(ok, pos, -1), index=ev.index)


def firing_rate(R, excl=None):
    df = pd.DataFrame(R)
    scale = df.abs().rolling(252, min_periods=200).median().shift(1).values * 1.4826
    valid = np.isfinite(R) & np.isfinite(scale)
    if excl is not None:
        valid &= ~excl
    fire = (np.abs(np.nan_to_num(R)) > K * np.nan_to_num(scale, nan=np.inf)) & valid
    n = valid.sum(1)
    rate = np.where(n > 0.8 * R.shape[1], fire.sum(1) / np.maximum(n, 1), np.nan)
    return rate


def lag1_slope(x):
    x = x[np.isfinite(x)]
    return float(np.polyfit(x[:-1], x[1:], 1)[0])


def _null_rep(args):
    seed, systematic, sdv, pools, mask, excl = args
    rng = np.random.default_rng(seed)
    T, N = sdv.shape
    eps = np.column_stack([rng.choice(pools[i], T) for i in range(N)])
    R = systematic + sdv * eps
    R[~mask] = np.nan
    return lag1_slope(firing_rate(R)), lag1_slope(firing_rate(R, excl))


def main():
    res = {}
    # ---------------- data
    d = np.load(OUT / "cache" / "z_2unit.npz", allow_pickle=True)
    z_all, syms, dates_all = d["z"], d["symbols"], pd.to_datetime(d["dates"])
    st, _ = load_stocks(ROOT / "data")
    R_all = st.loc[dates_all, syms].values * 100
    keep = dates_all >= START
    z, R, dates = z_all[keep], R_all[keep], dates_all[keep]
    T, N = z.shape
    meta = pd.read_csv(ROOT / "data" / "meta" / "constituents.csv").set_index("Symbol")
    sectors = np.array([meta.loc[s, "GICS Sector"] for s in syms])
    split = int(np.searchsorted(dates, pd.Timestamp(SPLIT)))

    ev = pd.read_csv(ROOT / "data" / "earnings" / "earnings_8k_202.csv")
    ev = ev[ev["filing_date"] >= START].copy()
    ev["pos"] = reaction_days(ev, pd.Series(dates))
    ev = ev[ev["pos"] >= 0]
    col = {s: i for i, s in enumerate(syms)}
    ev["i"] = ev["symbol"].map(col)
    ev = ev.dropna(subset=["i"]).astype({"i": int}).sort_values(["i", "pos"])
    ev = ev[ev.groupby("i")["pos"].diff().fillna(999) > 10]          # collapse repeat filings within 2 weeks
    E = np.zeros((T, N), bool)
    E[ev["pos"].values, ev["i"].values] = True
    W = E.copy()
    W[1:] |= E[:-1]                                                   # earnings window = reaction day + next day
    res["n_events"] = int(len(ev))
    res["events_per_stock_year_median"] = float(ev.groupby("i").size().median() / ((dates[-1] - dates[0]).days / 365.25))

    # ---------------- A. validation
    az = np.abs(z)
    base = np.nanmean(az[~W])
    prof = {}
    for off in (-2, -1, 0, 1, 2, 3):
        p = ev["pos"].values + off
        ok = (p >= 0) & (p < T)
        prof[f"day{off:+d}"] = float(np.nanmean(az[p[ok], ev["i"].values[ok]]) / base)
    res["A_abs_surprise_vs_normal_days"] = prof
    res["A_share_of_firings_in_earnings_windows"] = None  # filled below with the firing matrix

    # ---------------- B. reporting order
    rd = dates[ev["pos"].values]
    qend = pd.DatetimeIndex(rd).to_period("Q").start_time - pd.Timedelta(days=1)
    ev["lag_days"] = (rd - qend).days.values
    ev_in = ev[(ev["lag_days"] >= 0) & (ev["lag_days"] < 92)]
    lag_stock = ev_in.groupby("i")["lag_days"].median().reindex(range(N)).values
    leaders = {}
    for kind, raw in [("magnitude", az), ("direction", z)]:
        X = standardize(raw)
        A = antisym(lagcorr(X))
        leaders[kind] = A.sum(0)
    lag_sector = pd.Series(lag_stock).groupby(sectors).median()
    from scipy.stats import spearmanr
    sec_flow = {}
    for kind, raw in [("magnitude", az), ("direction", z)]:
        C = lagcorr(standardize(raw))
        sec_flow[kind] = pd.Series({s: group_flow(C, np.where(sectors == s)[0], np.where(sectors != s)[0]) for s in np.unique(sectors)})
    on_cycle = lag_stock < 50
    res["B_reporting_order"] = {
        "sector_median_lag_days": {k: float(v) for k, v in lag_sector.sort_values().items()},
        "stock_corr_lag_vs_leadership": {k: float(spearmanr(lag_stock[on_cycle], leaders[k][on_cycle]).statistic) for k in leaders},
        "stock_corr_p": {k: float(spearmanr(lag_stock[on_cycle], leaders[k][on_cycle]).pvalue) for k in leaders},
        "n_on_cycle_stocks": int(on_cycle.sum()),
        "sector_spearman_lag_vs_leadership": {k: float(spearmanr(lag_sector.values, sec_flow[k].reindex(lag_sector.index).values).statistic) for k in sec_flow},
        "sector_spearman_p": {k: float(spearmanr(lag_sector.values, sec_flow[k].reindex(lag_sector.index).values).pvalue) for k in sec_flow},
        "sector_flow_magnitude_2005_2026": {k: float(v) for k, v in sec_flow["magnitude"].sort_values(ascending=False).items()}}

    # ---------------- C. remove earnings windows; season vs off-season
    season_frac = pd.Series(E.mean(1)).rolling(11, center=True, min_periods=1).mean().values
    in_season = season_frac > np.quantile(season_frac, 0.6)
    res["C_share_of_days_in_season"] = float(in_season.mean())
    out = {}
    for kind in ("magnitude", "direction"):
        base_raw = az if kind == "magnitude" else z
        variants = {"all_days": base_raw, "earnings_windows_removed": np.where(W, np.nan, base_raw)}
        for vname, raw in variants.items():
            Ca, Cb, Cf = (lagcorr(standardize(raw[s])) for s in (slice(None, split), slice(split, None), slice(None)))
            flows = np.array([[group_flow(C, np.where(sectors == s)[0], np.where(sectors != s)[0]) for C in (Ca, Cb, Cf)]
                              for s in np.unique(sectors)])
            Aa, Ab = antisym(Ca), antisym(Cb)
            off = ~np.eye(N, dtype=bool)
            out[f"{kind}/{vname}"] = {
                "sector_flow_halves_corr": float(np.corrcoef(flows[:, 0], flows[:, 1])[0, 1]),
                "sector_flow_spread_sd_x1000": float(flows[:, 2].std() * 1000),
                "stock_leadership_halves_corr": float(np.corrcoef(Aa.sum(0), Ab.sum(0))[0, 1]),
                "pairs_halves_corr": float(np.corrcoef(upper(Aa), upper(Ab))[0, 1]),
                "mean_cross_link": float(Cf[off].mean()),
                "sector_flows_full": {s: float(v) for s, v in zip(np.unique(sectors), flows[:, 2])}}
        # season split: links where day t is in season vs off season (all days, unmasked)
        X = standardize(base_raw)
        for sname, m in [("in_season", in_season[1:]), ("off_season", ~in_season[1:])]:
            Cs = X[1:][m].T @ X[:-1][m] / m.sum()
            fl = np.array([group_flow(Cs, np.where(sectors == s)[0], np.where(sectors != s)[0]) for s in np.unique(sectors)])
            out[f"{kind}/{sname}"] = {"sector_flow_spread_sd_x1000": float(fl.std() * 1000),
                                      "corr_with_full_sector_flows": float(np.corrcoef(fl, out[f"{kind}/all_days"]["sector_flows_full"].values()
                                                                                       if False else list(out[f"{kind}/all_days"]["sector_flows_full"].values()))[0, 1]),
                                      "mean_cross_link": float(Cs[~np.eye(N, dtype=bool)].mean())}
    res["C_directed_structure"] = out

    # ---------------- D. firing persistence with / without earnings windows
    mask = np.isfinite(R)
    mkt = np.nanmean(R, axis=1)
    systematic, idio = np.empty_like(R), np.empty_like(R)
    for i in range(N):
        ok = mask[:, i]
        b = np.polyfit(mkt[ok], R[ok, i], 1)
        systematic[:, i] = b[1] + b[0] * mkt
        idio[:, i] = R[:, i] - systematic[:, i]
    sdv = pd.DataFrame(np.abs(idio / z)).ffill().bfill().values          # each stock's real 2-unit volatility path
    real_full, real_excl = lag1_slope(firing_rate(R)), lag1_slope(firing_rate(R, W))
    fire_real = None
    df = pd.DataFrame(R)
    scale = df.abs().rolling(252, min_periods=200).median().shift(1).values * 1.4826
    fire_real = (np.abs(np.nan_to_num(R)) > K * np.nan_to_num(scale, nan=np.inf)) & mask
    res["A_share_of_firings_in_earnings_windows"] = float((fire_real & W).sum() / fire_real.sum())
    res["A_share_of_stock_days_in_earnings_windows"] = float(W[mask].mean())
    nulls = {}
    for pname, pool_mask in [("shocks_from_all_days", np.ones_like(W)), ("shocks_from_non_earnings_days", ~W)]:
        pools = [z[:, i][np.isfinite(z[:, i]) & pool_mask[:, i]] for i in range(N)]
        with ProcessPoolExecutor(max_workers=4) as ex:
            reps = np.array(list(ex.map(_null_rep, [(8000 + r, systematic, sdv, pools, mask, W) for r in range(20)])))
        nulls[pname] = {"lag1_all_days": [float(reps[:, 0].mean()), float(np.quantile(reps[:, 0], 0.025)), float(np.quantile(reps[:, 0], 0.975))],
                        "lag1_earnings_windows_removed": [float(reps[:, 1].mean()), float(np.quantile(reps[:, 1], 0.025)), float(np.quantile(reps[:, 1], 0.975))]}
    res["D_firing_persistence"] = {"real_all_days": real_full, "real_earnings_windows_removed": real_excl, "nulls": nulls}
    res["D_firing_rate_season_vs_off"] = {"in_season": float(np.nanmean(firing_rate(R)[in_season])),
                                          "off_season": float(np.nanmean(firing_rate(R)[~in_season])),
                                          "in_season_excl_windows": float(np.nanmean(firing_rate(R, W)[in_season])),
                                          "off_season_excl_windows": float(np.nanmean(firing_rate(R, W)[~in_season]))}

    # ---------------- E. information transfer event study
    reported_q = np.zeros((T, N), bool)                     # has stock k reported since the last quarter end?
    qid = pd.DatetimeIndex(dates).to_period("Q").astype(str).values
    for i in range(N):
        seen, q_prev = False, None
        for t in range(T):
            if qid[t] != q_prev:
                seen, q_prev = False, qid[t]
            reported_q[t, i] = seen
            if E[t, i]:
                seen = True
    base_k = np.array([np.nanmean(az[~W[:, k], k]) for k in range(N)])
    evs = ev[(ev["pos"] + 1 < T)].copy()
    evs["surprise"] = az[evs["pos"].values, evs["i"].values]
    evs = evs.dropna(subset=["surprise"])
    q_hi, q_lo = evs["surprise"].quantile([0.8, 0.2])
    acc = {}
    for label, sub in [("big_surprise", evs[evs["surprise"] >= q_hi]), ("small_surprise", evs[evs["surprise"] <= q_lo])]:
        sums = {}
        for pos, j in zip(sub["pos"].values, sub["i"].values):
            t1 = pos + 1
            peers = np.ones(N, bool)
            peers[j] = False
            peers &= ~W[t1] & ~W[pos] & np.isfinite(az[t1])
            same = sectors == sectors[j]
            for gname, g in [("same_sector_not_yet_reported", same & ~reported_q[t1]), ("same_sector_already_reported", same & reported_q[t1]),
                             ("other_sector_not_yet_reported", ~same & ~reported_q[t1]), ("other_sector_already_reported", ~same & reported_q[t1])]:
                m = peers & g
                if m.any():
                    s = sums.setdefault(gname, [0.0, 0])
                    s[0] += float((az[t1, m] / base_k[m]).sum())
                    s[1] += int(m.sum())
        acc[label] = {g: {"next_day_vol_vs_normal": v[0] / v[1], "n_peer_days": v[1]} for g, v in sums.items()}
    acc["difference_big_minus_small_pct"] = {g: 100 * (acc["big_surprise"][g]["next_day_vol_vs_normal"] - acc["small_surprise"][g]["next_day_vol_vs_normal"])
                                             for g in acc["big_surprise"]}
    res["E_information_transfer"] = acc

    (OUT / "test8.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
