"""Test 5 - Is the post-2000 negative feedback related to options hedging (the Black-Scholes channel)?

Three checks:
  (a) Options-expiration cycle. Dealer gamma is concentrated in the week before monthly expiry
      (third Friday). If hedging drives the feedback sign, the autocorrelation in expiry weeks should
      differ from other weeks AFTER listed options existed, but not before (1926-1972 = placebo).
  (b) Overnight vs next-day split (SPY, 1993+). Where does the reversal happen? If the close
      overshoots (pushed by hedging/rebalancing flows near the close), the correction should show up
      mostly overnight, before regular trading resumes.
  (c) Dealer gamma (SqueezeMetrics GEX estimate, 2011+). Long-gamma dealers buy dips / sell rallies
      (predicts reversal); short-gamma dealers do the opposite (predicts continuation).

Usage: python3 -I src/test5_options.py <project_root>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

from common import load_market, load_vix

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
DATA, OUT = ROOT / "data", ROOT / "results"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}


# ----------------------------------------------------------------------------- data
def get_spy():
    out = DATA / "yahoo" / "SPY_ohlc.csv"
    if not out.exists():
        p1, p2 = int(pd.Timestamp("1993-01-01").timestamp()), int(pd.Timestamp.now().timestamp())
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/SPY?period1={p1}&period2={p2}"
               f"&interval=1d&events=div%2Csplits")
        res = requests.get(url, headers=UA, timeout=60).json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({"open": q["open"], "close": q["close"], "adjclose": res["indicators"]["adjclose"][0]["adjclose"]},
                          index=pd.to_datetime(res["timestamp"], unit="s").normalize())
        df = df[~df.index.duplicated(keep="last")]
        df.to_csv(out)
    df = pd.read_csv(out, index_col=0, parse_dates=True).dropna()
    f = df["adjclose"] / df["close"]          # dividend/split adjustment factor, applied to the open as well
    adj_open = df["open"] * f
    cc = np.log(df["adjclose"]).diff()
    on = np.log(adj_open / df["adjclose"].shift(1))
    idr = np.log(df["adjclose"] / adj_open)
    return pd.DataFrame({"cc": cc, "on": on, "id": idr}).dropna()


def get_gex():
    out = DATA / "gex" / "DIX.csv"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get("https://squeezemetrics.com/monitor/static/DIX.csv", headers=UA, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
    df = pd.read_csv(out, parse_dates=["date"]).set_index("date")
    return df


# ----------------------------------------------------------------------------- helpers
def third_fridays(index):
    """Last trading day on or before each month's third Friday (expiry moves to Thursday on holidays)."""
    out = []
    for (y, m), _ in pd.Series(0, index=index).groupby([index.year, index.month]):
        first = pd.Timestamp(y, m, 1)
        tf = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
        avail = index[(index <= tf) & (index >= tf - pd.Timedelta(days=6))]
        if len(avail):
            out.append(avail[-1])
    return pd.DatetimeIndex(out)


def pair_corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum()))


def cluster_boot_diff(df, flag, clusters, B=1000, seed=0):
    """df has columns prev, cur. Bootstrap (resampling clusters) the difference in lag-1 correlation
    between flagged and unflagged pairs. Returns (diff, se, two-sided p)."""
    rng = np.random.default_rng(seed)
    groups = {k: v for k, v in df.assign(flag=flag, cl=clusters).groupby("cl")}
    keys = np.array(list(groups))
    base = df.assign(flag=flag)

    def stat(d):
        a, b = d[d["flag"]], d[~d["flag"]]
        return pair_corr(a["prev"].values, a["cur"].values) - pair_corr(b["prev"].values, b["cur"].values)

    obs = stat(base)
    boots = np.array([stat(pd.concat([groups[k] for k in rng.choice(keys, len(keys))])) for _ in range(B)])
    se = boots.std()
    p = float(np.mean(np.abs(boots - boots.mean()) >= abs(obs)))
    return float(obs), float(se), p


def hac_reg(y, X, names, lags=5):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {n: {"coef": float(c), "t": float(t)} for n, c, t in zip(["const"] + names, m.params, m.tvalues)}


# ----------------------------------------------------------------------------- main
def main():
    res = {}
    mkt = load_market(DATA)
    vix = load_vix(DATA)

    # ---------------- (a) expiry-cycle test with a pre-options placebo
    idx = mkt.index
    tf = third_fridays(idx)
    week_of = pd.Series(False, index=idx)
    after = pd.Series(False, index=idx)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    for d in tf:
        i = pos[d]
        week_of.iloc[max(i - 4, 0):i + 1] = True          # 5 trading days ending at expiry
        after.iloc[i + 1:i + 6] = True                      # 5 trading days after
    sigma = mkt.rolling(63).std().shift(1)
    z = (mkt / sigma)                                       # vol-standardized so crises don't dominate
    pairs = pd.DataFrame({"prev": z.shift(1), "cur": z}).dropna()
    flag_week = week_of.reindex(pairs.index).values
    months = pairs.index.to_period("M").astype(str)
    eras = [("1926-1972 (no listed options: placebo)", "1926", "1972"),
            ("1973-1999 (listed options, index options from 1983)", "1973", "1999"),
            ("2000-2026", "2000", "2026"), ("2000-2012", "2000", "2012"), ("2013-2026 (weeklies/0DTE era)", "2013", "2026")]
    res["expiry_cycle"] = {}
    for name, a, b in eras:
        m = (pairs.index >= a) & (pairs.index <= b + "-12-31")
        d = pairs[m]
        fw = flag_week[m]
        diff, se, p = cluster_boot_diff(d, fw, months[m], B=600)
        absr = mkt.reindex(d.index).abs()
        res["expiry_cycle"][name] = {
            "ac1_expiry_week": pair_corr(d["prev"].values[fw], d["cur"].values[fw]),
            "ac1_other_weeks": pair_corr(d["prev"].values[~fw], d["cur"].values[~fw]),
            "difference": diff, "se_cluster_boot": se, "p": p,
            "n_expiry_pairs": int(fw.sum()), "n_other_pairs": int((~fw).sum()),
            "abs_ret_ratio_expiry_vs_other": float(absr[fw].mean() / absr[~fw].mean()),
            "abs_ret_ratio_week_after_vs_other": float(absr[after.reindex(d.index).values].mean()
                                                       / absr[~(fw | after.reindex(d.index).values)].mean())}

    # ---------------- (b) overnight vs next-day decomposition (SPY)
    spy = get_spy()
    spy["sig"] = spy["cc"].rolling(63).std().shift(1)
    spy["prev_cc"], spy["prev_on"], spy["prev_id"] = spy["cc"].shift(1), spy["on"].shift(1), spy["id"].shift(1)
    spy["vix_prev"] = np.log(vix.reindex(spy.index).shift(1))
    s = spy.dropna().copy()
    for c in ["cc", "on", "id", "prev_cc", "prev_on", "prev_id"]:
        s[c + "_z"] = s[c] / s["sig"]
    terc = pd.qcut(s["vix_prev"], 3, labels=["calm", "normal", "turbulent"])
    res["overnight_split"] = {"sample": [str(s.index[0].date()), str(s.index[-1].date())]}
    for name, a, b in [("1993-1999", "1993", "1999"), ("2000-2006", "2000", "2006"), ("2007-2026", "2007", "2026")]:
        d = s.loc[a:b + "-12-31"]
        row = {}
        for tgt in ["cc", "on", "id"]:
            row[f"slope_{tgt}_on_prev_cc"] = float(np.polyfit(d["prev_cc_z"], d[f"{tgt}_z"], 1)[0])
        # which part of yesterday reverses? yesterday's overnight gap or yesterday's trading-session move
        row["slope_next_cc_on_prev_id"] = float(np.polyfit(d["prev_id_z"], d["cc_z"], 1)[0])
        row["slope_next_cc_on_prev_on"] = float(np.polyfit(d["prev_on_z"], d["cc_z"], 1)[0])
        row["share_var_overnight"] = float(d["on"].var() / (d["on"].var() + d["id"].var()))
        res["overnight_split"][name] = row
    d = s.loc["2007":]
    res["overnight_split"]["2007-2026_by_vix_regime"] = {
        str(k): {f"slope_{t}_on_prev_cc": float(np.polyfit(g["prev_cc_z"], g[f"{t}_z"], 1)[0]) for t in ["cc", "on", "id"]}
        | {"n": int(len(g))}
        for k, g in d.groupby(terc.reindex(d.index), observed=True)}
    # HAC-robust version of the key 2007+ decomposition
    res["overnight_split"]["2007-2026_hac"] = {
        t: hac_reg(d[f"{t}_z"].values, d[["prev_cc_z"]].values, ["prev_cc_z"])["prev_cc_z"] for t in ["cc", "on", "id"]}

    # ---------------- (c) dealer gamma (SqueezeMetrics GEX estimate)
    gx = get_gex()
    g = s.join(gx[["gex"]], how="inner")
    g["gex_prev"] = gx["gex"].shift(1).reindex(g.index)       # gamma known at the previous close
    g = g.dropna(subset=["gex_prev"])
    g["gex_pct"] = g["gex_prev"].rolling(252, min_periods=120).rank(pct=True)
    g["neg"] = g["gex_prev"] < 0
    res["gex"] = {"sample": [str(g.index[0].date()), str(g.index[-1].date())], "share_days_negative": float(g["neg"].mean())}
    rows = {}
    for label, mask in [("negative gamma", g["neg"]), ("positive gamma", ~g["neg"])]:
        h = g[mask]
        rows[label] = {"n": int(len(h)), "ac1_z": pair_corr(h["prev_cc_z"].values, h["cc_z"].values),
                       "slope_on": float(np.polyfit(h["prev_cc_z"], h["on_z"], 1)[0]),
                       "slope_id": float(np.polyfit(h["prev_cc_z"], h["id_z"], 1)[0]),
                       "mean_abs_ret_bp": float(h["cc"].abs().mean() * 1e4),
                       "median_vix": float(np.exp(h["vix_prev"]).median())}
    pos_g = g[~g["neg"]].copy()
    pos_g["tq"] = pd.qcut(pos_g["gex_pct"], 3, labels=["low", "mid", "high"])
    for k, h in pos_g.dropna(subset=["tq"]).groupby("tq", observed=True):
        rows[f"positive gamma, {k} vs past year"] = {"n": int(len(h)), "ac1_z": pair_corr(h["prev_cc_z"].values, h["cc_z"].values),
                                                     "mean_abs_ret_bp": float(h["cc"].abs().mean() * 1e4),
                                                     "median_vix": float(np.exp(h["vix_prev"]).median())}
    res["gex"]["by_regime"] = rows
    # does gamma sign matter once you control for VIX? z_t = a + b z_{t-1} + c z_{t-1}*neg + d z_{t-1}*logVIX
    X = np.column_stack([g["prev_cc_z"], g["prev_cc_z"] * g["neg"], g["prev_cc_z"] * (g["vix_prev"] - g["vix_prev"].mean())])
    res["gex"]["interaction_reg_hac"] = hac_reg(g["cc_z"].values, X, ["prev_z", "prev_z_x_neg_gamma", "prev_z_x_logVIX"])
    X2 = np.column_stack([g["prev_cc_z"], g["prev_cc_z"] * g["neg"]])
    res["gex"]["interaction_reg_no_vix_hac"] = hac_reg(g["cc_z"].values, X2, ["prev_z", "prev_z_x_neg_gamma"])

    (OUT / "test5.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
