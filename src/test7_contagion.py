"""Test 7 - Contagion vs common factors: is there DIRECTED (one-way, lagged) structure among stocks?

Idea. Let C_ij = corr(x_i,t , x_j,t-1): stock j's surprise yesterday vs stock i's today.
Common factors with symmetric dynamics give C_ij = C_ji. Directed propagation (j -> i) gives C_ij > C_ji.
So the antisymmetric part A = (C - C')/2 is the fingerprint of direction. Net leadership of stock j is
L_j = sum_i A_ij (how much more j leads others than it follows them).

Checks
  - Stability: correlate A (all pairs) and L (all stocks) between 1990-2008 and 2008-2026.
    Noise does not repeat; real directed structure does. Null by relabelling stocks in the second half.
  - Structure: large -> small (dollar-volume terciles) with a block-bootstrap CI; sector leadership.
  - Validation on synthetic panels run through the SAME pipeline:
      factor-only: 3 latent volatility factors with time-irreversible (spike-up, slow decay) dynamics,
                   heterogeneous nonlinear exposures and tails, a market factor, own volatility; no propagation
      planted:     same, plus 40 "leader" stocks whose surprises yesterday raise everyone else's volatility today
  x = |z| (magnitude: "firing" contagion) and x = z (direction: classic return lead-lag), where z is each
  stock's market-residual return standardized by its own 2-unit volatility state.

Usage: python3 -I src/test7_contagion.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import load_market, load_stocks, multi_variance, simulate_multi, fit_multi
from test6b_timescales_stocks import _fit, h_path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"
CACHE = OUT / "cache"


# ----------------------------------------------------------------------------- pipeline
def surprises(R, workers=4):
    """Market-residual returns standardized by each stock's own fitted 2-unit volatility path."""
    T, N = R.shape
    mask = np.isfinite(R)
    mkt = np.nanmean(R, axis=1)
    idio = np.empty_like(R)
    for i in range(N):
        ok = mask[:, i]
        b = np.polyfit(mkt[ok], R[ok, i], 1)
        idio[:, i] = R[:, i] - (b[1] + b[0] * mkt)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fits = list(ex.map(_fit, [(idio[:, i], 2) for i in range(N)], chunksize=6))
    h = np.column_stack([h_path(idio[:, i], fits[i]) for i in range(N)])
    z = idio / np.sqrt(h)
    z[~mask] = np.nan
    return z


def standardize(X):
    X = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
    return np.nan_to_num(X)


def lagcorr(X):
    return X[1:].T @ X[:-1] / (len(X) - 1)       # C[i, j] = corr(x_i,t , x_j,t-1)


def antisym(C):
    return (C - C.T) / 2


def upper(M):
    return M[np.triu_indices(M.shape[0], 1)]


def stability(Xa, Xb, rng, n_perm=1000):
    Aa, Ab = antisym(lagcorr(Xa)), antisym(lagcorr(Xb))
    La, Lb = Aa.sum(0), Ab.sum(0)
    ua = upper(Aa)
    obs_pairs, obs_lead = np.corrcoef(ua, upper(Ab))[0, 1], np.corrcoef(La, Lb)[0, 1]
    null_p, null_l = [], []
    N = len(La)
    for _ in range(n_perm):
        p = rng.permutation(N)
        null_p.append(np.corrcoef(ua, upper(Ab[np.ix_(p, p)]))[0, 1])
        null_l.append(np.corrcoef(La, Lb[p])[0, 1])
    null_p, null_l = np.array(null_p), np.array(null_l)
    return {"pairs_corr": float(obs_pairs), "pairs_null_sd": float(null_p.std()),
            "pairs_p": float((1 + np.sum(null_p >= obs_pairs)) / (n_perm + 1)),
            "leadership_corr": float(obs_lead), "leadership_null_sd": float(null_l.std()),
            "leadership_p": float((1 + np.sum(null_l >= obs_lead)) / (n_perm + 1))}, La, Lb


def group_flow(C, lead, follow):
    """mean C[follow, lead] - mean C[lead, follow]: positive = `lead` group leads `follow` group."""
    return float(C[np.ix_(follow, lead)].mean() - C[np.ix_(lead, follow)].mean())


def block_boot(X, fns, L=63, B=400, seed=0):
    rng = np.random.default_rng(seed)
    T = len(X)
    starts_all = np.arange(0, T - L)
    nb = T // L
    out = []
    for _ in range(B):
        st = rng.choice(starts_all, nb)
        S = np.zeros((X.shape[1], X.shape[1]))
        n = 0
        for s in st:
            blk = X[s:s + L]
            S += blk[1:].T @ blk[:-1]
            n += L - 1
        C = S / n
        out.append([f(C) for f in fns])
    return np.array(out)


def analyze(z, split, groups, rng, label):
    """groups: dict name -> (lead_idx, follow_idx)."""
    res = {}
    for kind, raw in [("magnitude", np.abs(z)), ("direction", z)]:
        X = standardize(raw)
        C = lagcorr(X)
        A = antisym(C)
        off = ~np.eye(len(C), dtype=bool)
        r = {"mean_cross_link": float(C[off].mean()), "rms_symmetric": float(np.sqrt(((C + C.T)[off] / 2) ** 2).mean() ** 0.5) if False else
             float(np.sqrt(np.mean(((C + C.T) / 2)[off] ** 2))), "rms_antisymmetric": float(np.sqrt(np.mean(A[off] ** 2))),
             "noise_rms_one_link": float(1 / np.sqrt(len(X)))}
        stab, La, Lb = stability(standardize(raw[:split]), standardize(raw[split:]), rng)
        r["stability"] = stab
        fns, names = [], []
        for gname, (lead, follow) in groups.items():
            r[f"flow_{gname}"] = group_flow(C, lead, follow)
            fns.append(lambda C, lead=lead, follow=follow: group_flow(C, lead, follow))
            names.append(gname)
        if fns:
            bs = block_boot(X, fns)
            for j, g in enumerate(names):
                r[f"flow_{g}_ci95"] = [float(np.quantile(bs[:, j], 0.025)), float(np.quantile(bs[:, j], 0.975))]
                r[f"flow_{g}_halves"] = [group_flow(lagcorr(standardize(raw[:split])), *groups[g]),
                                         group_flow(lagcorr(standardize(raw[split:])), *groups[g])]
        r["_L"] = (La, Lb, A.sum(0))
        res[kind] = r
        print(label, kind, json.dumps({k: v for k, v in r.items() if not k.startswith("_")}), flush=True)
    return res


# ----------------------------------------------------------------------------- synthetic panels
def synthetic(T, N, planted, seed, mpar):
    rng = np.random.default_rng(seed)
    G = 3
    lat = simulate_multi(mpar, T, G, rng)                       # 3 latent processes, time-irreversible volatility
    V = np.column_stack([multi_variance(lat[:, g], mpar["omega"], np.array(mpar["a"]), np.array(mpar["b"]), mpar["delta"])
                         for g in range(G)])
    V = V / V.mean(0)
    grp = rng.integers(0, G, N)
    lam = rng.uniform(0.0, 0.5, (N, G))
    lam[np.arange(N), grp] = rng.uniform(0.5, 1.5, N)          # strong, heterogeneous, nonlinear exposure to own group
    common = np.exp(np.log(V) @ lam.T)                          # T x N variance multipliers
    own = np.exp(np.cumsum(rng.standard_normal((T, N)) * 0.06, 0))  # own slow log-vol drift, then mean-revert
    own = np.exp(pd.DataFrame(np.log(own)).ewm(span=400).mean().rsub(np.log(own)).values * 0.5)
    nu = rng.uniform(4, 10, N)
    eps = rng.standard_t(nu, (T, N)) * np.sqrt((nu - 2) / nu)
    scale = np.sqrt(common * own)
    leaders = rng.choice(N, 40, replace=False)
    if planted:
        P = np.abs(eps[:, leaders]).mean(1)
        P = (P - P.mean()) / P.std()
        boost = np.ones((T, N))
        followers = np.setdiff1d(np.arange(N), leaders)
        boost[1:, followers] = np.clip(1 + 0.04 * P[:-1], 0.5, 2.0)[:, None]  # +4% vol per SD of leaders' surprise
        scale = scale * boost
    m = simulate_multi(mpar, T, 1, rng)[:, 0] * 0.6
    beta = rng.uniform(0.6, 1.4, N)
    R = m[:, None] * beta + scale * eps
    return R, leaders


def main():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    rng = np.random.default_rng(17)
    res = {}

    # ---------------- real panel
    st, _ = load_stocks(ROOT / "data")
    panel = st.loc[:, st.notna().mean() >= 0.98]
    panel = panel.loc[panel.notna().mean(axis=1) >= 0.95]
    R = panel.values * 100
    T, N = R.shape
    cz = CACHE / "z_2unit.npz"
    if cz.exists():
        z = np.load(cz)["z"]
    else:
        z = surprises(R)
        np.savez(cz, z=z, symbols=np.array(panel.columns), dates=np.array(panel.index.astype(str)))
    dates = panel.index
    split = int(np.searchsorted(dates, pd.Timestamp("2008-06-01")))

    # size proxy: median dollar volume 2015-2026
    px = pd.read_csv(ROOT / "data" / "yahoo" / "sp500_adjclose.csv", index_col=0, parse_dates=True)
    vo = pd.read_csv(ROOT / "data" / "yahoo" / "sp500_volume.csv", index_col=0, parse_dates=True)
    dv = (px * vo).loc["2015":, panel.columns].median()
    size_rank = dv.rank().values
    big = np.where(size_rank > 2 * N / 3)[0]
    small = np.where(size_rank <= N / 3)[0]
    meta = pd.read_csv(ROOT / "data" / "meta" / "constituents.csv").set_index("Symbol")
    sectors = np.array([meta.loc[s, "GICS Sector"] for s in panel.columns])
    groups = {"big_to_small": (big, small)}
    real = analyze(z, split, groups, rng, "REAL")

    # sector leadership: net flow of each sector to all others, by half
    sec_rows = {}
    for kind in ("magnitude", "direction"):
        raw = np.abs(z) if kind == "magnitude" else z
        Cs = [lagcorr(standardize(raw[:split])), lagcorr(standardize(raw[split:])), lagcorr(standardize(raw))]
        rows = {}
        for s in np.unique(sectors):
            ins, outs = np.where(sectors == s)[0], np.where(sectors != s)[0]
            rows[s] = [group_flow(C, ins, outs) for C in Cs]
        vals = np.array(list(rows.values()))
        sec_rows[kind] = {"by_sector_[half1,half2,full]": {k: [float(x) for x in v] for k, v in rows.items()},
                          "halves_corr": float(np.corrcoef(vals[:, 0], vals[:, 1])[0, 1])}
    res["real"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in real.items()}
    res["real_sectors"] = sec_rows
    # same-sector vs cross-sector pairs: stability of directed links
    same = sectors[:, None] == sectors[None, :]
    iu = np.triu_indices(N, 1)
    for kind in ("magnitude", "direction"):
        raw = np.abs(z) if kind == "magnitude" else z
        Aa, Ab = antisym(lagcorr(standardize(raw[:split]))), antisym(lagcorr(standardize(raw[split:])))
        s_mask = same[iu]
        res["real"][kind]["pairs_corr_same_sector"] = float(np.corrcoef(Aa[iu][s_mask], Ab[iu][s_mask])[0, 1])
        res["real"][kind]["pairs_corr_cross_sector"] = float(np.corrcoef(Aa[iu][~s_mask], Ab[iu][~s_mask])[0, 1])
        res["real"][kind]["n_same_sector_pairs"] = int(s_mask.sum())
    # leaders / followers list (magnitude), ranked by full-sample leadership, with both halves
    La, Lb, Lf = real["magnitude"]["_L"]
    order = np.argsort(-Lf)
    tab = pd.DataFrame({"symbol": panel.columns, "sector": sectors, "dollar_vol_rank_pct": size_rank / N,
                        "lead_1990_2008": La, "lead_2008_2026": Lb, "lead_full": Lf}).iloc[order]
    tab.to_csv(OUT / "test7_leadership_magnitude.csv", index=False)
    res["real_top_leaders_magnitude"] = tab.head(10)[["symbol", "sector", "lead_1990_2008", "lead_2008_2026"]].round(3).values.tolist()
    res["real_top_followers_magnitude"] = tab.tail(10)[["symbol", "sector", "lead_1990_2008", "lead_2008_2026"]].round(3).values.tolist()
    res["corr_leadership_with_size"] = {k: float(np.corrcoef(real[k]["_L"][2], np.log(dv.values))[0, 1]) for k in real}
    np.savez(OUT / "test7_real.npz", La_mag=La, Lb_mag=Lb, La_dir=real["direction"]["_L"][0], Lb_dir=real["direction"]["_L"][1],
             size=np.log(dv.values), sectors=sectors, symbols=np.array(panel.columns))

    # ---------------- synthetic validation (same pipeline)
    mkt = load_market(ROOT / "data")
    mf = fit_multi(((mkt - mkt.mean()) * 100).values, 2)
    mpar = {k: mf[k] for k in ("omega", "a", "b", "delta", "nu")}
    res["synthetic"] = {}
    for name, planted in [("factor_only", False), ("planted_contagion", True)]:
        Rs, leaders = synthetic(T, N, planted, 99, mpar)
        zs = surprises(Rs)
        followers = np.setdiff1d(np.arange(N), leaders)
        out = analyze(zs, split, {"leaders_to_others": (leaders, followers)}, rng, f"SYN-{name}")
        res["synthetic"][name] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in out.items()}
        Lf_s = out["magnitude"]["_L"][2]
        res["synthetic"][name]["leaders_mean_leadership_pctile"] = float(pd.Series(Lf_s).rank(pct=True).values[leaders].mean())

    (OUT / "test7.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
