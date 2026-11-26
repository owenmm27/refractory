"""Test 3 - Neuronal-style avalanches in the stock cross-section, against null models.

Each stock is a "neuron". It "fires" on a day when its return exceeds K robust standard deviations
of its own trailing year (slow homeostatic threshold). Population activity A_t = number firing.
Avalanche (Beggs & Plenz 2003 definition): a run of consecutive days with >= 1 firing, bounded by
silent days. Size = total firings in the run; duration = run length in days.

Nulls (same thresholding pipeline applied to each):
  shuffled   - real days in random order (keeps same-day co-movement, kills memory)
  factor_iid - one-factor model, real betas/variances, i.i.d. Student-t shocks (no volatility memory)
  factor_gjr - one-factor model where market and idiosyncratic shocks each follow a fitted GJR-GARCH
               (a common slowly-varying volatility driver; Touboul & Destexhe 2017's warning case)
  realvol    - the strictest null: the REAL market factor path, plus each stock's REAL fitted idiosyncratic
               volatility path, with fresh i.i.d. shocks. Keeps every volatility state that actually
               happened (including the co-movement of idiosyncratic volatilities) but removes any
               day-by-day coordination of idiosyncratic shocks beyond the market factor.

Usage: python3 -I src/test3_avalanches.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import fit_gjr, gjr_variance, load_stocks, simulate_gjr

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"
K = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0  # firing threshold in robust SDs
SCALE_WIN = 252  # trailing window for each stock's robust scale
N_REPS = 20


# ----------------------------------------------------------------------------- pipeline
def firing_counts(R):
    """R: (T, N) array of returns (NaN allowed). Returns daily firing counts (float, NaN days = no data)."""
    df = pd.DataFrame(R)
    scale = df.abs().rolling(SCALE_WIN, min_periods=200).median().shift(1) * 1.4826
    fire = (df.abs() > K * scale) & scale.notna()
    valid = scale.notna() & df.notna()
    counts = fire.sum(axis=1).values.astype(float)
    nvalid = valid.sum(axis=1).values
    counts[nvalid < 0.9 * R.shape[1]] = np.nan
    return counts


def avalanches(counts):
    c = counts[np.isfinite(counts)]
    active = c > 0
    sizes, durs = [], []
    run_s, run_d = 0.0, 0
    for a, x in zip(active, c):
        if a:
            run_s += x
            run_d += 1
        elif run_d:
            sizes.append(run_s)
            durs.append(run_d)
            run_s, run_d = 0.0, 0
    return np.array(sizes), np.array(durs)


def mr_branching(counts, kmax=40):
    """Multistep-regression estimator (Wilting & Priesemann 2018): r_k = b * m^k.
    Returns (m, b, r1). m is undefined (NaN) when there is no memory to fit (b < 0.05 or r_k not positive)."""
    a = counts[np.isfinite(counts)]
    ks = np.arange(1, kmax + 1)
    rk = np.array([np.polyfit(a[:-k], a[k:], 1)[0] for k in ks])
    if (rk > 0).sum() < kmax * 0.8:
        return np.nan, 0.0, float(rk[0])
    good = rk > 0
    slope, icpt = np.polyfit(ks[good], np.log(rk[good]), 1)
    b = float(np.exp(icpt))
    return (float(np.exp(slope)) if b >= 0.05 else np.nan), b, float(rk[0])


def powerlaw_mle(x, discrete, xmins):
    """Clauset-Shalizi-Newman style: pick xmin minimizing KS, return (alpha, xmin, n_tail, ks)."""
    x = np.asarray(x, float)
    best = None
    for xm in xmins:
        t = x[x >= xm]
        if len(t) < 50:
            continue
        if discrete:
            alpha = 1 + len(t) / np.sum(np.log(t / (xm - 0.5)))
        else:
            alpha = 1 + len(t) / np.sum(np.log(t / xm))
        ts = np.sort(t)
        emp = np.arange(1, len(ts) + 1) / len(ts)
        model = 1 - (ts / xm) ** (1 - alpha)
        ks = np.max(np.abs(emp - model))
        if best is None or ks < best[3]:
            best = (float(alpha), float(xm), int(len(t)), float(ks))
    return best


def summarize(counts):
    s, d = avalanches(counts)
    c = counts[np.isfinite(counts)]
    m, b_mr, r1 = mr_branching(counts)
    # size ~ duration^gamma, using durations with >= 5 avalanches
    du = np.unique(d)
    mean_s = np.array([s[d == k].mean() for k in du])
    nk = np.array([(d == k).sum() for k in du])
    keep = nk >= 5
    gamma = float(np.polyfit(np.log(du[keep]), np.log(mean_s[keep]), 1)[0]) if keep.sum() >= 3 else np.nan
    tau = powerlaw_mle(s, False, np.unique(np.quantile(s, np.linspace(0, 0.9, 30))))
    alpha = powerlaw_mle(d, True, np.arange(1, 15))
    pred_gamma = (alpha[0] - 1) / (tau[0] - 1) if tau and alpha else np.nan
    return {
        "days": int(len(c)), "silent_share": float(np.mean(c == 0)), "mean_firing": float(c.mean()),
        "fano_factor": float(c.var() / c.mean()), "branching_m": m, "mr_b": b_mr, "lag1_slope": r1,
        "n_avalanches": int(len(s)), "dur_mean": float(d.mean()), "dur_p99": float(np.quantile(d, 0.99)), "dur_max": int(d.max()),
        "size_mean": float(s.mean()), "size_p99": float(np.quantile(s, 0.99)), "size_max": float(s.max()),
        "tau_size": tau[0] if tau else np.nan, "tau_xmin": tau[1] if tau else np.nan, "tau_ks": tau[3] if tau else np.nan,
        "alpha_dur": alpha[0] if alpha else np.nan, "alpha_xmin": alpha[1] if alpha else np.nan, "alpha_ks": alpha[3] if alpha else np.nan,
        "gamma_fit": gamma, "gamma_pred_crackling": float(pred_gamma),
    }, s, d


# ----------------------------------------------------------------------------- null-model fitting
def _fit_one(x):
    x = x[np.isfinite(x)]
    f = fit_gjr(x - x.mean())
    return {k: float(f[k]) for k in ("omega", "alpha", "gamma", "beta", "nu")}


def simulate_factor(T, betas, mkt_par, idio_par, mkt_var, idio_var, mode, rng):
    N = len(betas)
    if mode == "gjr":
        m, _ = simulate_gjr(mkt_par, T, 1, rng)
        e, _ = simulate_gjr(idio_par, T, N, rng)
        m = m[:, 0]
    else:  # iid Student-t with matched variance
        nu_m = mkt_par["nu"]
        m = rng.standard_t(nu_m, T) * np.sqrt((nu_m - 2) / nu_m) * np.sqrt(mkt_var)
        nu_i = idio_par["nu"]
        e = rng.standard_t(np.broadcast_to(nu_i, (N,)), size=(T, N)) * np.sqrt((nu_i - 2) / nu_i) * np.sqrt(idio_var)
    return m[:, None] * betas[None, :] + e


def _null_rep(args):
    mode, seed, T, betas, mkt_par, idio_par, mkt_var, idio_var = args[:8]
    rng = np.random.default_rng(seed)
    if mode == "realvol":
        mkt_real, alpha_i, h_idio, nu_i, mask = args[8:]
        eps = rng.standard_t(np.broadcast_to(nu_i, (len(nu_i),)), size=h_idio.shape) * np.sqrt((nu_i - 2) / nu_i)
        R = mkt_real[:, None] * betas[None, :] + alpha_i[None, :] + np.sqrt(h_idio) * eps
        R[~mask] = np.nan
    else:
        R = simulate_factor(T, betas, mkt_par, idio_par, mkt_var, idio_var, mode, rng)
    summ, s, d = summarize(firing_counts(R))
    return summ, s, d


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    st, _ = load_stocks(ROOT / "data")
    panel = st.loc[:, st.notna().mean() >= 0.98]
    panel = panel.loc[panel.notna().mean(axis=1) >= 0.95]
    R = panel.values * 100  # percent log returns
    T, N = R.shape
    print(f"panel: {T} days x {N} stocks, {panel.index[0].date()} to {panel.index[-1].date()}", flush=True)

    res = {"config": {"K_robust_sd": K, "scale_window": SCALE_WIN, "n_stocks": N, "n_days": T,
                      "sample": [str(panel.index[0].date()), str(panel.index[-1].date())], "null_reps": N_REPS}}
    counts = firing_counts(R)
    real, s_real, d_real = summarize(counts)
    res["real"] = real
    if K == 4.0:
        pd.Series(counts, index=panel.index, name="firing").to_csv(OUT / "test3_firing_counts.csv")
    print("real:", json.dumps(real), flush=True)

    # --- fit the factor model
    Rf = np.where(np.isfinite(R), R, np.nan)
    mkt = np.nanmean(Rf, axis=1)
    betas, alphas, idio = np.empty(N), np.empty(N), np.empty_like(Rf)
    for i in range(N):
        ok = np.isfinite(Rf[:, i])
        b = np.polyfit(mkt[ok], Rf[ok, i], 1)
        betas[i], alphas[i] = b[0], b[1]
        idio[:, i] = Rf[:, i] - (b[1] + b[0] * mkt)
    mkt_par = _fit_one(mkt)
    with ProcessPoolExecutor(max_workers=4) as ex:
        idio_fits = list(ex.map(_fit_one, [idio[:, i] for i in range(N)], chunksize=8))
    idio_par = {k: np.array([f[k] for f in idio_fits]) for k in idio_fits[0]}
    mkt_var, idio_var = float(np.nanvar(mkt)), np.nanvar(idio, axis=0)
    # real fitted idiosyncratic variance paths (for the realvol null); missing days filled with 0 shock
    h_idio = np.empty_like(Rf)
    for i in range(N):
        e = np.nan_to_num(idio[:, i] - np.nanmean(idio[:, i]))
        f = idio_fits[i]
        h_idio[:, i] = gjr_variance(e, f["omega"], f["alpha"], f["gamma"], f["beta"], np.nanvar(idio[:, i]))
    mask = np.isfinite(Rf)
    civ = pd.DataFrame(np.log(h_idio)).mean(axis=1)
    res["factor_model"] = {"beta_median": float(np.median(betas)),
                           "market_share_of_variance_median": float(np.median(betas ** 2 * mkt_var / np.nanvar(Rf, axis=0))),
                           "mkt_garch": mkt_par,
                           "idio_persistence_median": float(np.median(idio_par["alpha"] + idio_par["gamma"] / 2 + idio_par["beta"])),
                           # common factor in idiosyncratic volatility: avg pairwise corr of log idio-variance paths
                           "civ_mean_pairwise_corr_logvar": float((lambda c: (c.sum() - N) / (N * (N - 1)))(np.corrcoef(np.log(h_idio[::5]), rowvar=False)))}

    # --- nulls
    rng = np.random.default_rng(7)
    shuffled = []
    s_shuf = d_shuf = None
    for rep in range(N_REPS):
        perm = rng.permutation(T)
        summ, s, d = summarize(firing_counts(R[perm]))
        shuffled.append(summ)
        if rep == 0:
            s_shuf, d_shuf = s, d
    nulls = {"shuffled": shuffled}
    samples = {"shuffled": (s_shuf, d_shuf)}
    for mode in ("iid", "gjr", "realvol"):
        extra = (mkt, alphas, h_idio, idio_par["nu"], mask) if mode == "realvol" else ()
        jobs = [(mode, 100 + r, T, betas, mkt_par, idio_par, mkt_var, idio_var) + extra for r in range(N_REPS)]
        with ProcessPoolExecutor(max_workers=4) as ex:
            out = list(ex.map(_null_rep, jobs))
        nulls[f"factor_{mode}"] = [o[0] for o in out]
        samples[f"factor_{mode}"] = (np.concatenate([o[1] for o in out]), np.concatenate([o[2] for o in out]))
        print(mode, "done", flush=True)

    res["nulls"] = {}
    for name, reps in nulls.items():
        df = pd.DataFrame(reps)
        summ = {}
        for col in df.columns:
            v = df[col].astype(float)
            rv = real[col]
            summ[col] = {"mean": float(v.mean()), "p2.5": float(v.quantile(0.025)), "p97.5": float(v.quantile(0.975)),
                         "real_percentile": float((v < rv).mean() + 0.5 * (v == rv).mean())}
        res["nulls"][name] = summ

    # distribution comparison: two-sample KS (real vs pooled null)
    from scipy.stats import ks_2samp
    res["ks_vs_real"] = {}
    for name, (s, d) in samples.items():
        res["ks_vs_real"][name] = {"size_ks": float(ks_2samp(s_real, s).statistic), "dur_ks": float(ks_2samp(d_real, d).statistic)}

    # save CCDFs for charting
    def ccdf(x):
        xs = np.sort(x)
        return xs, 1 - np.arange(len(xs)) / len(xs)
    np.savez(OUT / ("test3_dists.npz" if K == 4.0 else f"test3_dists_K{K:g}.npz"), s_real=s_real, d_real=d_real,
             s_shuf=samples["shuffled"][0], d_shuf=samples["shuffled"][1],
             s_iid=samples["factor_iid"][0], d_iid=samples["factor_iid"][1],
             s_gjr=samples["factor_gjr"][0], d_gjr=samples["factor_gjr"][1],
             s_rv=samples["factor_realvol"][0], d_rv=samples["factor_realvol"][1])
    # the largest real avalanches, with dates
    c = pd.Series(counts, index=panel.index).dropna()
    run_id = (c == 0).cumsum()
    act = c[c > 0]
    g = act.groupby(run_id[c > 0])
    top = pd.DataFrame({"start": g.apply(lambda x: x.index[0]), "end": g.apply(lambda x: x.index[-1]),
                        "days": g.size(), "size": g.sum()}).nlargest(8, "size")
    res["largest_avalanches"] = [{"start": str(r.start.date()), "end": str(r.end.date()), "days": int(r.days),
                                  "size": float(r.size)} for r in top.itertuples()]
    suffix = "" if K == 4.0 else f"_K{K:g}"
    (OUT / f"test3{suffix}.json").write_text(json.dumps(res, indent=2))

    # compact console report
    cols = ["silent_share", "mean_firing", "fano_factor", "lag1_slope", "branching_m", "mr_b", "n_avalanches", "dur_mean", "dur_p99", "dur_max",
            "size_p99", "size_max", "tau_size", "alpha_dur", "gamma_fit", "gamma_pred_crackling"]
    print(f"{'stat':22s} {'real':>10s} " + " ".join(f"{n:>28s}" for n in res["nulls"]))
    for c in cols:
        row = f"{c:22s} {real[c]:10.3f} "
        for n in res["nulls"]:
            z = res["nulls"][n][c]
            row += f"   {z['mean']:8.3f} [{z['p2.5']:7.3f},{z['p97.5']:7.3f}]"
        print(row)
    print("KS vs real:", json.dumps(res["ks_vs_real"]))
    print("factor model:", json.dumps(res["factor_model"]))
    print("largest avalanches:", json.dumps(res["largest_avalanches"]))


if __name__ == "__main__":
    main()
