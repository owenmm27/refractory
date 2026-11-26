"""Test 14b - Is the global era pattern of the refractory period structural, or just stress?

Test 14 found the refractory period strongest in 2000-2012 in almost every market, weaker after 2013 in most,
and mostly absent before 2000. 2000-2012 was also the high-stress era (two global crashes), and Test 11 found the
dip deepens under stress. For each market (full history), the refractory strength is allowed to depend on a
volatility gauge (log trailing 21-day mean squared return) and on era dummies (pre-2000, post-2012; 2000-2012 is
the base):
    log h_t = log level_t - (rho0 + rho_g*g + rho_pre*pre + rho_post*post) * (s_{t-1} - mean s) - gam_g*g
If the era terms vanish once the gauge is in, the era pattern was stress composition; if they survive, it is
structural. Same estimator as Test 11 (nested fits, so likelihood ratios are >= 0).

Usage: python3 -I src/test14b_era_vs_stress.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import test12c_model_v2 as C
import test14_intl as T14
from test11_gamma import Model, base_start

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()


def job(code):
    r, _ = T14.load(code)
    x = r.values - r.values.mean()
    d = r.index
    g = C.regime(x)
    g = g - g.mean()
    cols, names = [g], ["g"]
    pre = (d < "2000-01-01").astype(float)
    if pre.sum() >= 2000:
        cols.append(pre - pre.mean())
        names.append("pre")
    post = (d >= "2013-01-01").astype(float)
    cols.append(post - post.mean())
    names.append("post")
    XI = np.column_stack(cols)
    XL = g[:, None]
    M = Model(x, XI, XL, names, ["g"])
    eras = [f"rho_{n}" for n in names if n != "g"]
    p0 = base_start(x, len(M.names))
    A = M.fit(["rho0", "gam_g"], p0)
    B = M.fit(["rho0", "gam_g"] + eras, A["p"])
    Cf = M.fit(["rho0", "gam_g", "rho_g"], A["p"])
    D = M.fit(["rho0", "gam_g", "rho_g"] + eras, Cf["p"])
    out = {"code": code, "n": len(x), "start": str(d[0].date()), "era_terms": eras,
           "LR_era_alone": 2 * (B["ll"] - A["ll"]), "LR_era_given_stress": 2 * (D["ll"] - Cf["ll"]),
           "LR_stress_alone": 2 * (Cf["ll"] - A["ll"]), "LR_stress_given_era": 2 * (D["ll"] - B["ll"]),
           "era_only": {k: (B["coef"][k], B["se"][k]) for k in B["coef"]},
           "full": {k: (D["coef"][k], D["se"][k]) for k in D["coef"]}}
    print(code, {k: round(v, 1) for k, v in out.items() if k.startswith("LR")},
          {k: round(v[0], 3) for k, v in out["full"].items() if k.startswith("rho")}, flush=True)
    return out


def main():
    codes = list(T14.NAMES)
    with ProcessPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(job, codes))
    df = pd.DataFrame([{k: v for k, v in r.items() if k.startswith("LR") or k in ("code", "n", "start")} for r in rows])
    for r, k in [(r, k) for r in rows for k in ("rho_g", "rho_pre", "rho_post", "rho0")]:
        if k in r["full"]:
            df.loc[df["code"] == r["code"], f"{k}_full"] = r["full"][k][0]
            df.loc[df["code"] == r["code"], f"{k}_se"] = r["full"][k][1]
        if k in r["era_only"]:
            df.loc[df["code"] == r["code"], f"{k}_era_only"] = r["era_only"][k][0]
    df.to_csv(ROOT / "results" / "test14b_era_vs_stress.csv", index=False)
    (ROOT / "results" / "test14b.json").write_text(json.dumps(rows, indent=2, default=float))
    print(df.round(3).to_string())
    print("pooled LR sums:", {k: round(df[k].sum(), 1) for k in df.columns if k.startswith("LR")})
    print("post-2012 coefficient (era only) negative in", int((df["rho_post_era_only"] < 0).sum()), "of", len(df),
          "| given stress negative in", int((df["rho_post_full"] < 0).sum()))
    print("stress coefficient rho_g positive in", int((df["rho_g_full"] > 0).sum()), "of", len(df))


if __name__ == "__main__":
    main()
