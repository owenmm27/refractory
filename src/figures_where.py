"""Chart for Test 10 (where the refractory period lives). Usage: python3 -I src/figures_where.py <project_root>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figures as F

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
RES = ROOT / "results"
dil = pd.read_csv(RES / "test10_dilution.csv").drop_duplicates(["name", "era"])
mf = json.loads((RES / "test10b.json").read_text())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
fig.subplots_adjust(left=0.065, right=0.9, top=0.77, bottom=0.12, wspace=0.42)

# ---- A. dilution scatter, 2000-2012
d = dil[dil["era"] == "2000-2012"].copy()
d["pct"] = 100 * d["next_day_after_3sd"]
ax1.axhline(100, color=F.BASE, lw=1)
st = d[d["level"] == "STOCK"]
sec = d[d["level"].isin(["GICS", "FF49"])]
idx = d[d["level"] == "INDEX"].iloc[0]
ax1.scatter(st["R2"], st["pct"], s=16, color=F.BLUE, alpha=0.6, edgecolor=F.SURF, linewidth=0.6,
            label="Individual stocks (237)", zorder=3)
ax1.scatter(sec["R2"], sec["pct"], s=26, color=F.ORANGE, alpha=0.85, edgecolor=F.SURF, linewidth=0.8,
            label="Sector and industry portfolios (60)", zorder=4)
ax1.scatter([idx["R2"]], [idx["pct"]], s=90, color=F.AQUA, edgecolor=F.SURF, linewidth=1.5, label="S&P 500 index", zorder=5)
ax1.annotate("S&P 500 index", (idx["R2"], idx["pct"]), xytext=(-8, -2), textcoords="offset points", ha="right",
             va="center", fontsize=9.5, color=F.INK2)
nd = d[d["level"] != "INDEX"]
bins = [0, .2, .3, .4, .5, .7, .9]
g = nd.groupby(pd.cut(nd["R2"], bins), observed=True).agg(x=("R2", "mean"), y=("pct", "median"))
gx, gy = np.r_[g["x"].values, 1.0], np.r_[g["y"].values, idx["pct"]]
ax1.plot(gx, gy, color=F.INK2, lw=1.6, zorder=6)
ax1.annotate("median by R² bin", (0.9, 0.5 * (gy[-2] + gy[-1])), xytext=(0.8, 92), textcoords="data", ha="center",
             fontsize=9, color=F.INK2, arrowprops=dict(arrowstyle="-", color=F.INK2, lw=0.8))
ax1.set_xlim(0, 1.04)
ax1.set_ylim(28, 195)
ax1.set_xlabel("Share of daily variance explained by the S&P 500 (R²)")
ax1.set_ylabel("Next-day volatility after a 3-SD surprise\n(% of what the volatility state implies)")
ax1.text(1.03, 102.5, "100% = no refractory effect", color=F.MUTED, fontsize=9, ha="right")
ax1.set_title("2000–2012: the more a series is the market, the stronger its refractory period", loc="left",
              fontsize=10, color=F.INK)
ax1.legend(loc="upper right", fontsize=8.5, handletextpad=0.3, borderaxespad=0.3)

# ---- B. day after a market surprise, 2000-2026
labs = ["0–1", "1–2", "2–3", "3+"]
keys = ["0-1", "1-2", "2-3", "3+"]
xs = np.arange(4)
ax2.axhline(1, color=F.BASE, lw=1)
for src, col, name in [(mf["A_by_level"]["index_sp500"], F.BLUE, "S&P 500 index"),
                       (mf["C_idio_today_by_market_yesterday"], F.ORANGE, "Stocks' own moves\n(market removed)")]:
    y = np.array([src[k]["z2_today"] for k in keys])
    lo = np.array([src[k]["ci95"][0] for k in keys])
    hi = np.array([src[k]["ci95"][1] for k in keys])
    ax2.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
    ax2.plot(xs, y, color=col, lw=2, marker="o", ms=6, mec=F.SURF, mew=1.2, label=name.replace("\n", " "))
    ax2.annotate(name, (xs[-1], y[-1]), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9.5,
                 color=F.INK2, annotation_clip=False)
ax2.set_xticks(xs, labs)
ax2.set_xlim(-0.2, 3.2)
ax2.set_ylim(0.4, 1.6)
ax2.set_xlabel("Size of yesterday's market surprise (SDs)")
ax2.set_ylabel("Today's variance ÷ model's forecast")
ax2.text(1.25, 0.985, "1.0 = as the model expects", color=F.MUTED, fontsize=9, va="top")
ax2.set_title("2000–2026: after a big market day the index quiets\nbut stocks' own moves get louder", loc="left",
              fontsize=10, color=F.INK)
ax2.legend(loc="lower left", fontsize=8.5, handletextpad=0.4, borderaxespad=0.3)

F.title(fig, "The refractory period belongs to the market as a whole, not to individual stocks",
        "Left: one dot per series, own-surprise refractory fit. Right: pooled across 237 stocks; shaded bands are "
        "95% block-bootstrap intervals.")
fig.savefig(RES / "figures" / "fig10_where_refractory.png", dpi=200)
print("saved fig10", "stock pct range", float(st["pct"].min()), float(st["pct"].max()))
