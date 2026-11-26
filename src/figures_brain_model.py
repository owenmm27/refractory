"""Chart for Test 12 (unified market-brain model scorecard). Usage: python3 -I src/figures_brain_model.py <project_root>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import figures as F

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
RES = ROOT / "results"
d = json.loads((RES / "test12.json").read_text())
d2 = json.loads((RES / "test12c.json").read_text())

ROWS = [  # (key, label, group)
    ("index_absacf_lag1", "Size memory: 1 day", "Index memory"),
    ("index_absacf_lag21", "Size memory: 1 month", "Index memory"),
    ("index_absacf_lag126", "Size memory: 6 months", "Index memory"),
    ("index_absacf_lag252", "Size memory: 1 year", "Index memory"),
    ("index_excess_kurtosis", "Fat tails", "Index memory"),
    ("index_leverage_corr", "Falls raise volatility", "Index memory"),
    ("index_ac1", "Next-day reversal", "Index direction"),
    ("index_VR5", "Weekly reversal", "Index direction"),
    ("index_VR21", "Monthly reversal", "Index direction"),
    ("index_absz_lag1_corr_vs_plain_model", "Overreaction a plain model leaves", "Refractory and desync"),
    ("index_dip_after_big_day", "Index: quiet day after a big day", "Refractory and desync"),
    ("idio_dip_after_big_market_day", "Stocks' own moves after a big market day", "Refractory and desync"),
    ("stock_dip_after_own_big_day", "Stocks after their own big day", "Refractory and desync"),
    ("idio_absacf_lag1_mean", "Stocks' own size memory: 1 day", "Stocks"),
    ("idio_absacf_lag21_mean", "Stocks' own size memory: 1 month", "Stocks"),
    ("idio_absacf_lag126_mean", "Stocks' own size memory: 6 months", "Stocks"),
    ("idio_excess_kurtosis_median", "Stocks' own fat tails", "Stocks"),
    ("sync_mean_pairwise_corr", "Average correlation", "Synchrony"),
    ("sync_vs_same_month_vol_corr", "Synchrony rises with volatility", "Synchrony"),
    ("sync_after_minus_before_vol", "Synchrony follows volatility", "Synchrony"),
    ("firing_silent_share", "Share of quiet days", "Collective firing"),
    ("firing_fano_factor", "Burstiness", "Collective firing"),
    ("firing_lag1_slope", "Day-to-day persistence", "Collective firing"),
    ("firing_size_p99", "Largest avalanches (size)", "Collective firing"),
    ("firing_dur_p99", "Longest avalanches", "Collective firing"),
]
COLS = [("full", "Full\nmodel"), ("no_refractory", "No\nrefractory"), ("no_reversal", "No\nreversal"),
        ("one_timescale_market", "One-speed\nmarket"), ("no_spillover", "No market\nspillover"),
        ("no_sector_layer", "No sector\nlayer"), ("v2_full", "Version 2\n(+2 pieces)")]

src = lambda c: d2 if c == "v2_full" else d
P = np.array([[src(c)["variants"][c][k]["real_pctile"] for c, _ in COLS] for k, _, _ in ROWS])
IN = np.array([[src(c)["variants"][c][k]["p5"] <= d["real"][k] <= src(c)["variants"][c][k]["p95"] for c, _ in COLS] for k, _, _ in ROWS])
cmap = LinearSegmentedColormap.from_list("div", [F.BLUE, "#e8e7e1", F.ORANGE])

# insert a header row before each group
disp, labels, is_head = [], [], []
prev = None
for i, (_, lab, g) in enumerate(ROWS):
    if g != prev:
        disp.append(None)
        labels.append(g.upper())
        is_head.append(True)
        prev = g
    disp.append(i)
    labels.append(lab)
    is_head.append(False)
PD = np.array([[np.nan] * P.shape[1] if r is None else P[r] for r in disp])
fig = plt.figure(figsize=(10, 12))
ax = fig.add_axes([0.36, 0.08, 0.58, 0.79])
ax.imshow(np.ma.masked_invalid(PD), cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_facecolor(F.SURF)
ax.grid(False)
for i in range(PD.shape[0] + 1):
    ax.axhline(i - 0.5, color=F.SURF, lw=2)
for j in range(PD.shape[1] + 1):
    ax.axvline(j - 0.5, color=F.SURF, lw=2)
for i, r in enumerate(disp):
    if r is None:
        continue
    for j in range(P.shape[1]):
        if not IN[r, j]:
            ax.text(j, i, "×", ha="center", va="center", fontsize=12, color=F.INK, fontweight="bold")
ax.set_xticks(range(len(COLS)), [c[1] for c in COLS], fontsize=9)
ax.xaxis.tick_top()
ax.set_yticks(range(len(labels)), labels, fontsize=9.5)
for tl, h in zip(ax.get_yticklabels(), is_head):
    if h:
        tl.set_fontsize(8)
        tl.set_fontweight("bold")
        tl.set_color(F.MUTED)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cax = fig.add_axes([0.36, 0.045, 0.58, 0.012])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1], labels=["real data below\nevery simulation", "real data in the\nmiddle of the model's range",
                                  "real data above\nevery simulation"], fontsize=8.5)
cb.outline.set_visible(False)
full_in = IN[:, 0].mean()
fig.text(0.06, 0.975, "One model reproduces the index; it can't reproduce how stocks burst together",
         fontsize=13.5, fontweight="bold", color=F.INK, va="top")
fig.text(0.06, 0.95, f"Where the real 2000–2026 data fall among {d['n_paths']} simulated histories of each model; "
        "× = outside the model's 5–95% range.\nRemoving a piece breaks the pattern that depends on it. "
        "Version 2 adds volatility-keyed reversal and a shared volatility gain.", fontsize=10, color=F.INK2, va="top")
fig.savefig(RES / "figures" / "fig12_brain_model.png", dpi=200)
print("saved fig12", "full inside", full_in, {c: int(IN[:, j].sum()) for j, (c, _) in enumerate(COLS)})
