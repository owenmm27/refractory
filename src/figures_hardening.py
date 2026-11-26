"""Chart for Test 15 (hardening the refractory period). Usage: python3 -I src/figures_hardening.py <project_root>"""
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
m = pd.read_csv(RES / "test15_score_driven.csv")
m = m[m["name"] != "CRSP market"].rename(columns={"name": "code"})
mf = pd.read_csv(RES / "test15b_range_intl.csv")
J = json.loads((RES / "test15b.json").read_text())
NAMES = mf.drop_duplicates("code").set_index("code")["name"]
DEV = ["GSPC", "FTSE", "DAX", "CAC", "SMI", "AEX", "N225", "HSI", "ASX200", "TSX"]
EM = ["KOSPI", "TAIEX", "SENSEX", "BOVESPA", "IPC"]
bt = m.pivot(index="code", columns="era", values="betat_next_day") * 100
sq = m.pivot(index="code", columns="era", values="sq_next_day") * 100
cal = mf.pivot(index="code", columns="era", values="calmer_net_move") * 100
order = ["GSPC"] + sorted([c for c in DEV if c != "GSPC"], key=lambda c: bt.loc[c, "2000-2012"]) + \
        sorted(EM, key=lambda c: bt.loc[c, "2000-2012"])
ypos = {c: len(order) - i + (0.6 if c in DEV else 0) for i, c in enumerate(order)}
y_stock = -0.6
ERA_COL = {"pre-2000": F.BLUE, "2000-2012": F.ORANGE, "2013-2026": F.AQUA}

fig = plt.figure(figsize=(15.5, 8.2))
ax1 = fig.add_axes([0.125, 0.085, 0.30, 0.64])
ax2 = fig.add_axes([0.44, 0.085, 0.215, 0.64], sharey=ax1)
ax3 = fig.add_axes([0.785, 0.085, 0.2, 0.64])

# ---- A: robust volatility model
ax1.axvline(100, color=F.BASE, lw=1.2)
for c in order:
    y = ypos[c]
    a, b = sq.loc[c, "2000-2012"], bt.loc[c, "2000-2012"]
    ax1.plot([a, b], [y, y], color=F.GRID, lw=2.5, zorder=1)
    ax1.plot(a, y, "o", ms=8, mfc=F.SURF, mec=F.ORANGE, mew=1.5, zorder=3,
             label="2000–2012, squared-return model (Tests 9–14)" if c == "GSPC" else None)
    ax1.plot(b, y, "o", ms=8, color=F.ORANGE, mec=F.SURF, mew=1.3, zorder=4,
             label="2000–2012, robust model (Beta-t-EGARCH)" if c == "GSPC" else None)
    ax1.plot(bt.loc[c, "2013-2026"], y - 0.28, "o", ms=6.5, color=F.AQUA, mec=F.SURF, mew=1.2, zorder=4,
             label="2013–2026, robust model" if c == "GSPC" else None)
ax1.text(31, y_stock, "not fitted here (model fits in Test 10: none)", fontsize=8.5, color=F.MUTED, va="center")
ax1.set_xlim(30, 140)
ax1.set_xlabel("Next-day volatility after a 3-SD surprise (% of the state's forecast)", fontsize=9.5)
ax1.text(101.5, len(order) + 1.25, "100% = no refractory effect", fontsize=8.5, color=F.MUTED)
ax1.set_title("A. With a volatility model that doesn't overreact", loc="left", fontsize=10.5, color=F.INK)

# ---- B: no model
nl = mf[mf["era"] == "2000-2012"]
lo, hi = nl["calmer_null_p05"].median() * 100, nl["calmer_null_p95"].median() * 100
ax2.axvspan(lo, hi, color=F.GRID, alpha=0.55, lw=0, zorder=0)
ax2.axvline(50, color=F.BASE, lw=1.2)
for c in order:
    y = ypos[c]
    vals = [(e, cal.loc[c, e]) for e in ("pre-2000", "2000-2012", "2013-2026") if e in cal.columns and pd.notna(cal.loc[c, e])]
    xs = [v for _, v in vals]
    ax2.plot([min(xs), max(xs)], [y, y], color=F.GRID, lw=2.5, zorder=1)
    for e, v in vals:
        ax2.plot(v, y, "o", ms=8, color=ERA_COL[e], mec=F.SURF, mew=1.3, zorder=3)
st = J["stocks"]
for e, key in (("2000-2012", "own_2005-2012"), ("2013-2026", "own_2013-2026")):
    ax2.plot(st[key]["calmer_net_move"]["est"] * 100, y_stock, "o", ms=8, color=ERA_COL[e], mec=F.SURF, mew=1.3, zorder=3)
ax2.set_xlim(40, 64)
ax2.set_xticks([45, 50, 55, 60])
ax2.set_xlabel("Share of 2-SD+ days followed by a calmer next day (%)", fontsize=9.5)
ax2.text(lo + 0.2, len(order) + 1.25, f"no-effect simulations\n(90% range, one market)", fontsize=8.5, color=F.MUTED, va="center")
ax2.set_title("B. No model: is the next day calmer?", loc="left", fontsize=10.5, color=F.INK)
for e, lab in (("pre-2000", "Before 2000"), ("2000-2012", "2000–2012 (stocks: 2005–2012)"), ("2013-2026", "2013–2026")):
    ax2.plot([], [], "o", ms=8, color=ERA_COL[e], mec=F.SURF, label=lab)

yt = [ypos[c] for c in order] + [y_stock]
ax1.set_yticks(yt, [NAMES[c] for c in order] + ["237 US stocks (own moves)"], fontsize=9.5)
ax1.set_ylim(-1.4, len(order) + 1.9)
first_em = [c for c in order if c in EM][0]
ax1.text(31, ypos[first_em] + 0.8, "EMERGING MARKETS", fontsize=8, color=F.MUTED, fontweight="bold", va="center")
for ax in (ax1, ax2):
    ax.grid(axis="y", visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.tick_params(axis="y", length=0)

# ---- C: what the next day looks like (SPY 2000-2012)
comp = [("P", "Intraday range\n(how far it travels)"), ("oc", "Session move\n(open to close)"), ("gap", "Overnight gap"),
        ("r", "Close-to-close move\n(how far it gets)")]
big, small = J["spy"]["2000-2012"]["component_ratios_2+"], J["spy"]["2000-2012"]["component_ratios_0-1"]
ax3.axvline(100, color=F.BASE, lw=1.2)
for i, (k, lab) in enumerate(comp):
    y = len(comp) - 1 - i
    for d, col, dy, name in ((big, F.ORANGE, 0.12, "After a 2-SD+ day (224 days)"), (small, F.MUTED, -0.18, "After an ordinary day")):
        v = d[k]
        ax3.plot([v["lo"] * 100, v["hi"] * 100], [y + dy, y + dy], color=col, lw=1.6, alpha=0.55, zorder=2)
        ax3.plot(v["est"] * 100, y + dy, "o", ms=8, color=col, mec=F.SURF, mew=1.3, zorder=3, label=name if i == 0 else None)
        if d is big:
            ax3.text(v["est"] * 100, y + dy + 0.2, f"{v['est'] * 100:.0f}%", fontsize=8.5, color=F.INK2, ha="center")
ax3.set_yticks(range(len(comp)), [lab for _, lab in comp][::-1], fontsize=9.5)
ax3.set_xlim(20, 170)
ax3.set_ylim(-0.7, len(comp) - 0.2)
ax3.grid(axis="y", visible=False)
ax3.set_xlabel("Next day's variance as % of the following day's", fontsize=9.5)
ax3.set_title("C. Inside the next day (SPY, 2000–2012)", loc="left", fontsize=10.5, color=F.INK)
ax3.text(101.5, len(comp) - 0.42, "100% = flat", fontsize=8.5, color=F.MUTED)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
h3, l3 = ax3.get_legend_handles_labels()
fig.legend(h1, l1, loc="upper left", fontsize=8.8, ncol=1, bbox_to_anchor=(0.12, 0.865), frameon=False, handletextpad=0.3)
fig.legend(h2, l2, loc="upper left", fontsize=8.8, ncol=1, bbox_to_anchor=(0.435, 0.865), frameon=False, handletextpad=0.3)
fig.legend(h3, l3, loc="upper left", fontsize=8.8, ncol=1, bbox_to_anchor=(0.78, 0.865), frameon=False, handletextpad=0.3)
F.title(fig, "The refractory period survives a robust volatility model and a model-free test, but it is smaller than first estimated",
        "2000–2012: significant in 12 of 15 markets with a model that doesn't overreact to big days, and visible without any model. "
        "After 2012 it is gone except in Hong Kong and Taiwan.")
fig.savefig(RES / "figures" / "fig15_hardening.png", dpi=200)
print("saved fig15")
