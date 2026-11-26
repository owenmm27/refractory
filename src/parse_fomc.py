"""Parse FOMC meeting dates from saved Federal Reserve calendar pages (data/macro/raw).

Statement day = last day of the meeting. Scheduled meetings vs unscheduled meetings/conference calls are
kept apart. Writes data/macro/fomc_dates.csv.

Usage: python3 -I src/parse_fomc.py <project_root>
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]).resolve()
RAW = ROOT / "data" / "macro" / "raw"
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], 1)}
ABBR = {m[:3]: i for m, i in MONTHS.items()}


def month_num(s):
    s = s.strip().rstrip(".")
    return MONTHS.get(s) or ABBR.get(s[:3])


rows = []
# historical pages, e.g. "January 26-27 Meeting - 2010", "April/May 30-1 Meeting - 2013", "October 16 (unscheduled) - 2013",
# "March 17-18 (cancelled) Meeting - 2020", "March 19 (notation vote) - 2020", "May 9 Conference Call - 2010"
PAT = re.compile(r"([A-Za-z]+)(?:/([A-Za-z]+))?\s+(\d{1,2})(?:\s*-\s*(?:([A-Za-z]+)\s+)?(\d{1,2}))?\s*(\([^)]*\))?\s*(Meeting|Conference Call)?\s*(\([^)]*\))?\s*-\s*(\d{4})")
for f in sorted(RAW.glob("fomchistorical*.htm")):
    s = f.read_text(errors="replace")
    for m in re.finditer(r"<h5[^>]*>(.*?)</h5>", s, re.S):
        h = re.sub(r"<[^>]+>|\s+", " ", m.group(1)).strip()
        mm = PAT.match(h)
        if not mm:
            continue
        mon1, mon_slash, d1, mon_day2, d2, note1, kind, note2, yr = mm.groups()
        notes = ((note1 or "") + (note2 or "")).lower()
        if "cancel" in notes:
            continue
        mon = month_num(mon_day2 or (mon_slash if (mon_slash and d2) else mon1))
        if mon is None:
            continue
        day = int(d2 or d1)
        if "notation" in notes:
            k = "notation_vote"
        elif kind == "Conference Call" or "unscheduled" in notes:
            k = "unscheduled"
        else:
            k = "scheduled"
        rows.append((pd.Timestamp(int(yr), mon, day), k, h))

# current calendar page: year panels with month / date rows
s = (RAW / "fomccalendars.htm").read_text(errors="replace")
parts = re.split(r"(\d{4}) FOMC Meetings", s)
for k in range(1, len(parts) - 1, 2):
    yr, body = int(parts[k]), parts[k + 1]
    for m in re.finditer(r'fomc-meeting__month[^>]*><strong>(.*?)</strong>.*?fomc-meeting__date[^>]*>(.*?)</div>', body, re.S):
        mon_txt = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        date_txt = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        mons = [x for x in re.split(r"[/-]", mon_txt) if x.strip()]
        days = re.findall(r"\d{1,2}", date_txt)
        if not days or not mons:
            continue
        mon = month_num(mons[-1])
        unscheduled = "unscheduled" in (date_txt + m.group(0)).lower() or "notation" in date_txt.lower()
        rows.append((pd.Timestamp(yr, mon, int(days[-1])), "unscheduled" if unscheduled else "scheduled", f"{mon_txt} {date_txt} {yr}"))

df = pd.DataFrame(rows, columns=["date", "kind", "source_text"]).drop_duplicates(subset=["date", "kind"]).sort_values("date")
df = df[(df["date"] >= "2000-01-01") & (df["date"] <= pd.Timestamp.today())]
sch = df[df["kind"] == "scheduled"].sort_values("date")
drop = sch.index[(sch["date"].shift(-1) - sch["date"]).dt.days.fillna(99) <= 2]
df = df.drop(drop)
df.to_csv(ROOT / "data" / "macro" / "fomc_dates.csv", index=False)
print(df.groupby([df["date"].dt.year, "kind"]).size().unstack(fill_value=0).to_string())
