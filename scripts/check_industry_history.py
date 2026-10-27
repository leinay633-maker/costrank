"""Gate 3 of RESEARCH section 9: are the provider's industry snapshots point in time? Offline.

usage: python scripts/check_industry_history.py <acquire_dir> --output report.json

A provider that returned today's classification for every historical date would show no company
changing industry across the monthly snapshots. Passes if some companies change, in at least
three different years; lists the changes so a sample can be checked against announcements.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("acquire_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frames = []
    for path in sorted((Path(args.acquire_dir) / "raw" / "industry").glob("*.csv")):
        f = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["updateDate", "code", "industry"])
        frames.append(f.assign(snapshot=path.stem))
    ind = pd.concat(frames, ignore_index=True)
    ind = ind[ind.code.str.fullmatch(r"(sh\.60|sz\.00)\d{4}|sz\.30\d{4}|sh\.688\d{3}") & ind.industry.ne("")]
    ind = ind.sort_values(["code", "snapshot"])
    ind["previous"] = ind.groupby("code").industry.shift()
    changes = ind[ind.previous.notna() & ind.previous.ne(ind.industry)]
    years = sorted(changes.snapshot.str[:4].unique())
    report = {"snapshots": int(ind.snapshot.nunique()), "codes": int(ind.code.nunique()),
              "changes": int(len(changes)), "changed_codes": int(changes.code.nunique()), "change_years": years,
              "update_dates": sorted(ind.updateDate.unique())[:3] + ["..."] + sorted(ind.updateDate.unique())[-3:],
              "passed": bool(changes.code.nunique() > 0 and len(years) >= 3),
              "examples": changes.head(40)[["code", "snapshot", "previous", "industry"]].to_dict("records")}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
