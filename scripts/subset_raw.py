"""Hard-link a dated prefix of an acquire() raw cache into a new directory whose plan ends no later,
so `costrank assemble` can seal a snapshot while the acquisition continues: a 2019-and-earlier
debug snapshot (RESEARCH section 9), or the full range without minute bars (supplement 11). Offline. Cached files
are never rewritten, so the links cannot change underneath the subset; deleting the subset
removes only its links.

usage: python scripts/subset_raw.py <acquire_dir> <new_dir> --end 2019-12-31
then:  costrank assemble --data <new_dir> --end 2019-12-31
"""
import argparse
import json
import os
import shutil
from pathlib import Path

DATED = ("daily", "adjust", "industry", "hs300", "zz500")  # one file per session, named by its date
YEARLY = ("dividend", "min30")  # one file per code and calendar year: <code>-<year>
WHOLE = ("master", "universe", "index", "forecast", "express")  # plan-range queries; assembly filters by date


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def subset(source, target, end):
    source, target = Path(source), Path(target)
    plan = json.loads((source / "collection-plan.json").read_text(encoding="utf-8"))
    if plan.get("schema") != 3 or not plan["start"] <= end <= plan["end"]:
        raise ValueError("Need a schema-3 acquisition plan that reaches the subset end")
    if target.exists():
        raise FileExistsError("Choose a new directory")
    raw, out = source / "raw", target / "raw"
    for path in [*sorted(raw.glob("calendar*.csv")), raw / "master.csv"]:
        link(path, out / path.name)
    counts = {}
    for folder in DATED + YEARLY + WHOLE:
        n = 0
        for path in sorted((raw / folder).glob("*.csv")):
            if (folder in DATED and path.stem > end) or (folder in YEARLY and path.stem[-4:] > end[:4]):
                continue
            link(path, out / folder / path.name)
            n += 1
        counts[folder] = n
    (target / "collection-plan.json").write_text(json.dumps(dict(plan, end=end), indent=2), encoding="utf-8")
    shutil.copy2(source / "request-ledger.json", target / "request-ledger.json")
    (target / "subset-of.json").write_text(json.dumps({"source": str(source.resolve()), "source_plan": plan,
                                                       "end": end, "files": counts}, indent=2), encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    print(json.dumps(subset(args.source, args.target, args.end), indent=2))


if __name__ == "__main__":
    main()
