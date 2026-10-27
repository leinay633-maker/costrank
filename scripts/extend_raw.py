"""Hard-link a whole acquire() raw cache into a new directory whose plan ends later (RESEARCH
section 9's final range after the freeze, or the forward period), so `costrank acquire` on the
new plan fetches only what is missing: new sessions, the uncovered ranges of the trading calendar
and of index and report queries as new segments, and new per-code items. Cached files are never
rewritten, so the earlier history stays byte-identical. Offline.

usage: python scripts/extend_raw.py <acquire_dir> <new_dir> --end 2025-12-31
then:  costrank acquire --output <new_dir> --start <same start> --end 2025-12-31 ...
"""
import argparse
import json
import shutil
from pathlib import Path
from subset_raw import link


def extend(source, target, end):
    source, target = Path(source), Path(target)
    plan = json.loads((source / "collection-plan.json").read_text(encoding="utf-8"))
    if plan.get("schema") != 3 or not end > plan["end"]:
        raise ValueError("Need a schema-3 acquisition plan that ends before the new end")
    if target.exists():
        raise FileExistsError("Choose a new directory")
    counts = {}
    for path in sorted((source / "raw").rglob("*.csv")):
        relative = path.relative_to(source / "raw")
        link(path, target / "raw" / relative)
        group = relative.parts[0] if len(relative.parts) > 1 else "."
        counts[group] = counts.get(group, 0) + 1
    (target / "collection-plan.json").write_text(json.dumps(dict(plan, end=end), indent=2), encoding="utf-8")
    shutil.copy2(source / "request-ledger.json", target / "request-ledger.json")
    (target / "extended-from.json").write_text(json.dumps({"source": str(source.resolve()), "source_plan": plan,
                                                           "end": end, "files": counts}, indent=2), encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    print(json.dumps(extend(args.source, args.target, args.end), indent=2))


if __name__ == "__main__":
    main()
