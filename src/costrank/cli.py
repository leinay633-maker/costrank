"""All commands are offline except the explicit free-data collect command."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import pandas as pd
from .config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="costrank")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Print environment; do not access the network")
    syn = sub.add_parser("synthetic", help="Accounting/CPU fixture, not financial evidence")
    syn.add_argument("--output", required=True)
    syn.add_argument("--end", default="2020-12-31")
    syn.add_argument("--symbols", type=int, default=60)
    syn.add_argument("--seed", type=int, default=7)
    collect = sub.add_parser("collect", help="Explicit free BaoStock download")
    collect.add_argument("--output", required=True)
    collect.add_argument("--config", default="configs/v01.json")
    collect.add_argument("--start", default="2016-01-01")
    collect.add_argument("--end")
    collect.add_argument("--limit-symbols", type=int, default=0)
    collect.add_argument("--pause", type=float, default=0.2)
    collect.add_argument("--freeze")
    collect.add_argument("--resolutions", help="Curated share-exchange/cash delisting outcomes CSV")
    acq = sub.add_parser("acquire", help="Explicit free BaoStock download into a resumable raw cache (schema v3)")
    acq.add_argument("--output", required=True)
    acq.add_argument("--config", default="configs/v01.json")
    acq.add_argument("--start", default="2016-01-01")
    acq.add_argument("--end")
    acq.add_argument("--stages", help="Comma-separated subset of the acquisition stages; default all")
    acq.add_argument("--budget", type=int, default=45_000, help="Requests per Beijing calendar day")
    acq.add_argument("--pause", type=float, default=0.2)
    acq.add_argument("--min30-top", type=int, default=2500)
    acq.add_argument("--through", help="Fetch only items dated on or before this day (same plan, shared cache)")
    acq.add_argument("--retries", type=int, default=3, help="Attempts per request (reconnecting between them)")
    acq.add_argument("--backoff", type=int, default=8, help="Longest wait in seconds between attempts")
    acq.add_argument("--freeze")
    asm = sub.add_parser("assemble", help="Seal a schema-3 snapshot from a completed acquire raw cache (offline)")
    asm.add_argument("--data", required=True)
    asm.add_argument("--config", default="configs/v01.json")
    asm.add_argument("--start", default="2016-01-01")
    asm.add_argument("--end")
    asm.add_argument("--resolutions", help="Curated share-exchange/cash delisting outcomes CSV")
    asm.add_argument("--min30-top", type=int, default=2500)
    asm.add_argument("--no-minutes", action="store_true",
                     help="Seal without the 30-minute table (open-fill decision path; RESEARCH section 9 supplement 11)")
    asm.add_argument("--partial", action="store_true",
                     help="Engineering preview before every per-code file exists (2019 and earlier debugging only)")
    imp = sub.add_parser("import-csv", help="Local licensed-provider adapter boundary")
    for key in ("output", "bars", "instruments", "calendar", "metadata"):
        imp.add_argument("--" + key, required=True)
    imp.add_argument("--actions")
    validate = sub.add_parser("validate")
    validate.add_argument("--data", required=True)
    validate.add_argument("--config", default="configs/v01.json")
    qlib = sub.add_parser("qlib-export")
    qlib.add_argument("--data", required=True)
    qlib.add_argument("--config", default="configs/v01.json")
    qlib.add_argument("--output", required=True)
    qlib.add_argument("--native-handler", action="store_true")
    execute = sub.add_parser("run")
    execute.add_argument("--config", default="configs/v01.json")
    execute.add_argument("--data", required=True)
    execute.add_argument("--output", default="runs")
    execute.add_argument("--allow-proxy", action="store_true")
    execute.add_argument("--holdout", action="store_true")
    execute.add_argument("--freeze")
    execute.add_argument("--ack-final", action="store_true")
    execute.add_argument("--workers", type=int, default=1, help="Parallel scenario processes (Linux fork only)")
    paper = sub.add_parser("daily", help="Forward paper trading: one run per trading day after the close (docs/PAPER.md)")
    paper.add_argument("--root", required=True, help="Live raw cache, extended from the final snapshot's cache")
    paper.add_argument("--freeze", required=True)
    paper.add_argument("--protocol", required=True, help="Forward protocol JSON (live.Protocol)")
    paper.add_argument("--output", required=True)
    paper.add_argument("--day", help="Run only this session (default: every unrecorded forward session, in order)")
    paper.add_argument("--no-fetch", action="store_true")
    paper.add_argument("--refetch", action="store_true",
                       help="Set the --day session's cached market files aside (raw/rejected/) and fetch them again")
    lock = sub.add_parser("freeze")
    for key in ("data", "development-run", "output"):
        lock.add_argument("--"+key, required=True)
    lock.add_argument("--config", default="configs/v01.json")
    args = parser.parse_args(argv)
    from .data import read_bundle, synthetic
    from .experiment import check_freeze, freeze, run
    if args.command == "doctor":
        versions = {}
        for package in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "clarabel", "pyarrow", "baostock", "pyqlib"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        print(json.dumps({"python": sys.version, "platform": platform.platform(),
                          "logical_cpus": os.cpu_count(), "packages": versions}, indent=2))
        return
    if args.command == "synthetic":
        days = len(pd.bdate_range("2018-01-02", args.end))
        if days < 2 or args.symbols < 1:
            parser.error("Synthetic fixture needs >=2 days and >=1 symbol")
        synthetic(args.output, days=days, symbols=args.symbols, seed=args.seed)
        print(args.output)
        return
    if args.command == "daily":
        from .live import daily
        result = daily(args.root, args.freeze, args.protocol, args.output, day=args.day, do_fetch=not args.no_fetch,
                       refetch=args.refetch)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    if args.command == "import-csv":
        from .providers import import_csv
        import_csv(args.output, args.bars, args.instruments, args.calendar, args.metadata, args.actions)
        print(args.output)
        return
    config = Config.load(args.config)
    if args.command in {"collect", "acquire"}:
        end = args.end or config.dev_end
        if pd.Timestamp(args.start) < pd.Timestamp("2015-08-01") or not args.start < end <= config.holdout_end:
            parser.error("Collection outside supported scope")
        if end >= config.holdout_start:
            if not args.freeze:
                parser.error("Do not acquire final-holdout data before --freeze")
            check_freeze(args.freeze, config)
    if args.command == "collect":
        from .providers import collect_baostock
        collect_baostock(args.output, args.start, end, limit=args.limit_symbols, pause=args.pause,
                         resolutions=args.resolutions)
        print(args.output)
    elif args.command == "acquire":
        from .acquire import STAGES, acquire
        stages = tuple(args.stages.split(",")) if args.stages else STAGES
        summary = acquire(args.output, args.start, end, stages=stages, budget=args.budget, pause=args.pause,
                          min30_top=args.min30_top, through=args.through, retries=args.retries, backoff=args.backoff)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "assemble":
        from .snapshot import assemble
        bundle = assemble(args.data, args.start, args.end or config.dev_end, resolutions=args.resolutions,
                          min30_top=args.min30_top, partial=args.partial, minutes=not args.no_minutes)
        print(json.dumps({"bars": len(bundle.bars), "instruments": len(bundle.instruments),
                          "prefix_hash": bundle.prefix_hash(config.holdout_start)}, indent=2))
    elif args.command == "run":
        if args.holdout and not args.ack_final:
            parser.error("Holdout requires explicit --ack-final; its opening is recorded")
        result = run(config, args.data, args.output, holdout=args.holdout,
                     freeze_file=args.freeze, allow_proxy=args.allow_proxy, workers=args.workers)
        print(result)
    elif args.command == "freeze":
        print(freeze(args.output, config, args.data, args.development_run))
    else:
        bundle = read_bundle(args.data, allow_after=config.holdout_start)
        if args.command == "validate":
            print(json.dumps({"bars": len(bundle.bars), "instruments": len(bundle.instruments),
                "sessions": len(bundle.calendar), "schema_version": bundle.metadata["schema_version"],
                "delisting_resolutions": len(bundle.delistings), "source": bundle.metadata["source"],
                "corporate_actions": bundle.metadata["corporate_actions"],
                "prefix_hash": bundle.prefix_hash(config.holdout_start)}, indent=2))
        else:
            from .features import prepare, qlib_frame
            path = Path(args.output)
            path.mkdir(parents=True, exist_ok=False)
            frame = qlib_frame(prepare(bundle, config))
            frame.to_csv(path / "qlib_frame.csv.gz")
            if args.native_handler:
                from qlib.data.dataset import DataHandlerLP
                DataHandlerLP.from_df(frame).to_pickle(path / "handler.pkl", dump_all=True)
            print(path)


if __name__ == "__main__":
    main()
