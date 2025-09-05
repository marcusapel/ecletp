
# -*- coding: utf-8 -*-
"""
CLI entry point for RESQML -> GRDECL export over ETP.
Matches the behavior of the previous `resqml2grdecl-main.py`.
"""
from __future__ import annotations
import argparse
import logging
import sys
from .etp_pipeline import process_all
# Config loader (user-provided)
try:
    from .config_etp import Config as ETPConfig  # rddms_host, data_partition_id, dataspace_uri, token
except Exception as e:
    print("ERROR: config_etp.py not found or failed to import:", e, file=sys.stderr)
    raise

def _bool_flag_default_true(parser: argparse.ArgumentParser, name: str, help_text: str):
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(f"--{name}", dest=name, action="store_true", help=help_text + " (default)")
    grp.add_argument(f"--no-{name}", dest=name, action="store_false", help="Disable " + help_text)
    parser.set_defaults(**{name: True})

def _parse_uuid_list(val: str | None) -> list[str]:
    if not val: return []
    items: list[str] = []
    for chunk in val.replace(";", ",").split(","):
        s = chunk.strip()
        if s: items.append(s)
    return items

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog='resqml2grdecl',
        description="Export RESQML IJK grids over ETP to GRDECL (and optional EGRID) using XTGeo"
    )
    ap.add_argument("--out", default="./outgrid", help="Output directory for files (default: ./outgrid)")
    ap.add_argument("--grid", default="", help="Comma-separated list of IjkGridRepresentation UUIDs to export")
    ap.add_argument("--egrid", action="store_true", help="Also export EGRID binary grid (default: off)")
    ap.add_argument(
        "--local-xy", action="store_true",
        help="MODE B: keep LOCAL XY in GRDECL and write MAPAXES from LocalDepth3dCrs (default: off)"
    )
    _bool_flag_default_true(ap, "debug", "Verbose debug logging")
    return ap

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s"
    )
    cfg = ETPConfig.from_env()
    uuids = _parse_uuid_list(args.grid)
    process_all(cfg, args.out, uuids, export_egrid=args.egrid, local_xy_mode=bool(args.local_xy))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
