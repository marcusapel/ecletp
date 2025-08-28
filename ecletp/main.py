# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

"""
ecletp.main — uses only the exact public APIs in this repo:

- Grid: ecletp.grdecl_reader.read_grdecl(path)
    • When path is None, read_grdecl returns a built-in 2x2x3 demo grid and logs a warning.
- Builder: ecletp.resqml_builder.build_in_memory(grid, title_prefix, dataspace)
    • Returns a bundle with .crs_uuid, .grid_uuid, .property_uuids, .xml_by_uuid, .array_uris

No constructor/kwarg guessing. No type import from resqml_builder.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List
from uuid import UUID

from .grdecl_reader import read_grdecl           # exact API in your repo
from .resqml_builder import build_in_memory      # exact API in your repo

log = logging.getLogger("ecl2resqml_etp")


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


# ----------------------------------------------------------------------
# Build & Export
# ----------------------------------------------------------------------
def build_and_export(args: argparse.Namespace) -> None:
    # 1) Read grid (repo API). Returns (grid, meta)
    grid, meta = read_grdecl(args.eclgrd)

    # Optional summary (uses attributes from your GrdeclGrid dataclass)
    try:
        log.info(
            "Parsed grid: %dx%dx%d, geometry=%s",
            grid.ni, grid.nj, grid.nk, grid.geometry_kind
        )
        if args.verbose and meta:
            log.debug("GRDECL meta: %s", meta)
    except Exception:
        pass

    # 2) Build in-memory RESQML (repo API)
    title_prefix = args.title or (
        os.path.splitext(os.path.basename(args.eclgrd))[0] if args.eclgrd else "Demo"
    )
    # Duck-typed bundle: has .crs_uuid, .grid_uuid, .property_uuids, .xml_by_uuid, .array_uris
    bundle = build_in_memory(grid=grid, title_prefix=title_prefix, dataspace=args.dataspace)

    # 3) Emit XML parts to disk if requested
    if args.emit_xml_dir:
        out_dir = args.emit_xml_dir
        os.makedirs(out_dir, exist_ok=True)
        count = 0
        for u, xml_bytes in bundle.xml_by_uuid.items():
            with open(os.path.join(out_dir, f"{u}.xml"), "wb") as f:
                f.write(xml_bytes)
            count += 1
        log.info("Wrote %d XML parts to %s", count, out_dir)

    # 4) List DataArray UUID-path URIs (for Protocol 9 PutDataArrays) if requested
    if args.list_arrays:
        for label, (da_uri, arr) in bundle.array_uris.items():
            print(f"{label}: {da_uri} shape={arr.shape} dtype={arr.dtype}")

    # 5) Simple manifest (CRS, Grid, then Properties) if requested
    if args.print_manifest:
        ordering: List[UUID] = []
        ordering.append(bundle.crs_uuid)
        ordering.append(bundle.grid_uuid)
        ordering.extend(bundle.property_uuids.values())
        for u in ordering:
            print(str(u))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ecletp",
        description="Build an in-memory RESQML bundle from GRDECL and export XML/array info."
    )
    p.add_argument(
        "--eclgrd",
        type=str,
        default=None,
        help="Path to GRDECL/EGRID. If omitted, the built-in 2x2x3 demo grid is used."
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title prefix used for generated objects (defaults to case name or 'Demo')."
    )
    p.add_argument(
        "--dataspace",
        type=str,
        default="maap/m25test",  # matches default used by build_in_memory
        help="Dataspace to embed in suggested DataArray URIs."
    )
    p.add_argument(
        "--emit-xml-dir",
        type=str,
        default=None,
        help="Directory to write serialized XML parts (one file per UUID)."
    )
    p.add_argument(
        "--list-arrays",
        action="store_true",
        help="Print DataArray UUID-path URIs with shapes/dtypes (ETP Protocol 9)."
    )
    p.add_argument(
        "--print-manifest",
        action="store_true",
        help="Print a simple manifest (CRS, Grid, then Property UUIDs)."
    )
    p.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-vv for debug)."
    )
    return p


def cli(argv: List[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _configure_logging(args.verbose)
    build_and_export(args)


if __name__ == "__main__":
    cli()
