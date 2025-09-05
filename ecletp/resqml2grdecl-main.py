#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ijk_etp_ecl.py
--------------

ETP connection + main pipeline.
- Discovers RESQML IjkGridRepresentation over ETP (FETPAPI)
- Builds FESAPI repository
- Calls conversion/writers from resqml_to_grdecl.py
- Supports Mode A (default) and Mode B (--local-xy)
- Optional EGRID export via XTGeo

Usage examples:
  # Mode A (default: convert to projected/global CRS, identity MAPAXES)
  python ijk_etp_ecl.py --out ./outgrid --grid <uuid>

  # Mode B (keep local XY, write MAPAXES from LocalDepth3dCrs)
  python ijk_etp_ecl.py --local-xy --out ./outgrid --grid <uuid>
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

# Third-party runtime deps
import fetpapi
import fesapi

# Local converter
from resqml2grdecl import (
    xtgeo_grid_from_fesapi_grid,
    write_grdecl_geometry,
    write_property_grdecl,
    grid_parity_left,
    k_is_down,
    compute_mapaxes_from_local_depth_crs,
)

# Config loader (user-provided)
try:
    from config_etp import Config as ETPConfig  # rddms_host, data_partition_id, dataspace_uri, token
except ImportError as e:
    print("ERROR: config_etp.py not found:", e, file=sys.stderr)
    sys.exit(2)

log = logging.getLogger("etp_xtgeo_grdecl")

# ------------------------------- Timing helpers ------------------------------- #
def now() -> float: return time.perf_counter()
def dt(t0: float) -> float: return now() - t0

# ------------------------------- Retry helper -------------------------------- #
def _retry_with_reconnect(do_call, reopen, retries: int = 3, base_delay: float = 0.5):
    retry_on = ("EOF detected", "SSL SYSCALL", "Unexpected Exception")
    for attempt in range(retries):
        try:
            return do_call()
        except Exception as exc:
            msg = str(exc)
            if not any(tag in msg for tag in retry_on) or attempt == retries - 1:
                raise
            delay = base_delay * (attempt + 1)
            log.warning(
                "Transient ETP error (%s). Reconnecting and retrying (%d/%d) after %.1fs ...",
                msg, attempt + 1, retries, delay
            )
            try:
                reopen()
            except Exception as rex:
                log.debug("Reconnect failed: %s", rex)
            time.sleep(delay)

# ------------------------------ ETP session helpers --------------------------- #
def open_etp_session(ws_url: str, authorization: str, partition: str, timeout_s: float = 15.0):
    """Create and start a FETPAPI client session."""
    init = fetpapi.InitializationParameters(str(os.getpid()), ws_url)
    if partition:
        hdrs = fetpapi.MapStringString()
        hdrs["data-partition-id"] = partition
        init.setAdditionalHandshakeHeaderFields(hdrs)
    session = fetpapi.createClientSession(init, authorization)
    t = threading.Thread(target=session.run, name="fetpapi-run", daemon=True)
    t.start()
    start = time.perf_counter()
    while session.isEtpSessionClosed() and (time.perf_counter() - start) < timeout_s:
        time.sleep(0.25)
    if session.isEtpSessionClosed():
        raise RuntimeError("ETP session could not be established.")
    return session

@dataclass
class EtpResource:
    uri: str
    name: str = ""

def _ctx_for(uri: str, depth: int, include_sources: bool, include_targets: bool):
    ctx = fetpapi.ContextInfo()
    ctx.uri = uri
    ctx.depth = depth
    ctx.navigableEdges = fetpapi.RelationshipKind_Both
    ctx.includeSecondaryTargets = bool(include_targets)
    ctx.includeSecondarySources = bool(include_sources)
    return ctx

def list_resources(session, dataspace: str) -> List[EtpResource]:
    ctx = _ctx_for(dataspace, depth=1, include_sources=False, include_targets=False)
    res_list = session.getResources(ctx, fetpapi.ContextScopeKind__self)
    return [EtpResource(uri=r.uri, name=r.name) for r in res_list]

def discover_ijk_grids(session, dataspace: str) -> List[EtpResource]:
    return [r for r in list_resources(session, dataspace) if "ijkgridrepresentation" in r.uri.lower()]

def expand_related(session, root_uri: str, max_depth: int = 3) -> List[EtpResource]:
    """Breadth-first expansion of related resources in both directions (sources+targets)."""
    seen: Set[str] = set()
    queue: List[Tuple[str, int]] = [(root_uri, 0)]
    out: List[EtpResource] = []
    while queue:
        uri, d = queue.pop(0)
        if uri in seen or d > max_depth:
            continue
        seen.add(uri)
        ctx = _ctx_for(uri, depth=1, include_sources=True, include_targets=True)
        res_list = session.getResources(ctx, fetpapi.ContextScopeKind_targetsOrSelf)
        for r in res_list:
            out.append(EtpResource(uri=r.uri, name=r.name))
            if r.uri not in seen:
                queue.append((r.uri, d + 1))
    return out

def fetch_data_objects(session, uris: Iterable[str]):
    uri_map = fetpapi.MapStringString()
    for i, u in enumerate(uris):
        uri_map[str(i)] = u
    return session.getDataObjects(uri_map)

# ------------------------------ FESAPI repository ----------------------------- #
def build_repo(session, dataobjects):
    """Create a FESAPI DataObjectRepository tied to the ETP session (HDF via ETP)."""
    repo = fesapi.DataObjectRepository()
    repo.setHdfProxyFactory(fetpapi.FesapiHdfProxyFactory(session))
    for dobj in dataobjects.values():
        uri = dobj.resource.uri
        repo.addOrReplaceGsoapProxy(dobj.data, fetpapi.getDataObjectType(uri), fetpapi.getDataspaceUri(uri))
    try:
        fetpapi.setSessionOfHdfProxies(repo, session)
    except Exception as e:
        log.debug("setSessionOfHdfProxies skipped: %s", e)
    return repo

# ----------------------------------- Utilities -------------------------------- #
def _uuid_from_uri(uri: str) -> Optional[str]:
    try:
        l = uri.rfind("("); r = uri.rfind(")")
        if l != -1 and r != -1 and r > l:
            u = uri[l+1:r]
            return u if 8 <= len(u) <= 36 else None
    except Exception:
        pass
    return None

def _filter_grids_by_uuid(grids: List[EtpResource], uuid_list: List[str]) -> List[EtpResource]:
    if not uuid_list: return grids
    wanted = {u.lower() for u in uuid_list}
    kept = []
    for g in grids:
        u = (_uuid_from_uri(g.uri) or "").lower()
        if u in wanted:
            kept.append(g)
    return kept

# ----------------------------------- Main pipeline --------------------------- #
def process_all(cfg: ETPConfig, out_dir: str, filter_uuids: List[str], export_egrid: bool, local_xy_mode: bool):
    os.makedirs(out_dir, exist_ok=True)
    log.info("Connecting to ETP... ws=%s, partition=%s, dataspace=%s",
             cfg.rddms_host, cfg.data_partition_id, cfg.dataspace_uri)
    auth = f"Bearer {cfg.token}"
    t_connect = now()
    session = open_etp_session(cfg.rddms_host, auth, cfg.data_partition_id)
    log.debug("Connected in %.3fs", dt(t_connect))

    def reopen():
        nonlocal session
        try: session.close()
        except Exception: pass
        session = open_etp_session(cfg.rddms_host, auth, cfg.data_partition_id)

    try:
        # Discover
        t_disc = now()
        grids = _retry_with_reconnect(lambda: discover_ijk_grids(session, cfg.dataspace_uri), reopen)
        log.debug("Discovery in %.3fs", dt(t_disc))
        if not grids:
            log.info("No IjkGridRepresentation found in %s", cfg.dataspace_uri)
            return
        grids_all = len(grids)
        grids = _filter_grids_by_uuid(grids, filter_uuids)
        if filter_uuids:
            log.info("Filtered grids by UUID: %d matched of %d discovered", len(grids), grids_all)
        log.info("Found %d grid(s) to export", len(grids))

        for g in grids:
            log.info("Processing grid: %s (%s)", g.name, g.uri)
            # Related objects
            t_rel = now()
            related = _retry_with_reconnect(lambda: expand_related(session, g.uri, max_depth=3), reopen)
            log.debug("Related object count (incl. grid): %d (%.3fs)", len(related), dt(t_rel))
            # Fetch objects
            t_fetch = now()
            uris = list({r.uri for r in related})
            if g.uri not in uris: uris.append(g.uri)
            dataobjects = _retry_with_reconnect(lambda: fetch_data_objects(session, uris), reopen)
            log.debug("Fetch DataObjects in %.3fs (uris=%d)", dt(t_fetch), len(uris))
            # Build repo
            t_repo = now()
            repo = build_repo(session, dataobjects)
            log.debug("Repo build in %.3fs", dt(t_repo))
            if repo.getIjkGridRepresentationCount() == 0:
                log.warning("No IJK grid loaded for %s", g.uri)
                continue
            grid = repo.getIjkGridRepresentation(0)

            # Convert to XTGeo & build GRDECL arrays
            t_xt = now()
            xgrid, props, coordsv, zcorn4, crs = xtgeo_grid_from_fesapi_grid(
                grid, repo, grid_title=(g.name or "GRID"), local_xy_mode=local_xy_mode
            )
            log.debug("XTGeo grid+props in %.3fs", dt(t_xt))

            # Geometry GRDECL
            t_wr = now()
            safe_name = (g.name or "grid").replace(" ", "_")
            out_geom = os.path.join(out_dir, f"{safe_name}.grdecl")

            left = grid_parity_left(grid)
            kdown = k_is_down(grid)
            gdorient = f"INC INC INC {'DOWN' if kdown else 'UP'} {'LEFT' if left else 'RIGHT'}"

            if local_xy_mode:
                # Mode B: local XY with MAPAXES from CRS
                if crs is not None:
                    try:
                        mapaxes = compute_mapaxes_from_local_depth_crs(crs)
                    except Exception:
                        log.warning("MAPAXES computation failed; using identity.")
                        mapaxes = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
                else:
                    mapaxes = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            else:
                # Mode A: global/proj XY, identity MAPAXES
                mapaxes = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

            write_grdecl_geometry(
                xgrid, coordsv, zcorn4, out_geom,
                include_map_headers=True,
                mapaxes=mapaxes,
                units="METRES",
                gdorient=gdorient,
            )
            log.info("Exported geometry GRDECL: %s", out_geom)

            # Property GRDECLs
            for kw, gp, meta in props:
                write_property_grdecl(out_dir, safe_name, kw, gp, meta)
            log.debug("Writing done in %.3fs", dt(t_wr))

            # Optional EGRID
            if export_egrid:
                egrid_path = os.path.join(out_dir, f"{safe_name}.EGRID")
                try:
                    xgrid.to_file(egrid_path, fformat="egrid")
                    log.info("Exported EGRID: %s", egrid_path)
                except Exception as e:
                    log.warning("EGRID export failed: %s", e)

    finally:
        try: session.close()
        except Exception: pass

# --------------------------------------- CLI --------------------------------- #
def _bool_flag_default_true(parser: argparse.ArgumentParser, name: str, help_text: str):
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(f"--{name}", dest=name, action="store_true", help=help_text + " (default)")
    grp.add_argument(f"--no-{name}", dest=name, action="store_false", help="Disable " + help_text)
    parser.set_defaults(**{name: True})

def _parse_uuid_list(val: Optional[str]) -> List[str]:
    if not val: return []
    items: List[str] = []
    for chunk in val.replace(";", ",").split(","):
        s = chunk.strip()
        if s: items.append(s)
    return items

def main():
    ap = argparse.ArgumentParser(
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
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s"
    )

    cfg = ETPConfig.from_env()
    uuids = _parse_uuid_list(args.grid)

    process_all(cfg, args.out, uuids, export_egrid=args.egrid, local_xy_mode=bool(args.local_xy))

if __name__ == "__main__":
    sys.exit(main())
