#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETP → RESQML → XTGeo → GRDECL
- Config-driven ETP session (no token logged)
- Finds RESQML IjkGridRepresentation + associated objects (CRS/properties/HDF/EML links)
- Builds XTGeo Grid + GridProperty in memory (documented kwargs only)
- OSDU→ECL keyword mapping via osdu_mapping.py (optional), with OSDU metadata in GRDECL comments
- Exports GRDECL to --out (default: ./outgrid)
- Logging: --debug/--no-debug (default: --debug)

2025-09-04 updates:
* FIX: Load/unload RESQML IJK split info around corner→XYZ index queries.
  (Required before getXyzPointIndexFromCellCorner; see F2I example.)
* ZCORN is built in XTGeo-native 4D shape: (nx+1, ny+1, nz+1, 4).
  For GRDECL export, it is flattened to Eclipse order.
* Auto-parallel ZCORN with 8 workers by default; safe fallback to 1 if any
  concurrent FESAPI call fails.
* Removed --workers CLI (automatic behavior).

Refs:
- F2I/FETPAPI Python example (loadSplitInformation before corner index): 
  https://github.com/F2I-Consulting/fetpapi/blob/main/python/example/etp_client_example.py
- XTGeo grid formats / constructors (project docs & issues):
  https://pypi.org/project/xtgeo/
  https://github.com/equinor/xtgeo/issues/556
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import numpy as np
    import xtgeo  # Grid, GridProperty, GridProperties
except Exception as e:
    print("ERROR: Missing required package(s) for XTGeo handling:", e, file=sys.stderr)
    sys.exit(2)

try:
    import fetpapi
    import fesapi
except ImportError as e:
    print("ERROR: Missing required packages (fetpapi, fesapi):", e, file=sys.stderr)
    sys.exit(2)

try:
    from config_etp import Config as ETPConfig
except ImportError as e:
    print("ERROR: config_etp.py not found:", e, file=sys.stderr)
    sys.exit(2)

# Optional OSDU mapping module (user-provided)
_OSDU_MAP = None
try:
    import osdu_mapping as _OSDU_MAP
except Exception:
    _OSDU_MAP = None

# Build lookup tables if mapping is present
ECL_TO_OSDU: Dict[str, dict] = {}
OSDU_NAME_TO_ECL: Dict[str, str] = {}
OSDU_REF_TO_ECL: Dict[str, str] = {}
if _OSDU_MAP is not None and hasattr(_OSDU_MAP, "OSDU_PROPERTY_MAP"):
    ECL_TO_OSDU = dict(_OSDU_MAP.OSDU_PROPERTY_MAP)
    for ecl_kw, meta in ECL_TO_OSDU.items():
        name = (meta.get("osdu_name") or "").lower()
        ref = meta.get("osdu_ref")
        if name:
            OSDU_NAME_TO_ECL[name] = ecl_kw
        if ref:
            OSDU_REF_TO_ECL[ref] = ecl_kw

log = logging.getLogger("etp_xtgeo_grdecl")

# ------------------------------ Timing helpers ------------------------------ #
def now() -> float: return time.perf_counter()
def dt(t0: float) -> float: return now() - t0

# ------------------------------ Retry helper -------------------------------- #
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

# --------------------------- ETP session helpers ---------------------------- #
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
    res_list = session.getResources(ctx, fetpapi.ContextScopeKind__self)  # __self (double underscore)
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
        res_list = session.getResources(ctx, fetpapi.ContextScopeKind_targetsOrSelf)  # single underscore
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

# --------------------------- FESAPI repository ------------------------------ #
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

# ----------------------- URI/UUID helpers & filtering ----------------------- #
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

# -------- Geometry helpers: COORD (float64, (ni+1,nj+1,6)) & ZCORN (4D) ---- #
TOP_SW, TOP_SE, TOP_NW, TOP_NE = 0, 1, 2, 3
BOT_SW, BOT_SE, BOT_NW, BOT_NE = 4, 5, 6, 7

# XTGeo quadrant order at a node/interface: [SW, SE, NW, NE] => indices 0..3
Q_SW, Q_SE, Q_NW, Q_NE = 0, 1, 2, 3

def _points_accessor(grid):
    npts = grid.getXyzPointCountOfAllPatches()
    arr = fesapi.DoubleArray(npts * 3)
    grid.getXyzPointsOfAllPatches(arr)
    def get_xyz(idx: int) -> Tuple[float, float, float]:
        return (arr.getitem(idx * 3), arr.getitem(idx * 3 + 1), arr.getitem(idx * 3 + 2))
    return get_xyz

def _build_coord(grid, ni: int, nj: int, nk: int, get_xyz) -> np.ndarray:
    """Return coordsv shaped (ni+1, nj+1, 6), dtype float64."""
    out = np.empty(((ni + 1), (nj + 1), 6), dtype=np.float64)
    for j in range(nj + 1):
        jj = min(j, nj - 1)
        for i in range(ni + 1):
            ii = min(i, ni - 1)
            top_idx = grid.getXyzPointIndexFromCellCorner(ii, jj, 0, TOP_SW)
            bot_idx = grid.getXyzPointIndexFromCellCorner(ii, jj, nk - 1, BOT_SW)
            bx, by, bz = get_xyz(bot_idx)
            tx, ty, tz = get_xyz(top_idx)
            out[i, j, 0:3] = (bx, by, bz)
            out[i, j, 3:6] = (tx, ty, tz)
    return np.ascontiguousarray(out, dtype=np.float64)

def _build_zcorn_xtgeo_auto(grid, ni: int, nj: int, nk: int, get_xyz, default_workers: int = 8) -> np.ndarray:
    """
    Build ZCORN in XTGeo 4-D shape: (ni+1, nj+1, nk+1, 4), dtype float32.
    Try with parallel workers (default 8); if any exception occurs during parallel
    FESAPI calls, fall back to sequential (workers=1).

    Mapping from cell corners to node/quadrant (interface index k for TOP, k+1 for BOTTOM):
      For cell (i,j,k):
        TOP_SW -> node (i,   j,   k)   quad NE
        TOP_SE -> node (i+1, j,   k)   quad NW
        TOP_NW -> node (i,   j+1, k)   quad SE
        TOP_NE -> node (i+1, j+1, k)   quad SW
        BOT_SW -> node (i,   j,   k+1) quad NE
        BOT_SE -> node (i+1, j,   k+1) quad NW
        BOT_NW -> node (i,   j+1, k+1) quad SE
        BOT_NE -> node (i+1, j+1, k+1) quad SW
    """
    z4 = np.empty((ni + 1, nj + 1, nk + 1, 4), dtype=np.float32)

    def fill_k_range(k0: int, k1: int, target: np.ndarray):
        for k in range(k0, k1):
            for j in range(nj):
                for i in range(ni):
                    # TOP 4 at interface k
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, TOP_SW)
                    target[i,     j,     k, Q_NE] = get_xyz(idx)[2]
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, TOP_SE)
                    target[i + 1, j,     k, Q_NW] = get_xyz(idx)[2]
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, TOP_NW)
                    target[i,     j + 1, k, Q_SE] = get_xyz(idx)[2]
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, TOP_NE)
                    target[i + 1, j + 1, k, Q_SW] = get_xyz(idx)[2]
                    # BOTTOM 4 at interface k+1
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, BOT_SW)
                    target[i,     j,     k + 1, Q_NE] = get_xyz(idx)[2]
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, BOT_SE)
                    target[i + 1, j,     k + 1, Q_NW] = get_xyz(idx)[2]
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, BOT_NW)
                    target[i,     j + 1, k + 1, Q_SE] = get_xyz(idx)[2]
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, BOT_NE)
                    target[i + 1, j + 1, k + 1, Q_SW] = get_xyz(idx)[2]

    # Parallel attempt
    workers = max(1, int(default_workers or 1))
    if workers > 1 and nk >= 2:
        import concurrent.futures
        try:
            t0 = now()
            # Split by k-chunks; each worker writes to disjoint k-slices
            chunks: List[Tuple[int, int]] = []
            step = max(1, nk // workers)
            start = 0
            while start < nk:
                end = min(nk, start + step)
                chunks.append((start, end))
                start = end
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(fill_k_range, a, b, z4) for (a, b) in chunks]
                for f in concurrent.futures.as_completed(futs):
                    f.result()
            log.debug("ZCORN built with workers=%d in %.3fs", workers, dt(t0))
            return z4
        except Exception as exc:
            log.warning(
                "Parallel ZCORN build failed (%s). FESAPI likely not thread-safe in this build. "
                "Rebuilding sequentially (workers=1)...", exc
            )

    # Sequential fallback
    t1 = now()
    fill_k_range(0, nk, z4)
    log.debug("ZCORN built sequentially in %.3fs", dt(t1))
    return z4

# --------- Flatten ZCORN (4-D → Eclipse flat) for GRDECL export ------------- #
def _flatten_zcorn_ecl(z4: np.ndarray, ni: int, nj: int, nk: int) -> np.ndarray:
    """
    Flatten from (ni+1, nj+1, nk+1, 4) to Eclipse (8*ni*nj*nk,) in order:
    TOP: SW, SE, NW, NE, then BOTTOM: SW, SE, NW, NE for each cell (i,j,k).
    This is the inverse of the mapping used in _build_zcorn_xtgeo_auto.
    """
    out = np.empty((8 * ni * nj * nk,), dtype=np.float64)
    pos = 0
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                # TOP 4 at interface k
                out[pos] = z4[i,     j,     k, Q_NE]; pos += 1  # TOP_SW
                out[pos] = z4[i + 1, j,     k, Q_NW]; pos += 1  # TOP_SE
                out[pos] = z4[i,     j + 1, k, Q_SE]; pos += 1  # TOP_NW
                out[pos] = z4[i + 1, j + 1, k, Q_SW]; pos += 1  # TOP_NE
                # BOTTOM 4 at interface k+1
                out[pos] = z4[i,     j,     k + 1, Q_NE]; pos += 1  # BOT_SW
                out[pos] = z4[i + 1, j,     k + 1, Q_NW]; pos += 1  # BOT_SE
                out[pos] = z4[i,     j + 1, k + 1, Q_SE]; pos += 1  # BOT_NW
                out[pos] = z4[i + 1, j + 1, k + 1, Q_SW]; pos += 1  # BOT_NE
    return out.astype(np.float32, copy=False)

# ---------------------- Property mapping & XTGeo assembly ------------------- #
def _sanitize_keyword(title: str, fallback: str = "PROP", used: Optional[Dict[str, int]] = None) -> str:
    import re
    base = re.sub(r"[^A-Za-z0-9_]", "_", (title or fallback).upper())[:8]
    if used is None:
        return base
    if base not in used:
        used[base] = 1
        return base
    n = used[base] + 1
    while True:
        suff = str(n)
        trimmed = (base[:8 - len(suff)] + suff)[:8]
        if trimmed not in used:
            used[trimmed] = 1
            used[base] = n
            return trimmed
        n += 1

def _map_property_osdu(title: Optional[str], osdu_name: Optional[str], osdu_ref: Optional[str],
                       resqml_kind: Optional[str], uom: Optional[str],
                       used: Dict[str, int]) -> Tuple[str, Dict[str, str]]:
    if osdu_ref and osdu_ref in OSDU_REF_TO_ECL:
        kw = OSDU_REF_TO_ECL[osdu_ref]; meta = ECL_TO_OSDU.get(kw, {})
        return _sanitize_keyword(kw, "PROP", used), {
            "osdu_ref": meta.get("osdu_ref", ""), "osdu_name": meta.get("osdu_name", ""),
            "resqml_kind_expected": meta.get("resqml_property_kind", ""), "uom_expected": meta.get("uom", ""),
        }
    if osdu_name:
        key = osdu_name.lower()
        if key in OSDU_NAME_TO_ECL:
            kw = OSDU_NAME_TO_ECL[key]; meta = ECL_TO_OSDU.get(kw, {})
            return _sanitize_keyword(kw, "PROP", used), {
                "osdu_ref": meta.get("osdu_ref", ""), "osdu_name": meta.get("osdu_name", osdu_name),
                "resqml_kind_expected": meta.get("resqml_property_kind", ""), "uom_expected": meta.get("uom", ""),
            }
    if title and title.upper() in ECL_TO_OSDU:
        kw = title.upper(); meta = ECL_TO_OSDU.get(kw, {})
        return _sanitize_keyword(kw, "PROP", used), {
            "osdu_ref": meta.get("osdu_ref", ""), "osdu_name": meta.get("osdu_name", ""),
            "resqml_kind_expected": meta.get("resqml_property_kind", ""), "uom_expected": meta.get("uom", ""),
        }
    kw = _sanitize_keyword(title or "PROP", "PROP", used)
    return kw, {"osdu_ref": "", "osdu_name": osdu_name or "", "resqml_kind_expected": "", "uom_expected": ""}

def _xtgeo_grid_from_fesapi_grid(grid, repo, grid_title: str) -> Tuple[xtgeo.Grid, List[Tuple[str, xtgeo.GridProperty, Dict[str, str]]]]:
    ni, nj, nk = grid.getICellCount(), grid.getJCellCount(), grid.getKCellCount()
    get_xyz = _points_accessor(grid)

    # Load split info before any corner->point index queries
    try:
        if hasattr(grid, "loadSplitInformation"):
            grid.loadSplitInformation()
            log.debug("Loaded split information for IJK grid.")

        # COORD
        t_coord = now()
        coordsv = _build_coord(grid, ni, nj, nk, get_xyz)  # float64, (ni+1, nj+1, 6)
        log.debug("COORD built in %.3fs", dt(t_coord))

        # ZCORN: XTGeo-native (4D) with auto-parallel fallback
        t_zcorn = now()
        zcorn4 = _build_zcorn_xtgeo_auto(grid, ni, nj, nk, get_xyz, default_workers=8)
        log.debug("ZCORN built in %.3fs", dt(t_zcorn))
    finally:
        try:
            if hasattr(grid, "unloadSplitInformation"):
                grid.unloadSplitInformation()
                log.debug("Unloaded split information.")
        except Exception as e:
            log.debug("unloadSplitInformation skipped: %s", e)

    # ACTNUM (default all active)
    actnumsv = np.ones(ni * nj * nk, dtype=np.int32)

    # XTGeo Grid (documented kwargs only)
    t_grid = now()
    xgrid = xtgeo.Grid(coordsv=coordsv, zcornsv=zcorn4, actnumsv=actnumsv, name=grid_title)
    log.debug("XTGeo Grid constructed in %.3fs", dt(t_grid))

    # Properties
    props_out: List[Tuple[str, xtgeo.GridProperty, Dict[str, str]]] = []
    used: Dict[str, int] = {}
    try:
        count = repo.getDataObjectCount()
    except Exception:
        count = 0
    t_props = now()
    for i in range(count):
        try:
            obj = repo.getDataObject(i)
            otype = obj.getXmlTag() if hasattr(obj, "getXmlTag") else ""
            if "Property" not in otype:
                continue
            if hasattr(obj, "getRepresentation") and obj.getRepresentation() != grid:
                continue

            title = obj.getTitle() if hasattr(obj, "getTitle") else None
            osdu_name = title
            osdu_ref = None
            resqml_kind = getattr(obj, "getPropertyKindAsString", lambda: None)()
            uom = getattr(obj, "getUom", lambda: None)()
            kw, osdu_meta = _map_property_osdu(title or "", osdu_name, osdu_ref, resqml_kind, uom, used)

            is_cont = "ContinuousProperty" in otype
            is_disc = "DiscreteProperty" in otype
            values = None
            if is_cont and hasattr(obj, "getDoubleValuesOfPatch"):
                arr = fesapi.DoubleArray(ni * nj * nk)
                obj.getDoubleValuesOfPatch(0, arr)
                values = np.asarray([arr.getitem(x) for x in range(ni * nj * nk)], dtype=np.float32)
            elif hasattr(obj, "getValuesOfPatch"):
                raw = obj.getValuesOfPatch(0)
                values = np.asarray(list(raw), dtype=np.float32 if is_cont else np.int32)
            elif hasattr(obj, "getValues"):
                raw = obj.getValues(0)
                values = np.asarray(list(raw), dtype=np.float32 if is_cont else np.int32)
            else:
                continue

            values = np.ascontiguousarray(values, dtype=(np.float32 if is_cont else np.int32))
            gp = xtgeo.GridProperty(
                ncol=ni, nrow=nj, nlay=nk,
                name=kw, discrete=bool(is_disc),
                values=values, grid=xgrid
            )
            meta = {
                "title": title or "", "resqml_kind": str(resqml_kind or ""), "uom": str(uom or ""),
                "ecl_keyword": kw, "osdu_ref": osdu_meta.get("osdu_ref", ""),
                "osdu_name": osdu_meta.get("osdu_name", ""),
                "resqml_kind_expected": osdu_meta.get("resqml_kind_expected", ""),
                "uom_expected": osdu_meta.get("uom_expected", ""),
            }
            if kw == "ACTNUM":
                try:
                    xgrid.actnumsv = gp.values.astype(np.int32)
                    log.debug("Applied ACTNUM from property; %d active cells", int(np.count_nonzero(xgrid.actnumsv)))
                except Exception as e:
                    log.debug("Failed to apply ACTNUM from property, default all active: %s", e)
                continue

            props_out.append((kw, gp, meta))
        except Exception as exc:
            log.debug("Skipping property due to error: %s", exc)

    log.debug("Properties processed in %.3fs (count ~%d)", dt(t_props), len(props_out))
    if props_out:
        xgrid.props = xtgeo.GridProperties([p[1] for p in props_out])

    return xgrid, props_out

# ------------------------------- GRDECL writer ------------------------------ #
def _write_grdecl_from_xtgeo(xgrid: xtgeo.Grid, props: List[Tuple[str, xtgeo.GridProperty, Dict[str, str]]], out_path: str):
    ni = xgrid.ncol; nj = xgrid.nrow; nk = xgrid.nlay

    def _write_wrapped(f, vals: Iterable[float | int], fmt: str):
        line = 0
        for v in vals:
            f.write(fmt % v); line += 1
            if line % 8 == 0: f.write("\n")
        if line % 8: f.write("\n")

    # Prepare ZCORN flat in Eclipse order if needed
    zc = xgrid.zcornsv
    if isinstance(zc, np.ndarray) and zc.ndim == 4:
        zcorn_flat = _flatten_zcorn_ecl(zc, ni, nj, nk)
    else:
        # already flat
        zcorn_flat = np.asarray(zc).reshape(-1)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"-- Exported by etp_xtgeo_grdecl\n")
        f.write(f"-- Grid: {getattr(xgrid, 'name', '')}\n\n")

        f.write("SPECGRID\n")
        f.write(f" {ni} {nj} {nk} 1 /\n")

        f.write("COORD\n")
        _write_wrapped(f, xgrid.coordsv.reshape(-1), " %.6f")
        f.write("/\n")

        f.write("ZCORN\n")
        _write_wrapped(f, zcorn_flat, " %.6f")
        f.write("/\n")

        f.write("ACTNUM\n")
        if getattr(xgrid, "actnumsv", None) is not None:
            _write_wrapped(f, xgrid.actnumsv, " %d")
        else:
            _write_wrapped(f, np.ones(ni * nj * nk, dtype=np.int32), " %d")
        f.write("/\n")

        for kw, gp, meta in props:
            title = meta.get("title", ""); resqml_kind = meta.get("resqml_kind", ""); uom = meta.get("uom", "")
            osdu_name = meta.get("osdu_name", ""); osdu_ref = meta.get("osdu_ref", "")
            if osdu_name or osdu_ref:
                f.write(f"\n-- OSDU: {osdu_name}  REF: {osdu_ref}\n")
            f.write(f"-- Property: {title}  RESQML kind: {resqml_kind}  UOM: {uom}\n")
            f.write(f"{kw}\n")
            if getattr(gp, "discrete", False):
                _write_wrapped(f, gp.values.astype(np.int64), " %d")
            else:
                _write_wrapped(f, gp.values.astype(float), " %.6g")
            f.write("/\n")

# ------------------------------- Main pipeline ------------------------------ #
def process_all(cfg: ETPConfig, out_dir: str, filter_uuids: List[str]):
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
        # Dataspace probe
        t_ds = now()
        try:
            ds_list = session.getDataspaces()
            log.debug("Server dataspaces: %s", [d.uri for d in ds_list])
        except Exception as e:
            log.debug("Dataspace probe failed: %s", e)
        log.debug("Dataspace probe in %.3fs", dt(t_ds))

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
            if related:
                log.debug("Related sample: %s", [r.name or r.uri for r in related[:10]])

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

            # XTGeo grid + props
            t_xt = now()
            xgrid, props = _xtgeo_grid_from_fesapi_grid(grid, repo, grid_title=(g.name or "GRID"))
            log.debug("XTGeo grid+props in %.3fs", dt(t_xt))

            # Write GRDECL
            t_wr = now()
            safe_name = (g.name or "grid").replace(" ", "_")
            out_file = os.path.join(out_dir, f"{safe_name}.grdecl")
            _write_grdecl_from_xtgeo(xgrid, props, out_file)
            log.info("Exported: %s (props: %d) in %.3fs", out_file, len(props), dt(t_wr))

    finally:
        try: session.close()
        except Exception: pass

# ----------------------------------- CLI ----------------------------------- #
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
    ap = argparse.ArgumentParser(description="Export RESQML IJK grids over ETP to GRDECL using XTGeo")
    ap.add_argument("--out", default="./outgrid", help="Output directory for GRDECL files (default: ./outgrid)")
    ap.add_argument("--grid", default="", help="Comma-separated list of IjkGridRepresentation UUIDs to export")
    # NOTE: No --workers flag anymore; ZCORN build is auto-parallel (default 8), with safe fallback to 1.
    _bool_flag_default_true(ap, "debug", "Verbose debug logging")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    cfg = ETPConfig.from_env()  # rddms_host, data_partition_id, dataspace_uri, token
    uuids = _parse_uuid_list(args.grid)

    process_all(cfg, args.out, uuids)

if __name__ == "__main__":
    sys.exit(main())
