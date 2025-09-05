#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETP → RESQML → XTGeo → GRDECL (and optional EGRID)

Creates Eclipse GRDECL from RESQML 2.0/2.0.1 IjkGridRepresentation fetched over ETP.
Two geometry export modes are supported:

Mode A (default):
  - Convert RESQML LocalDepth3dCrs XYZ to *projected/global* CRS using FESAPI
    convertXyzPointsToGlobalCrs(...). Write GRDECL with identity MAPAXES.

Mode B (--local-xy):
  - Keep *local XY* in GRDECL, normalize Z to *depth positive down*, and write MAPAXES
    computed from LocalDepth3dCrs (origin + areal rotation). This mirrors how your
    drogon.txt example carries MAPAXES/GDORIENT.

COORD & ZCORN construction:
  - COORD: per-pillar TOP/BOTTOM endpoints (prefer coordinate line nodes; fallback to
    owning cell corners). GRDECL expects straight coordinate lines (pillars).
  - ZCORN: per-cell corner depths in Eclipse natural order (8 per cell).

References (design intent and data layout):
  - Eclipse GRDECL file as an IJK grid: COORD pillars, ZCORN node depths, monotonic Z
    and mapping semantics. https://docs.energistics.org/RESQML/RESQML_TOPICS/RESQML-000-292-0-C-sv2010.html
  - GRDECL conventions (ordering, natural order, depths positive down):
    https://docs.opengosim.com/manual/input_deck/grid/grdecl/
  - LocalDepth3dCrs semantics (translation + areal rotation; local↔global conversion):
    https://docs.energistics.org/RESQML/RESQML_TOPICS/RESQML-000-129-0-C-sv2010.html
    and FESAPI LocalDepth3dCrs API: https://f2i-consulting.com/fesapi/doxygen/0.16/class_r_e_s_q_m_l2__0__1___n_s_1_1_local_depth3d_crs.html
  - Column-layer/IJK grids and parity/K-direction considerations:
    https://docs.energistics.org/RESQML/RESQML_TOPICS/RESQML-500-179-0-R-sv2010.html

Property export over ETP (Python/SWIG wrapper):
  - Generic templated getDataObjects<T>() isn’t exposed; robust pattern is to iterate
    repo.getUuids() and filter by type and supporting representation. (FESAPI guidance)
    https://discourse.f2i-consulting.com/t/reading-of-seismiclattice-from-dataobjectrepository/167
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
    from config_etp import Config as ETPConfig  # Your local config loader
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

# ------------------------------ UUID helpers & filter ------------------------- #
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

# ------------------------------- Geometry constants --------------------------- #
TOP_SW, TOP_SE, TOP_NW, TOP_NE = 0, 1, 2, 3
BOT_SW, BOT_SE, BOT_NW, BOT_NE = 4, 5, 6, 7
Q_SW, Q_SE, Q_NW, Q_NE = 0, 1, 2, 3  # XTGeo quadrants at node/interface

# ------------------------------- CRS helpers --------------------------------- #
def _find_crs_for_grid(grid, repo):
    """Return the grid's LocalDepth3dCrs or LocalEngineeringCompoundCrs if found, else None."""
    try:
        if hasattr(grid, "getCrs"):
            crs = grid.getCrs()
            if crs: return crs
    except Exception:
        pass
    try:
        for obj in repo.getTargetObjects(grid):
            tag = obj.getXmlTag() if hasattr(obj, "getXmlTag") else ""
            if "LocalDepth3dCrs" in tag or "LocalEngineeringCompoundCrs" in tag:
                return obj
    except Exception:
        pass
    try:
        for uid in repo.getUuids():
            obj = repo.getDataObjectByUuid(uid)
            tag = obj.getXmlTag() if hasattr(obj, "getXmlTag") else ""
            if "LocalDepth3dCrs" in tag or "LocalEngineeringCompoundCrs" in tag:
                return obj
    except Exception:
        pass
    return None

def _crs_is_up_oriented(crs) -> bool:
    """Return True if vertical axis is up (so depths require sign flip)."""
    try:
        return bool(crs.isVerticalUpOriented())
    except Exception:
        try:
            return bool(crs.isUpOriented())
        except Exception:
            return False  # assume depth positive down

def _compute_mapaxes_from_local_depth_crs(crs):
    """
    Return MAPAXES triple ((x1,y1),(x2,y2),(x3,y3)) from CRS origin & areal rotation.
    This expresses mapping of local (0,0), (1,0), (0,1) to projected XY.
    """
    import math
    # Accessors naming varies a bit across FESAPI versions; try a few
    def _val(getter_names, default=0.0):
        for g in getter_names:
            try:
                return float(getattr(crs, g)())
            except Exception:
                pass
        return float(default)
    x0 = _val(["getOriginOrdinal1", "getXOffset", "getProjectedOriginEasting"], 0.0)
    y0 = _val(["getOriginOrdinal2", "getYOffset", "getProjectedOriginNorthing"], 0.0)
    theta = _val(["getArealRotation"], 0.0)  # radians by standard; some builds use degrees—assume radians per spec
    ct, st = math.cos(theta), math.sin(theta)
    p1 = (x0, y0)              # local (0,0)
    p2 = (x0 + ct, y0 + st)    # local (1,0)
    p3 = (x0 - st, y0 + ct)    # local (0,1)
    return (p1, p2, p3)

# -------------------------- Orientation / parity helpers --------------------- #
def _grid_parity_left(grid) -> bool:
    """Return True if grid parity is left-handed."""
    try:
        return not bool(grid.isRightHanded())
    except Exception:
        return False  # default RIGHT

def _k_is_down(grid) -> bool:
    try:
        if hasattr(grid, "isKDirectionDown"):
            return bool(grid.isKDirectionDown())
        kd = str(grid.getKDirection())
        return "down" in kd.lower()
    except Exception:
        return True  # default DOWN

# -------------------------- Geometry accessors & builders -------------------- #
def _points_accessor(grid, repo, local_xy_mode: bool):
    """
    Return get_xyz(idx) yielding XYZ prepared for GRDECL and a flag (z_up) indicating if
    original CRS is up-oriented. Behavior:
      - Mode A (local_xy_mode=False): convert to global CRS if possible (preferred).
      - Mode B (local_xy_mode=True): keep local XY (do not convert).
    In both modes: normalize Z later to *depth positive down* for GRDECL consumers.
    """
    npts = grid.getXyzPointCountOfAllPatches()
    buf = fesapi.DoubleArray(npts * 3)
    grid.getXyzPointsOfAllPatches(buf)

    crs = _find_crs_for_grid(grid, repo)
    z_up = False

    if crs is not None:
        z_up = _crs_is_up_oriented(crs)
        if not local_xy_mode:
            try:
                crs.convertXyzPointsToGlobalCrs(buf, npts, False)
                log.debug("Converted %d points to global CRS using %s",
                          npts, getattr(crs, "getXmlTag", lambda: type(crs))())
            except Exception as e:
                log.warning("CRS conversion failed (%s). Falling back to LOCAL XY; will write MAPAXES.", e)
    else:
        log.warning("No CRS linked to grid; using LOCAL XY and identity MAPAXES.")
        z_up = False  # unknown; assume depths already down

    def get_xyz(idx: int):
        j = idx * 3
        x, y, z = buf.getitem(j), buf.getitem(j + 1), buf.getitem(j + 2)
        # Normalize to *depth positive down* for GRDECL
        if z_up:
            z = -z
        return (x, y, z)

    return get_xyz, crs

def _build_coord_from_pillars(grid, ni: int, nj: int, nk: int, get_xyz) -> np.ndarray:
    """
    Build COORD using pillar endpoints TOP/BOTTOM, not arbitrary cell corners.
    Prefer coordinate line nodes if available; fallback to owning cell corner indices.
    Output shape: (ni+1, nj+1, 6), dtype float64.
    """
    out = np.empty(((ni + 1), (nj + 1), 6), dtype=np.float64)

    def _owner(i: int, j: int):
        if i < ni and j < nj: return i, j, TOP_SW, BOT_SW
        if i == ni and j < nj: return i - 1, j, TOP_SE, BOT_SE
        if i < ni and j == nj: return i, j - 1, TOP_NW, BOT_NW
        return i - 1, j - 1, TOP_NE, BOT_NE

    for j in range(nj + 1):
        for i in range(ni + 1):
            # Try coordinate line node accessors (if exposed by the Python wrapper)
            top_idx = bot_idx = None
            # Some builds offer getXyzPointIndexFromCoordinateLineNode(i, j, interfaceK)
            for try_api in ("getXyzPointIndexFromCoordinateLineNode",):
                if hasattr(grid, try_api):
                    try:
                        top_idx = getattr(grid, try_api)(i, j, 0)
                        bot_idx = getattr(grid, try_api)(i, j, nk)
                        break
                    except Exception:
                        top_idx = bot_idx = None
            if top_idx is None or bot_idx is None:
                ci, cj, ctop, cbot = _owner(i, j)
                top_idx = grid.getXyzPointIndexFromCellCorner(ci, cj, 0, ctop)
                bot_idx = grid.getXyzPointIndexFromCellCorner(ci, cj, nk - 1, cbot)

            tx, ty, tz = get_xyz(top_idx)
            bx, by, bz = get_xyz(bot_idx)
            out[i, j, 0:3] = (tx, ty, tz)
            out[i, j, 3:6] = (bx, by, bz)

    return np.ascontiguousarray(out, dtype=np.float64)

def _build_zcorn_xtgeo_auto(grid, ni: int, nj: int, nk: int, get_xyz, default_workers: int = 8) -> np.ndarray:
    """
    Build ZCORN in XTGeo 4-D shape: (ni+1, nj+1, nk+1, 4), dtype float32.
    """
    z4 = np.empty((ni + 1, nj + 1, nk + 1, 4), dtype=np.float32)

    def fill_k_range(k0: int, k1: int, target: np.ndarray):
        for k in range(k0, k1):
            for j in range(nj):
                for i in range(ni):
                    # TOP interface k
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, TOP_SW)
                    target[i, j, k, Q_NE] = np.float32(get_xyz(idx)[2])
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, TOP_SE)
                    target[i + 1, j, k, Q_NW] = np.float32(get_xyz(idx)[2])
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, TOP_NW)
                    target[i, j + 1, k, Q_SE] = np.float32(get_xyz(idx)[2])
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, TOP_NE)
                    target[i + 1, j + 1, k, Q_SW] = np.float32(get_xyz(idx)[2])
                    # BOTTOM interface k+1
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, BOT_SW)
                    target[i, j, k + 1, Q_NE] = np.float32(get_xyz(idx)[2])
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, BOT_SE)
                    target[i + 1, j, k + 1, Q_NW] = np.float32(get_xyz(idx)[2])
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, BOT_NW)
                    target[i, j + 1, k + 1, Q_SE] = np.float32(get_xyz(idx)[2])
                    idx = grid.getXyzPointIndexFromCellCorner(i, j, k, BOT_NE)
                    target[i + 1, j + 1, k + 1, Q_SW] = np.float32(get_xyz(idx)[2])

    workers = max(1, int(default_workers or 1))
    if workers > 1 and nk >= 2:
        import concurrent.futures
        try:
            t0 = now()
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
            log.warning("Parallel ZCORN build failed (%s). Rebuilding sequentially (workers=1).", exc)

    t1 = now()
    fill_k_range(0, nk, z4)
    log.debug("ZCORN built sequentially in %.3fs", dt(t1))
    return z4

def _flatten_zcorn_ecl(z4: np.ndarray, ni: int, nj: int, nk: int) -> np.ndarray:
    """
    Flatten from (ni+1, nj+1, nk+1, 4) to Eclipse flat (8*ni*nj*nk,) in order:
    TOP: SW, SE, NW, NE, then BOTTOM: SW, SE, NW, NE for each cell (i,j,k).
    """
    out = np.empty((8 * ni * nj * nk,), dtype=np.float32)
    pos = 0
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                # TOP 4 at interface k
                out[pos] = z4[i, j, k, Q_NE]; pos += 1  # TOP_SW
                out[pos] = z4[i + 1, j, k, Q_NW]; pos += 1  # TOP_SE
                out[pos] = z4[i, j + 1, k, Q_SE]; pos += 1  # TOP_NW
                out[pos] = z4[i + 1, j + 1, k, Q_SW]; pos += 1  # TOP_NE
                # BOTTOM 4 at interface k+1
                out[pos] = z4[i, j, k + 1, Q_NE]; pos += 1  # BOT_SW
                out[pos] = z4[i + 1, j, k + 1, Q_NW]; pos += 1  # BOT_SE
                out[pos] = z4[i, j + 1, k + 1, Q_SE]; pos += 1  # BOT_NW
                out[pos] = z4[i + 1, j + 1, k + 1, Q_SW]; pos += 1  # BOT_NE
    return out

# -------------------------- Property mapping & XTGeo assembly ---------------- #
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
    # 1) OSDU ref/name mapping wins if available
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
    # 2) Title maps directly if known
    if title and title.upper() in ECL_TO_OSDU:
        kw = title.upper(); meta = ECL_TO_OSDU.get(kw, {})
        return _sanitize_keyword(kw, "PROP", used), {
            "osdu_ref": meta.get("osdu_ref", ""), "osdu_name": meta.get("osdu_name", ""),
            "resqml_kind_expected": meta.get("resqml_property_kind", ""), "uom_expected": meta.get("uom", ""),
        }
    # 3) Fallback to sanitized title
    kw = _sanitize_keyword(title or "PROP", "PROP", used)
    return kw, {"osdu_ref": "", "osdu_name": osdu_name or "", "resqml_kind_expected": "", "uom_expected": ""}

def _xtgeo_grid_from_fesapi_grid(grid, repo, grid_title: str, local_xy_mode: bool):
    """
    Returns:
      xgrid: xtgeo.Grid
      props: List[(kw, xtgeo.GridProperty, meta dict)]
      coordsv: np.ndarray (float64, (nx+1, ny+1, 6))
      zcorn4: np.ndarray (float32, (nx+1, ny+1, nz+1, 4))
      crs:    the CRS object if found (used for MAPAXES)
    """
    ni, nj, nk = grid.getICellCount(), grid.getJCellCount(), grid.getKCellCount()

    # Access points per selected mode
    get_xyz, crs = _points_accessor(grid, repo, local_xy_mode=local_xy_mode)

    # Load split info before corner index queries (FESAPI guidance)
    try:
        if hasattr(grid, "loadSplitInformation"):
            grid.loadSplitInformation()
            log.debug("Loaded split information for IJK grid.")
        # COORD via pillar endpoints
        t_coord = now()
        coordsv = _build_coord_from_pillars(grid, ni, nj, nk, get_xyz)  # float64
        log.debug("COORD built in %.3fs", dt(t_coord))
        # ZCORN (XTGeo-native 4D)
        t_z = now()
        zcorn4 = _build_zcorn_xtgeo_auto(grid, ni, nj, nk, get_xyz, default_workers=8)
        log.debug("ZCORN built in %.3fs", dt(t_z))
    finally:
        try:
            if hasattr(grid, "unloadSplitInformation"):
                grid.unloadSplitInformation()
                log.debug("Unloaded split information.")
        except Exception as e:
            log.debug("unloadSplitInformation skipped: %s", e)

    # ACTNUM default = all active
    actnumsv = np.ones((ni, nj, nk), dtype=np.int32)

    # XTGeo Grid
    t_grid = now()
    xgrid = xtgeo.Grid(coordsv=coordsv, zcornsv=zcorn4, actnumsv=actnumsv, name=grid_title)
    log.debug("XTGeo Grid constructed in %.3fs", dt(t_grid))

    # Properties: UUID scan approach (robust with Python wrappers)
    props_out: List[Tuple[str, xtgeo.GridProperty, Dict[str, str]]] = []
    used: Dict[str, int] = {}
    try:
        uuids = list(repo.getUuids())
    except Exception:
        uuids = []
    t_props = now()
    for uid in uuids:
        try:
            obj = repo.getDataObjectByUuid(uid)
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
                values = np.asarray(list(raw), dtype=(np.float32 if is_cont else np.int32))
            elif hasattr(obj, "getValues"):
                raw = obj.getValues(0)
                values = np.asarray(list(raw), dtype=(np.float32 if is_cont else np.int32))
            else:
                continue
            gp = xtgeo.GridProperty(
                ncol=ni, nrow=nj, nlay=nk,
                name=kw, discrete=bool(is_disc),
                values=np.ascontiguousarray(values),
                grid=xgrid
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
                    xgrid.actnumsv = gp.values.astype(np.int32).reshape(ni, nj, nk)
                    log.debug("Applied ACTNUM from property; %d active cells", int(np.count_nonzero(xgrid.actnumsv)))
                    continue  # do not write ACTNUM as separate property
                except Exception as e:
                    log.debug("Failed to apply ACTNUM from property, keeping default: %s", e)
                    continue
            props_out.append((kw, gp, meta))
        except Exception as exc:
            log.debug("Skipping object %s due to error: %s", uid, exc)

    if props_out:
        xgrid.props = xtgeo.GridProperties([p[1] for p in props_out])
    log.debug("Properties processed in %.3fs (count ~%d)", dt(t_props), len(props_out))
    return xgrid, props_out, coordsv, zcorn4, crs

# --------------------------------- GRDECL writers ---------------------------- #
def _write_wrapped(f, vals: Iterable[float | int], fmt: str):
    line = 0
    for v in vals:
        f.write(fmt % v); line += 1
        if line % 8 == 0: f.write("\n")
    if line % 8: f.write("\n")

def _write_grdecl_geometry(
    xgrid: xtgeo.Grid,
    coordsv: np.ndarray,
    zcorn4: np.ndarray,
    out_path: str,
    include_map_headers: bool = True,
    mapaxes: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
    units: str = "METRES",
    gdorient: str = "INC INC INC DOWN RIGHT",
):
    """
    Write GRDECL geometry for the grid:
      - SPECGRID, COORD, ZCORN, ACTNUM
    And optional headers:
      - MAPUNITS / GRIDUNIT / GDORIENT / MAPAXES
    """
    ni = xgrid.ncol
    nj = xgrid.nrow
    nk = xgrid.nlay
    zcorn_flat = _flatten_zcorn_ecl(zcorn4, ni, nj, nk)
    if mapaxes is None:
        mapaxes = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"-- Exported by etp_xtgeo_grdecl\n")
        f.write(f"-- Grid: {getattr(xgrid, 'name', '')}\n\n")
        if include_map_headers:
            f.write("MAPUNITS\n")
            f.write(f"'{units:<8}' /\n\n")
            f.write("GRIDUNIT\n")
            f.write(f"'{units:<8}' ' ' /\n\n")
            f.write("GDORIENT\n")
            f.write(f"{gdorient} /\n\n")
            f.write("MAPAXES\n")
            f.write(f"{mapaxes[0][0]} {mapaxes[0][1]}\n")
            f.write(f"{mapaxes[1][0]} {mapaxes[1][1]}\n")
            f.write(f"{mapaxes[2][0]} {mapaxes[2][1]} /\n\n")
        f.write("SPECGRID\n")
        f.write(f" {ni} {nj} {nk} 1 /\n")
        f.write("COORD\n")
        _write_wrapped(f, coordsv.reshape(-1), " %.6f")
        f.write("/\n")
        f.write("ZCORN\n")
        _write_wrapped(f, zcorn_flat, " %.6f")
        f.write("/\n")
        f.write("ACTNUM\n")
        if getattr(xgrid, "actnumsv", None) is not None:
            _write_wrapped(f, xgrid.actnumsv.reshape(-1), " %d")
        else:
            _write_wrapped(f, np.ones(ni * nj * nk, dtype=np.int32), " %d")
        f.write("/\n")

def _write_property_grdecl(out_dir: str, grid_name: str, kw: str, gp: xtgeo.GridProperty, meta: Dict[str, str]):
    safe_name = grid_name.replace(" ", "_") or "grid"
    out_path = os.path.join(out_dir, f"{safe_name}__{kw}.grdecl")
    with open(out_path, "w", encoding="utf-8") as f:
        title = meta.get("title", ""); resqml_kind = meta.get("resqml_kind", ""); uom = meta.get("uom", "")
        osdu_name = meta.get("osdu_name", ""); osdu_ref = meta.get("osdu_ref", "")
        if osdu_name or osdu_ref:
            f.write(f"-- OSDU: {osdu_name} REF: {osdu_ref}\n")
        f.write(f"-- Property: {title} RESQML kind: {resqml_kind} UOM: {uom}\n")
        f.write(f"{kw}\n")
        if getattr(gp, "discrete", False):
            _write_wrapped(f, gp.values.astype(np.int64), " %d")
        else:
            _write_wrapped(f, gp.values.astype(float), " %.6g")
        f.write("/\n")
    log.info("Exported property: %s", out_path)

def _compose_gdorient(grid, zcorn4):
    """
    Compose a GDORIENT header string using both RESQML flags and the ZCORN geometry as a safety-net.
    Inputs:
    grid   : FESAPI IjkGridRepresentation (used for handedness & RESQML K flag)
    zcorn4 : np.ndarray with shape (ni+1, nj+1, nk+1, 4) containing interface corner depths
             in XTGeo 4D layout (as built in _xtgeo_grid_from_fesapi_grid).
    Returns:
    gdorient : str         Formatted as: "INC INC INC <DOWN|UP> <LEFT|RIGHT>"
    kdown    : bool        Final K-direction used (True if DOWN)
    left     : bool        Final handedness used (True if LEFT)
    """
    import numpy as _np

    def _infer_kdown_from_zcorn4(z4, tol=1e-6):
        """Infer if K increases downwards by comparing successive interface mean depths."""
        # Expect shape (ni+1, nj+1, nk+1, 4)
        if z4 is None or z4.ndim != 4 or z4.shape[-1] != 4:
            return None  # can't infer
        nkp1 = z4.shape[2]
        if nkp1 < 2:
            return None
        # Mean depth for each interface k (average over all 4 corners & all i,j)
        iface_means = [_float(z4[:, :, k, :].mean()) for k in range(nkp1)]
        # If depths increase with k (positive-down), it's K-DOWN.
        return iface_means[1] > iface_means[0] + tol

    def _float(x):
        try:
            return float(x)
        except Exception:
            return x
    # 1) Parity from RESQML (LEFT if cross(I,J) > 0) using FESAPI helper already in the script
    left = _grid_parity_left(grid)

    # 2) K-direction from RESQML
    kdown = _k_is_down(grid)

    # 3) Geometry-based inference from ZCORN (safety-net)
    kdown_geom = _infer_kdown_from_zcorn4(zcorn4)
    if kdown_geom is not None and kdown_geom != kdown:
        # Prefer the *geometry* to avoid header/data mismatches
        try:
            log.warning(
                "K-direction inferred from geometry (%s) != RESQML flag (%s); using geometry.",
                'DOWN' if kdown_geom else 'UP',
                'DOWN' if kdown else 'UP'
            )
        except Exception:
            pass
        kdown = kdown_geom

    gdorient, kdown, left = _compose_gdorient(grid, zcorn4)
    return gdorient, bool(kdown), bool(left)

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
            # XTGeo grid + props + arrays
            t_xt = now()
            xgrid, props, coordsv, zcorn4, crs = _xtgeo_grid_from_fesapi_grid(
                grid, repo, grid_title=(g.name or "GRID"), local_xy_mode=local_xy_mode
            )
            log.debug("XTGeo grid+props in %.3fs", dt(t_xt))

            # Geometry GRDECL
            t_wr = now()
            safe_name = (g.name or "grid").replace(" ", "_")
            out_geom = os.path.join(out_dir, f"{safe_name}.grdecl")

            left = _grid_parity_left(grid)
            kdown = _k_is_down(grid)
            gdorient = f"INC INC INC {'DOWN' if kdown else 'UP'} {'LEFT' if left else 'RIGHT'}"

            if local_xy_mode:
                # Mode B: local XY with MAPAXES from CRS
                if crs is not None:
                    try:
                        mapaxes = _compute_mapaxes_from_local_depth_crs(crs)
                    except Exception:
                        log.warning("MAPAXES computation failed; using identity.")
                        mapaxes = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
                else:
                    mapaxes = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            else:
                # Mode A: global/proj XY, identity MAPAXES
                mapaxes = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

            _write_grdecl_geometry(
                xgrid, coordsv, zcorn4, out_geom,
                include_map_headers=True,
                mapaxes=mapaxes,
                units="METRES",
                gdorient=gdorient,
            )
            log.info("Exported geometry GRDECL: %s", out_geom)

            # Property GRDECLs (always separate)
            for kw, gp, meta in props:
                _write_property_grdecl(out_dir, safe_name, kw, gp, meta)
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

    cfg = ETPConfig.from_env()  # rddms_host, data_partition_id, dataspace_uri, token
    uuids = _parse_uuid_list(args.grid)

    process_all(cfg, args.out, uuids, export_egrid=args.egrid, local_xy_mode=bool(args.local_xy))

if __name__ == "__main__":
    sys.exit(main())
