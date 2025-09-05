
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
resqml_to_grdecl.py
-------------------

Core conversion from RESQML (via FESAPI) to XTGeo + GRDECL artifacts.

Exposes:
- xtgeo_grid_from_fesapi_grid(grid, repo, grid_title, local_xy_mode) -> (xgrid, props, coordsv, zcorn4, crs)
- write_grdecl_geometry(xgrid, coordsv, zcorn4, out_path, include_map_headers=True, mapaxes=<id>, units="METRES", gdorient="...")
- write_property_grdecl(out_dir, grid_name, kw, gp, meta)
- grid_parity_left(grid) -> bool
- k_is_down(grid) -> bool
- compute_mapaxes_from_local_depth_crs(crs) -> ((x1,y1),(x2,y2),(x3,y3))

Design:
- Mode A (default): convert RESQML LocalDepth3dCrs XYZ to projected/global CRS, identity MAPAXES.
- Mode B (--local-xy in the pipeline): keep local XY, normalize Z to depth (positive down), write MAPAXES from CRS origin & rotation.

COORD is built from pillar endpoints (prefer coordinate line nodes). ZCORN is per-cell corner depths
in Eclipse ordering (8 per cell). Properties are discovered reliably by UUID scan (wrapper-friendly).

This module does NOT open ETP sessions. It expects a FESAPI grid + repository passed in.
"""

from __future__ import annotations
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Third-party runtime deps (import errors will surface to caller)
import fesapi
import xtgeo

log = logging.getLogger("resqml_to_grdecl")

# Optional OSDU mapping module
ECL_TO_OSDU: Dict[str, dict] = {}
OSDU_NAME_TO_ECL: Dict[str, str] = {}
OSDU_REF_TO_ECL: Dict[str, str] = {}
try:
    import osdu_mapping as _OSDU_MAP
    if hasattr(_OSDU_MAP, "OSDU_PROPERTY_MAP"):
        ECL_TO_OSDU = dict(_OSDU_MAP.OSDU_PROPERTY_MAP)
        for ecl_kw, meta in ECL_TO_OSDU.items():
            name = (meta.get("osdu_name") or "").lower()
            ref = meta.get("osdu_ref")
            if name:
                OSDU_NAME_TO_ECL[name] = ecl_kw
            if ref:
                OSDU_REF_TO_ECL[ref] = ecl_kw
except Exception:
    pass

# ------------------------------- Geometry constants --------------------------- #
TOP_SW, TOP_SE, TOP_NW, TOP_NE = 0, 1, 2, 3
BOT_SW, BOT_SE, BOT_NW, BOT_NE = 4, 5, 6, 7
Q_SW, Q_SE, Q_NW, Q_NE = 0, 1, 2, 3  # XTGeo quadrants at node/interface


# ---------------------------------- CRS helpers ------------------------------- #
def _find_crs_for_grid(grid, repo):
    """Return the grid's LocalDepth3dCrs/LocalEngineeringCompoundCrs if found, else None."""
    try:
        if hasattr(grid, "getCrs"):
            crs = grid.getCrs()
            if crs:
                return crs
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
def compute_mapaxes_from_local_depth_crs(crs):
    """
    Return MAPAXES triple ((x1,y1),(x2,y2),(x3,y3)) from CRS origin & areal rotation.
    This expresses mapping of local (0,0), (1,0), (0,1) to projected XY.
    """
    import math

    def _val(getter_names, default=0.0):
        for g in getter_names:
            try:
                return float(getattr(crs, g)())
            except Exception:
                pass
        return float(default)

    # RESQML LocalDepth3dCrs semantics: translation + areal rotation from projected CRS
    x0 = _val(["getOriginOrdinal1", "getXOffset", "getProjectedOriginEasting"], 0.0)
    y0 = _val(["getOriginOrdinal2", "getYOffset", "getProjectedOriginNorthing"], 0.0)
    theta = _val(["getArealRotation"], 0.0)  # radians per spec
    ct, st = math.cos(theta), math.sin(theta)
    p1 = (x0, y0)              # (0,0)
    p2 = (x0 + ct, y0 + st)    # (1,0)
    p3 = (x0 - st, y0 + ct)    # (0,1)
    return (p1, p2, p3)


# Make this public alias to satisfy requested API name
compute_mapaxes_from_local_depth_crs.__name__ = "compute_mapaxes_from_local_depth_crs"


# -------------------------- Orientation / parity helpers --------------------- #
def grid_parity_left(grid) -> bool:
    """True if grid parity is left-handed."""
    try:
        return not bool(grid.isRightHanded())
    except Exception:
        return False  # default RIGHT


def k_is_down(grid) -> bool:
    """True if K increases downward."""
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
    Return (get_xyz, crs) where get_xyz(idx)->(x,y,z) prepared for GRDECL.
    - Mode A (local_xy_mode=False): convert to global CRS if possible.
    - Mode B (local_xy_mode=True): keep local XY.
    In both: normalize Z to depth positive down.
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
                log.warning("CRS conversion failed (%s). Falling back to LOCAL XY; MAPAXES needed.", e)
    else:
        log.warning("No CRS on grid; using LOCAL XY; MAPAXES identity.")

    def get_xyz(idx: int):
        j = idx * 3
        x, y, z = buf.getitem(j), buf.getitem(j + 1), buf.getitem(j + 2)
        if z_up:
            z = -z  # GRDECL depths positive DOWN
        return (x, y, z)

    return get_xyz, crs


def _build_coord_from_pillars(grid, ni: int, nj: int, nk: int, get_xyz) -> np.ndarray:
    """
    Build COORD using pillar endpoints TOP/BOTTOM, not arbitrary cell corners.
    Prefer coordinate line node API if present; fallback to owning cell corner indices.
    Output: (ni+1, nj+1, 6), float64.
    """
    out = np.empty(((ni + 1), (nj + 1), 6), dtype=np.float64)

    def _owner(i: int, j: int):
        if i < ni and j < nj: return i, j, TOP_SW, BOT_SW
        if i == ni and j < nj: return i - 1, j, TOP_SE, BOT_SE
        if i < ni and j == nj: return i, j - 1, TOP_NW, BOT_NW
        return i - 1, j - 1, TOP_NE, BOT_NE

    for j in range(nj + 1):
        for i in range(ni + 1):
            top_idx = bot_idx = None
            if hasattr(grid, "getXyzPointIndexFromCoordinateLineNode"):
                try:
                    top_idx = grid.getXyzPointIndexFromCoordinateLineNode(i, j, 0)
                    bot_idx = grid.getXyzPointIndexFromCoordinateLineNode(i, j, nk)
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
    Build ZCORN in XTGeo 4-D shape: (ni+1, nj+1, nk+1, 4), float32.
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
            return z4
        except Exception as exc:
            log.warning("Parallel ZCORN build failed (%s). Rebuilding sequentially.", exc)

    fill_k_range(0, nk, z4)
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
    # 1) OSDU ref/name mapping wins
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
    # 3) Fallback sanitized
    kw = _sanitize_keyword(title or "PROP", "PROP", used)
    return kw, {"osdu_ref": "", "osdu_name": osdu_name or "", "resqml_kind_expected": "", "uom_expected": ""}


def xtgeo_grid_from_fesapi_grid(grid, repo, grid_title: str, local_xy_mode: bool):
    """
    Returns:
      xgrid: xtgeo.Grid
      props: List[(kw, xtgeo.GridProperty, meta dict)]
      coordsv: np.ndarray (float64, (nx+1, ny+1, 6))
      zcorn4: np.ndarray (float32, (nx+1, ny+1, nz+1, 4))
      crs:    CRS object if found
    """
    ni, nj, nk = grid.getICellCount(), grid.getJCellCount(), grid.getKCellCount()

    # Points accessor
    get_xyz, crs = _points_accessor(grid, repo, local_xy_mode=local_xy_mode)

    # Load split info around geometry extraction
    try:
        if hasattr(grid, "loadSplitInformation"):
            grid.loadSplitInformation()
            log.debug("Loaded split information for IJK grid.")
        coordsv = _build_coord_from_pillars(grid, ni, nj, nk, get_xyz)
        zcorn4 = _build_zcorn_xtgeo_auto(grid, ni, nj, nk, get_xyz, default_workers=8)
    finally:
        try:
            if hasattr(grid, "unloadSplitInformation"):
                grid.unloadSplitInformation()
        except Exception:
            pass

    actnumsv = np.ones((ni, nj, nk), dtype=np.int32)
    xgrid = xtgeo.Grid(coordsv=coordsv, zcornsv=zcorn4, actnumsv=actnumsv, name=grid_title)

    # Properties: robust enumeration via UUID scan
    props_out: List[Tuple[str, xtgeo.GridProperty, Dict[str, str]]] = []
    used: Dict[str, int] = {}
    try:
        uuids = list(repo.getUuids())
    except Exception:
        uuids = []
    for uid in uuids:
        try:
            obj = repo.getDataObjectByUuid(uid)
            otype = obj.getXmlTag() if hasattr(obj, "getXmlTag") else ""
            if "Property" not in otype:
                continue
            if hasattr(obj, "getRepresentation") and obj.getRepresentation() != grid:
                continue
            title = obj.getTitle() if hasattr(obj, "getTitle") else None
            osdu_name = title; osdu_ref = None
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
                    continue
                except Exception:
                    continue
            props_out.append((kw, gp, meta))
        except Exception:
            pass

    if props_out:
        xgrid.props = xtgeo.GridProperties([p[1] for p in props_out])

    return xgrid, props_out, coordsv, zcorn4, crs


# --------------------------------- GRDECL writers ---------------------------- #
def _write_wrapped(f, vals: Iterable[float | int], fmt: str):
    line = 0
    for v in vals:
        f.write(fmt % v); line += 1
        if line % 8 == 0: f.write("\n")
    if line % 8: f.write("\n")


def write_grdecl_geometry(
    xgrid: xtgeo.Grid,
    coordsv: np.ndarray,
    zcorn4: np.ndarray,
    out_path: str,
    include_map_headers: bool = True,
    mapaxes: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
    units: str = "METRES",
    gdorient: str = "INC INC INC DOWN RIGHT",
):
    """Write GEOMETRY (SPECGRID, COORD, ZCORN, ACTNUM) with optional headers."""
    ni = xgrid.ncol; nj = xgrid.nrow; nk = xgrid.nlay
    if mapaxes is None:
        mapaxes = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    zcorn_flat = _flatten_zcorn_ecl(zcorn4, ni, nj, nk)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"-- Exported by resqml_to_grdecl\n")
        f.write(f"-- Grid: {getattr(xgrid, 'name', '')}\n\n")
        if include_map_headers:
            f.write("MAPUNITS\n"); f.write(f"'%s' /\n\n" % units)
            f.write("GRIDUNIT\n"); f.write(f"'%s' ' ' /\n\n" % units)
            f.write("GDORIENT\n"); f.write(f"{gdorient} /\n\n")
            f.write("MAPAXES\n")
            f.write(f"{mapaxes[0][0]} {mapaxes[0][1]}\n")
            f.write(f"{mapaxes[1][0]} {mapaxes[1][1]}\n")
            f.write(f"{mapaxes[2][0]} {mapaxes[2][1]} /\n\n")
        f.write("SPECGRID\n"); f.write(f" {ni} {nj} {nk} 1 /\n")
        f.write("COORD\n"); _write_wrapped(f, coordsv.reshape(-1), " %.6f"); f.write("/\n")
        f.write("ZCORN\n"); _write_wrapped(f, zcorn_flat, " %.6f"); f.write("/\n")
        f.write("ACTNUM\n")
        if getattr(xgrid, "actnumsv", None) is not None:
            _write_wrapped(f, xgrid.actnumsv.reshape(-1), " %d")
        else:
            _write_wrapped(f, np.ones(ni * nj * nk, dtype=np.int32), " %d")
        f.write("/\n")


def write_property_grdecl(out_dir: str, grid_name: str, kw: str, gp: xtgeo.GridProperty, meta: Dict[str, str]):
    safe_name = grid_name.replace(" ", "_") or "grid"
    out_path = f"{out_dir.rstrip('/')}/{safe_name}__{kw}.grdecl"
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
