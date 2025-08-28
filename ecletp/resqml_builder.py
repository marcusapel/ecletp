#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a RESQML model from an Eclipse corner-point grid and properties using resqpy.

- Creates EPC/HDF5 model & CRS
- Imports corner-point grid (7D array: (nk, nj, ni, 2, 2, 2, 3))
- Adds ACTNUM & other properties (facets & lookup supported)
- Optional GRDECL import (xtgeo)
- build_in_memory(source_grid, ...) accepts a parsed grid from grdecl_reader
  or creates a tiny synthetic demo if source_grid is None.

Author: Marcus Apel (Equinor) | Scaffolded by Copilot
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Any

import numpy as np
import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.rq_import as rqimp
import resqpy.property as rqp

try:
    import xtgeo  # optional, only needed for GRDECL import
    _HAS_XTGEO = True
except Exception:
    _HAS_XTGEO = False


__all__ = [
    "create_crs",
    "add_property",
    "build_from_corner_points",
    "build_from_grdecl",
    "build_in_memory",
]


# ---------------- CRS ----------------
def create_crs(model: rq.Model,
               *,
               xy_units: str = "m",
               z_units: str = "m",
               z_inc_down: bool = True,
               epsg_code: Optional[str] = None,
               rotation: float = 0.0,
               offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> rqc.Crs:
    """Create & register a LocalDepth3d CRS."""
    crs = rqc.Crs(
        parent_model=model,
        xy_units=xy_units,
        z_units=z_units,
        z_inc_down=z_inc_down,
        epsg_code=epsg_code,
        rotation=rotation,
        x_offset=offsets[0],
        y_offset=offsets[1],
        z_offset=offsets[2],
        title="LocalDepth3dCrs",
    )
    crs.create_xml(add_as_part=True)
    return crs


# ---------------- Property Mapping ----------------
ECL_TO_RESQML: Dict[str, Dict[str, Optional[str]]] = {
    "PORO": {"kind": "porosity", "uom": None, "facet_type": None, "facet": None},
    "NTG": {"kind": "net to gross ratio", "uom": None, "facet_type": None, "facet": None},
    "PERMX": {"kind": "rock permeability", "uom": "mD", "facet_type": "direction", "facet": "I"},
    "PERMY": {"kind": "rock permeability", "uom": "mD", "facet_type": "direction", "facet": "J"},
    "PERMZ": {"kind": "rock permeability", "uom": "mD", "facet_type": "direction", "facet": "K"},
    "ACTNUM": {"kind": "discrete", "uom": None, "facet_type": None, "facet": None},
    # Extend as needed (PVTNUM, SATNUM, saturations, multipliers, etc.)
}


def add_property(model: rq.Model,
                 grid_uuid,
                 array: np.ndarray,
                 keyword: str,
                 *,
                 string_lookup_uuid: Optional[str] = None,
                 citation_title: Optional[str] = None) -> rqp.Property:
    """
    Add a (cell) property to the model referencing the grid.
    Array must be shaped (nk, nj, ni) in RESQML order [k, j, i].
    """
    meta = ECL_TO_RESQML.get(keyword.upper(), {})
    prop = rqp.Property.from_array(
        parent_model=model,
        cached_array=array,
        support_uuid=grid_uuid,
        property_kind=meta.get("kind", "continuous"),
        indexable_element="cells",
        uom=meta.get("uom"),
        facet_type=meta.get("facet_type"),
        facet=meta.get("facet"),
        string_lookup_uuid=string_lookup_uuid,
        citation_title=citation_title or keyword.upper(),
    )
    prop.write_hdf5()
    prop.create_xml(add_as_part=True)
    return prop


# ---------------- Build from corner points ----------------
def build_from_corner_points(
    *,
    epc_path: str,
    cp_kjinxyz: np.ndarray,                      # (nk, nj, ni, 2, 2, 2, 3)
    actnum_kji: Optional[np.ndarray] = None,     # (nk, nj, ni)
    properties: Optional[Dict[str, np.ndarray]] = None,  # keyword -> array (nk,nj,ni)
    epsg_code: Optional[str] = None,
    title: str = "Eclipse Grid",
    xy_units: str = "m",
    z_units: str = "m",
    z_inc_down: bool = True,
    rotation: float = 0.0,
    offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    facies_lookup: Optional[Dict[int, str]] = None,
) -> rq.Model:
    """
    Creates a new RESQML model with:
      * LocalDepth3d CRS
      * IJK grid from corner points
      * ACTNUM & other properties
    """
    # 1) Model
    model = rq.Model(epc_file=epc_path, new_epc=True, create_basics=True, create_hdf5_ext=True)

    # 2) CRS
    crs = create_crs(
        model,
        xy_units=xy_units,
        z_units=z_units,
        z_inc_down=z_inc_down,
        epsg_code=epsg_code,
        rotation=rotation,
        offsets=offsets,
    )

    # 3) Grid import (in-memory only here)
    cp_shape = cp_kjinxyz.shape
    assert len(cp_shape) == 7 and cp_shape[-1] == 3, f"Corner points must be (nk, nj, ni, 2, 2, 2, 3), got {cp_shape}"

    _active_mask = actnum_kji.astype(bool) if actnum_kji is not None else None

    grid = rqimp.grid_from_cp(
        model,
        cp_kjinxyz,
        crs_uuid=crs.uuid,
        active_mask=_active_mask,  # optional; only if ACTNUM given
        # Valid modes in this resqpy: 'none', 'dots', 'ij_dots', 'morse', 'inactive'
        treat_as_nan=('inactive' if _active_mask is not None else 'none'),
    )

    # Persist geometry
    grid.write_hdf5()
    grid.create_xml(add_as_part=True)

    nk, nj, ni = cp_shape[0], cp_shape[1], cp_shape[2]

    # 4) ACTNUM (explicitly stored as discrete)
    if actnum_kji is not None:
        assert actnum_kji.shape == (nk, nj, ni), f"ACTNUM must be {(nk, nj, ni)}, got {actnum_kji.shape}"
        add_property(model, grid.uuid, actnum_kji.astype(np.int32), "ACTNUM")

    # 5) Categorical lookup (FACIES) – provide if you need categorical properties
    facies_lookup_uuid = None
    if facies_lookup:
        sl = rqp.StringLookup(model, int_to_str_dict=facies_lookup, title="FACIES_LOOKUP")
        sl.create_xml(add_as_part=True)
        facies_lookup_uuid = sl.uuid

    # 6) Other properties
    if properties:
        for kw, arr in properties.items():
            assert arr.shape == (nk, nj, ni), f"{kw} must be {(nk, nj, ni)}, got {arr.shape}"
            add_property(
                model,
                grid.uuid,
                arr,
                kw,
                string_lookup_uuid=(facies_lookup_uuid if kw.upper() == "FACIES" else None),
            )

    # 7) Finalize EPC
    model.store_epc()
    return model


# ---------------- GRDECL import using xtgeo ----------------
def build_from_grdecl(grdecl_path: str, epc_path: str, *, epsg_code: Optional[str] = None) -> rq.Model:
    """
    Read GRDECL via xtgeo, compute corner points, gather ACTNUM & properties,
    then build the RESQML model.
    """
    if not _HAS_XTGEO:
        raise ImportError("xtgeo is required for build_from_grdecl(). Install xtgeo or call build_from_corner_points().")

    grid = xtgeo.grid_from_file(grdecl_path)
    cp = grid.corner_points()  # (nk, nj, ni, 2, 2, 2, 3)
    nk, nj, ni, *_ = cp.shape

    act = getattr(grid, "actnum", None)
    if act is not None:
        act = np.asarray(act).reshape(nk, nj, ni)

    props: Dict[str, np.ndarray] = {}
    if hasattr(grid, "properties"):
        for p in grid.properties:
            name = p.name.upper()
            if name in ECL_TO_RESQML or name == "FACIES":
                arr = np.asarray(p.values).reshape(nk, nj, ni)
                props[name] = arr

    return build_from_corner_points(
        epc_path=epc_path,
        cp_kjinxyz=cp,
        actnum_kji=act,
        properties=props if props else None,
        epsg_code=epsg_code,
        title=os.path.splitext(os.path.basename(grdecl_path))[0],
    )


# ---------------- In-memory builder (used by ecletp.main) ----------------
def _make_synthetic_cp(nk=2, nj=2, ni=2, dx=100.0, dy=100.0, dz=10.0) -> np.ndarray:
    """Produce a simple orthogonal 7D corner-points array."""
    x = np.linspace(0.0, dx * ni, ni + 1)
    y = np.linspace(0.0, dy * nj, nj + 1)
    z0 = 0.0

    cp = np.zeros((nk, nj, ni, 2, 2, 2, 3), dtype=float)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                zt, zb = z0 + k * dz, z0 + (k + 1) * dz
                i0, i1 = i, i + 1
                j0, j1 = j, j + 1
                cp[k, j, i, 0, 0, 0] = (x[i0], y[j0], zt)
                cp[k, j, i, 0, 0, 1] = (x[i1], y[j0], zt)
                cp[k, j, i, 0, 1, 0] = (x[i0], y[j1], zt)
                cp[k, j, i, 0, 1, 1] = (x[i1], y[j1], zt)
                cp[k, j, i, 1, 0, 0] = (x[i0], y[j0], zb)
                cp[k, j, i, 1, 0, 1] = (x[i1], y[j0], zb)
                cp[k, j, i, 1, 1, 0] = (x[i0], y[j1], zb)
                cp[k, j, i, 1, 1, 1] = (x[i1], y[j1], zb)
    return cp


def build_in_memory(source_grid: Optional[Any] = None,
                    *,
                    epc_path: str = "in_memory.epc",
                    title_prefix: str = "In-Memory",
                    dataspace: Optional[str] = None) -> rq.Model:
    """
    Build a RESQML model entirely in memory.

    Parameters
    ----------
    source_grid : optional
        Parsed grid from ecletp.grdecl_reader (dict-like). Expected keys if provided:
          - 'cp' or 'corner_points': (nk,nj,ni,2,2,2,3) float64
          - 'actnum' (optional): (nk,nj,ni) int/bool
          - 'properties' (optional): dict[str, (nk,nj,ni)]
          - 'epsg', 'xy_units', 'z_units', 'z_inc_down' (optional metadata)
    epc_path : str
        Output EPC path.
    title_prefix : str
        Title prefix used for the grid/metadata (not a kwarg to grid_from_cp).
    dataspace : str | None
        Passed through (not used here, but accepted for CLI compatibility).

    Returns
    -------
    resqpy.model.Model
    """

    # If a grid object was provided (from your reader), use it; else make a tiny demo.
    if source_grid is not None:
        # Accept dict-like inputs
        get = (source_grid.get if isinstance(source_grid, dict) else lambda k, d=None: getattr(source_grid, k, d))

        cp = get("cp", None) or get("corner_points", None)
        if cp is None:
            # Fallback to demo if cp missing
            cp = _make_synthetic_cp(nk=3, nj=2, ni=2, dx=100.0, dy=100.0, dz=10.0)

        act = get("actnum", None)
        props = get("properties", None) or {}

        epsg = get("epsg", None)
        xy_units = get("xy_units", "m")
        z_units = get("z_units", "m")
        z_inc_down = bool(get("z_inc_down", True))

        model = build_from_corner_points(
            epc_path=epc_path,
            cp_kjinxyz=np.asarray(cp),
            actnum_kji=(np.asarray(act) if act is not None else None),
            properties={k.upper(): np.asarray(v) for k, v in props.items()},
            epsg_code=epsg,
            title=f"{title_prefix} Grid",
            xy_units=xy_units,
            z_units=z_units,
            z_inc_down=z_inc_down,
        )
        return model

    # No source grid: produce a small synthetic example (2x2x2)
    nk, nj, ni = 2, 2, 2
    cp = _make_synthetic_cp(nk=nk, nj=nj, ni=ni, dx=100.0, dy=100.0, dz=10.0)
    poro = np.full((nk, nj, ni), 0.25, dtype=float)
    ntg = np.ones((nk, nj, ni), dtype=float)
    permz = np.full((nk, nj, ni), 100.0, dtype=float)  # mD
    actnum = np.ones((nk, nj, ni), dtype=np.int32)

    model = build_from_corner_points(
        epc_path=epc_path,
        cp_kjinxyz=cp,
        actnum_kji=actnum,
        properties={"PORO": poro, "NTG": ntg, "PERMZ": permz},
        epsg_code=None,
        title=f"{title_prefix} Demo Grid",
    )
    return model


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build RESQML EPC from Eclipse corner-point grid.")
    ap.add_argument("epc", nargs="?", help="Output EPC file path")
    ap.add_argument("--grdecl", help="Input GRDECL file (requires xtgeo)")
    ap.add_argument("--epsg", help="EPSG code, e.g. 'EPSG:32632'", default=None)
    args = ap.parse_args()

    if args.grdecl:
        if not args.epc:
            args.epc = os.path.splitext(os.path.basename(args.grdecl))[0] + ".epc"
        build_from_grdecl(args.grdecl, args.epc, epsg_code=args.epsg)
    elif args.epc:
        # run synthetic in-memory build to the given EPC
        build_in_memory(epc_path=args.epc)
    else:
        print("Provide --grdecl or an EPC path; or import build_in_memory()/build_from_corner_points() from your code.")
