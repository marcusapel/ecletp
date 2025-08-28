#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a RESQML model from an Eclipse corner-point grid and properties using resqpy.

- Creates EPC/HDF5 model & CRS
- Imports corner-point grid (7D array: (nk, nj, ni, 2, 2, 2, 3))
- Adds ACTNUM & other properties (facets & lookup supported)
- Optional GRDECL import (xtgeo)
- Adds build_in_memory() for synthetic in-memory model

Author: Marcus Apel (Equinor) | Scaffolded by Copilot
"""

from __future__ import annotations
import os
from typing import Dict, Optional, Tuple

import numpy as np
import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.rq_import as rqimp
import resqpy.property as rqp

try:
    import xtgeo
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
    # Add more mappings as needed
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

    # 3) Grid (rq_import.grid_from_cp does NOT persist; we'll write next)
    cp_shape = cp_kjinxyz.shape
    assert len(cp_shape) == 7 and cp_shape[-1] == 3, f"Corner points must be (nk, nj, ni, 2, 2, 2, 3), got {cp_shape}"
    grid = rqimp.grid_from_cp(
        model,
        cp_kjinxyz,
        crs_uuid=crs.uuid,
        set_parent_window=False,
        title=title,
        treat_as_nan=np.nan,  # modern flag per docs
    )
    grid.write_hdf5()
    grid.create_xml(add_as_part=True)

    nk, nj, ni = cp_shape[0], cp_shape[1], cp_shape[2]

    # 4) ACTNUM
    if actnum_kji is not None:
        assert actnum_kji.shape == (nk, nj, ni), f"ACTNUM must be {(nk, nj, ni)}, got {actnum_kji.shape}"
        add_property(model, grid.uuid, actnum_kji.astype(np.int32), "ACTNUM")

    # 5) Categorical lookup (FACIES)
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

    # 7) Store EPC at end
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
            # take a conservative set; extend mapping as needed
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


# ---------------- In-memory tiny demo grid ----------------
def build_in_memory(*args, **kwargs) -> rq.Model:
    """
    Create a tiny synthetic 3D corner-point grid (2x2x2) entirely in memory,
    attach a few demo properties, and optionally persist to EPC if `epc_path` is provided.

    Parameters (all optional, taken from kwargs):
      epc_path: str or None (if provided, the model is written to disk)
      dx, dy, dz: float cell sizes (default: 100.0, 100.0, 10.0)

    Returns:
      resqpy.model.Model
    """
    epc_path = kwargs.get("epc_path", None)
    dx = float(kwargs.get("dx", 100.0))
    dy = float(kwargs.get("dy", 100.0))
    dz = float(kwargs.get("dz", 10.0))

    nk, nj, ni = 2, 2, 2

    # Build simple vertical pillars & orthogonal cells
    # Corner-points array: (nk, nj, ni, 2, 2, 2, 3) in [k,j,i] ordering
    x = np.array([0.0, dx, 2 * dx])
    y = np.array([0.0, dy, 2 * dy])
    z_top = 0.0
    z_bot = dz * nk

    cp = np.zeros((nk, nj, ni, 2, 2, 2, 3), dtype=float)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                # eight corners: (k-face 0/1, j-face 0/1, i-face 0/1)
                # z increases downwards (depth) – CRS will set z_inc_down=True
                cell_z_top = z_top + k * dz
                cell_z_bot = z_top + (k + 1) * dz
                # corners in local [i0,i1] x [j0,j1]
                i0, i1 = i, i + 1
                j0, j1 = j, j + 1

                # Lower face (k=0) & Upper face (k=1) (RESQML uses [k,j,i] axis for cp blocks)
                cp[k, j, i, 0, 0, 0] = (x[i0], y[j0], cell_z_top)  # k0, j0, i0
                cp[k, j, i, 0, 0, 1] = (x[i1], y[j0], cell_z_top)  # k0, j0, i1
                cp[k, j, i, 0, 1, 0] = (x[i0], y[j1], cell_z_top)  # k0, j1, i0
                cp[k, j, i, 0, 1, 1] = (x[i1], y[j1], cell_z_top)  # k0, j1, i1

                cp[k, j, i, 1, 0, 0] = (x[i0], y[j0], cell_z_bot)  # k1, j0, i0
                cp[k, j, i, 1, 0, 1] = (x[i1], y[j0], cell_z_bot)  # k1, j0, i1
                cp[k, j, i, 1, 1, 0] = (x[i0], y[j1], cell_z_bot)  # k1, j1, i0
                cp[k, j, i, 1, 1, 1] = (x[i1], y[j1], cell_z_bot)  # k1, j1, i1

    # Demo properties (shape (nk, nj, ni) in [k, j, i] order)
    poro = np.full((nk, nj, ni), 0.25, dtype=float)
    ntg = np.ones((nk, nj, ni), dtype=float)
    permz = np.full((nk, nj, ni), 100.0, dtype=float)  # mD
    actnum = np.ones((nk, nj, ni), dtype=np.int32)

    # Build the model (persist if epc_path is supplied)
    epc = epc_path or "in_memory.epc"
    model = build_from_corner_points(
        epc_path=epc,
        cp_kjinxyz=cp,
        actnum_kji=actnum,
        properties={"PORO": poro, "NTG": ntg, "PERMZ": permz},
        epsg_code=None,
        title="In-Memory Demo Grid",
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
        build_from_grdecl(args.grdecl, args.epc or "grid.epc", epsg_code=args.epsg)
    elif args.epc:
        # run synthetic in-memory build to the given EPC
        build_in_memory(epc_path=args.epc)
    else:
        print("Provide --grdecl or an EPC path; or import build_in_memory()/build_from_corner_points() from your code.")
