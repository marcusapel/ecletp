# ecletp/resqml_builder.py
from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np

# Public exports
__all__ = [
    "BuildBundle",
    "ArrayBinding",
    "build_in_memory",
    "build_from_corner_points",
    "add_property",
    "stable_uuid",
]

# ============================================================================
# UUIDs & metadata
# ============================================================================

# Stable namespace for deterministic UUIDs used in ETP paths & property identity
NAMESPACE = _uuid.UUID("9a2a3bbb-6b23-4f6e-9b1c-9e1b1eec1e06")


def stable_uuid(*name_parts: str) -> _uuid.UUID:
    """Deterministically derive a UUID5 from a set of name parts."""
    name = "|".join(str(p) for p in name_parts)
    return _uuid.uuid5(NAMESPACE, name)


@dataclass
class ArrayBinding:
    uuid: _uuid.UUID
    title: str
    indexable: str
    kind: str
    planned_hdf5_path: str
    shape: Tuple[int, ...]
    dtype: str
    uom: Optional[str] = None
    facet_type: Optional[str] = None
    facet: Optional[str] = None


@dataclass
class BuildBundle:
    mode: str  # "etp" or "resqpy"
    dataspace: str
    model: Optional[object] = None  # resqpy Model in "resqpy" mode
    grid_uuid: Optional[_uuid.UUID] = None
    crs_uuid: Optional[_uuid.UUID] = None
    property_index: Dict[str, _uuid.UUID] = field(default_factory=dict)
    # Registry for ETP (uuid -> ndarray)
    array_registry: Dict[_uuid.UUID, np.ndarray] = field(default_factory=dict)
    # Metadata for arrays (uuid -> binding)
    array_meta: Dict[_uuid.UUID, ArrayBinding] = field(default_factory=dict)


# Eclipse keyword → RESQML kind & uom mapping (extend as needed)
ECL_TO_RESQML = {
    "ACTNUM": ("discrete", None),
    "PORO": ("porosity", None),
    "NTG": ("net to gross ratio", None),
    "SATNUM": ("discrete", None),
    "PVTNUM": ("discrete", None),
    "PERMX": ("permeability rock", "m2"),
    "PERMY": ("permeability rock", "m2"),
    "PERMZ": ("permeability rock", "m2"),
}


def infer_kind_uom(keyword: str, values: np.ndarray) -> Tuple[str, Optional[str]]:
    kind, uom = ECL_TO_RESQML.get(keyword.upper(), (None, None))
    if kind is None:
        kind = "continuous" if values.dtype.kind in ("f", "c") else "discrete"
    return kind, uom


# ============================================================================
# Public entry points
# ============================================================================

def build_in_memory(
    xtg_grid,
    *,
    title_prefix: str = "ECL2RESQML",
    dataspace: str = "default",
    epsg: Optional[int] = None,
    mode: str = "etp",  # "etp" (no disk write) or "resqpy" (canonical EPC+HDF5)
    properties: Optional[Dict[str, np.ndarray]] = None,  # extra cell properties by keyword
) -> BuildBundle:
    """
    Build a grid + properties mapping ready for either:
      - ETP streaming ('etp' mode): no HDF5 writes; arrays in memory
      - Canonical RESQML dataset ('resqpy' mode): EPC + HDF5 on disk
    """
    if mode not in ("etp", "resqpy"):
        raise ValueError("mode must be 'etp' or 'resqpy'")

    bundle = BuildBundle(mode=mode, dataspace=dataspace)

    bundle = build_from_corner_points(
        xtg_grid,
        bundle=bundle,
        title_prefix=title_prefix,
        dataspace=dataspace,
        epsg=epsg,
        properties=properties or {},
    )
    return bundle


def build_from_corner_points(
    xtg_grid,
    *,
    bundle: BuildBundle,
    title_prefix: str,
    dataspace: str,
    epsg: Optional[int],
    properties: Dict[str, np.ndarray],
) -> BuildBundle:
    """
    Build from a corner-point representation (or derive it for cartesian demos).

    - For cartesian demos: derive corner points from (origin, spacing) that may be
      scalar/1D/2D/3D arrays (dx/dy/dz).
    - If xtgeo.Grid is passed: use its corner points and ACTNUM when available.
    """
    nk, nj, ni = xtg_grid.nk, xtg_grid.nj, xtg_grid.ni
    cp, actnum_kji = _get_corner_points_and_actnum(xtg_grid)

    # Deterministic grid & CRS UUIDs
    grid_title = f"{title_prefix}_Grid_{nk}x{nj}x{ni}"
    grid_uuid = stable_uuid(dataspace, "grid", grid_title)
    crs_uuid = stable_uuid(dataspace, "crs", str(epsg or "unknown"))
    bundle.grid_uuid = grid_uuid
    bundle.crs_uuid = crs_uuid

    # Planned HDF5 dataset paths (used both by resqpy & ETP registry)
    points_path = f"/RESQML/IjkGridRepresentation/{grid_uuid}/Points"
    actnum_path = f"/RESQML/GridProperty/{grid_uuid}/ACTNUM"

    # ----------------------------------------------------------------------
    # ETP-first mode: do NOT write EPC/HDF5. Keep arrays in memory & record metadata.
    # ----------------------------------------------------------------------
    if bundle.mode == "etp":
        # Register corner points
        _register_array(
            bundle,
            uuid=stable_uuid(dataspace, "grid", "points", str(grid_uuid)),
            title=f"{grid_title}_Points",
            indexable="cells",
            kind="points",
            array=cp,
            planned_path=points_path,
            uom=None,
        )
        # Register ACTNUM
        _register_array(
            bundle,
            uuid=stable_uuid(dataspace, "grid", "ACTNUM", str(grid_uuid)),
            title="ACTNUM",
            indexable="cells",
            kind="discrete",
            array=actnum_kji.astype(np.int32, copy=False),
            planned_path=actnum_path,
            uom=None,
        )
        # Register additional properties
        for kw, arr in properties.items():
            arr = np.asarray(arr)
            knd, uom = infer_kind_uom(kw, arr)
            path = f"/RESQML/GridProperty/{grid_uuid}/{kw.upper()}"
            _register_array(
                bundle,
                uuid=stable_uuid(dataspace, "grid", kw.upper(), str(grid_uuid)),
                title=kw.upper(),
                indexable="cells",
                kind=knd,
                array=arr,
                planned_path=path,
                uom=uom,
            )
        return bundle

    # ----------------------------------------------------------------------
    # resqpy mode: build a canonical RESQML dataset (writes HDF5)
    # ----------------------------------------------------------------------
    import resqpy.model as rq
    import resqpy.crs as rqc
    import resqpy.rq_import as rqi

    model = rq.Model(new_epc=True, epc_file=f"{title_prefix}.epc")
    bundle.model = model

    # CRS
    crs = rqc.Crs(
        model,
        title=f"{title_prefix}_CRS",
        epsg_code=epsg,
        xy_units="m",
        z_units="m",
        z_inc_down=True,
    )
    crs.create_xml()

    # Build grid from corner points
    grid = rqi.grid_from_cp(
        cp_array=cp,
        model=model,
        crs_uuid=crs.uuid,
        title=grid_title,
        max_z_void=0.1,
    )
    # Set inactive mask on grid
    grid.set_inactive(actnum_kji == 0)
    grid.create_xml()

    bundle.grid_uuid = grid.uuid
    bundle.crs_uuid = crs.uuid

    # Add ACTNUM & extra properties
    add_property(
        model,
        support_uuid=grid.uuid,
        values=actnum_kji.astype(np.int32, copy=False),
        title="ACTNUM",
        property_kind="discrete",
        indexable_element="cells",
    )
    for kw, arr in properties.items():
        arr = np.asarray(arr)
        knd, uom = infer_kind_uom(kw, arr)
        add_property(
            model,
            support_uuid=grid.uuid,
            values=arr,
            title=kw.upper(),
            property_kind=knd,
            indexable_element="cells",
            uom=uom,
        )

    # Persist to EPC/HDF5
    model.store_epc()
    return bundle


def add_property(
    model,
    support_uuid,
    values: np.ndarray,
    title: str,
    *,
    property_kind: Optional[str] = None,
    indexable_element: str = "cells",
    facet_type: Optional[str] = None,
    facet: Optional[str] = None,
    uom: Optional[str] = None,
    time_series_uuid: Optional[_uuid.UUID] = None,
    time_index: Optional[int] = None,
    null_value: Optional[int] = None,
    realization: Optional[int] = None,
    local_property_kind_uuid: Optional[_uuid.UUID] = None,
    string_lookup_uuid: Optional[_uuid.UUID] = None,
):
    """
    Create a RESQML Property (resqpy mode). In ETP mode we don't call this.

    NOTE: resqpy uses 'title' as the citation title; 'citation_title' is NOT a valid kwarg.
    """
    import resqpy.property as rqp

    if property_kind is None:
        property_kind = "continuous" if values.dtype.kind in ("f", "c") else "discrete"

    return rqp.Property.from_array(
        parent_model=model,
        support_uuid=support_uuid,
        values=values,
        title=title,  # citation title
        property_kind=property_kind,
        indexable_element=indexable_element,
        facet_type=facet_type,
        facet=facet,
        uom=uom,
        time_series_uuid=time_series_uuid,
        time_index=time_index,
        null_value=null_value,
        realization=realization,
        local_property_kind_uuid=local_property_kind_uuid,
        string_lookup_uuid=string_lookup_uuid,
    )


# ============================================================================
# Internal helpers: geometry assembly (cartesian & xtgeo)
# ============================================================================

def _normalize_array(val, shape):
    """Return an array broadcastable to `shape` from scalar or array-like."""
    a = np.asarray(val) if np.ndim(val) else np.array(val)
    return np.broadcast_to(a, shape) if a.shape != shape else a


def _edges_from_spacing(origin, spacing, axis, nk, nj, ni):
    """
    Compute edge coordinates along a given axis from origin & spacing.

    axis: 0->K, 1->J, 2->I
    spacing can be scalar, 1D, 2D, or 3D; we reshape/broadcast accordingly.

    Returns an edges array with shape:
      axis=2 (I): (nk, nj, ni+1)
      axis=1 (J): (nk, nj+1, ni)
      axis=0 (K): (nk+1, nj, ni)
    """
    # Target shapes
    if axis == 2:  # I
        line_shape = (nk, nj, ni)   # spacing per (k,j,i)
        org_shape  = (nk, nj)       # origin per (k,j)
        out_shape  = (nk, nj, ni+1) # edges
        cum_axis   = 2
    elif axis == 1:  # J
        line_shape = (nk, nj, ni)
        org_shape  = (nk, ni)
        out_shape  = (nk, nj+1, ni)
        cum_axis   = 1
    elif axis == 0:  # K
        line_shape = (nk, nj, ni)
        org_shape  = (nj, ni)
        out_shape  = (nk+1, nj, ni)
        cum_axis   = 0
    else:
        raise ValueError("axis must be 0 (K), 1 (J), or 2 (I)")

    s = np.asarray(spacing)

    # Accept common shapes and broadcast to line_shape
    if s.ndim == 0:
        s3 = np.full(line_shape, float(s))
    elif s.shape == (line_shape[cum_axis],):
        # 1D along the cumulative axis; broadcast across the others
        if axis == 2:      # I
            s3 = np.broadcast_to(s.reshape(1, 1, ni), line_shape)
        elif axis == 1:    # J
            s3 = np.broadcast_to(s.reshape(1, nj, 1), line_shape)
        else:              # K
            s3 = np.broadcast_to(s.reshape(nk, 1, 1), line_shape)
    elif axis == 2 and s.shape == (nj, ni):
        s3 = np.broadcast_to(s.reshape(1, nj, ni), line_shape)
    elif axis in (0, 1) and s.shape == (nk, nj):
        # for J/K, allow plane spacing (nk, nj)
        if axis == 1:
            s3 = np.broadcast_to(s.reshape(nk, nj, 1), line_shape)
        else:
            s3 = np.broadcast_to(s.reshape(nk, nj, 1), line_shape)
    elif s.shape == (nk, nj, ni):
        s3 = s
    else:
        # Try to infer: if total size matches, reshape to line_shape
        if s.size == nk * nj * ni:
            s3 = s.reshape(line_shape)
        else:
            raise ValueError(f"Unsupported spacing shape {s.shape} for axis {axis}")

    # Normalize origin
    o = np.asarray(origin)
    if o.ndim == 0:
        oN = np.full(org_shape, float(o))
    else:
        # Broadcast to org_shape
        oN = _normalize_array(o, org_shape)

    # Compute edges by cumulative sum along the axis
    if axis == 2:  # I
        out = np.concatenate([oN[..., None], oN[..., None] + np.cumsum(s3, axis=2)], axis=2)
    elif axis == 1:  # J
        out = np.concatenate([oN[:, None, :], oN[:, None, :] + np.cumsum(s3, axis=1)], axis=1)
    else:  # K
        out = np.concatenate([oN[None, ...], oN[None, ...] + np.cumsum(s3, axis=0)], axis=0)
    assert out.shape == out_shape
    return out


def _cartesian_corner_points_from_edges(xe, ye, ze):
    """
    xe: (nk, nj, ni+1)
    ye: (nk, nj+1, ni)
    ze: (nk+1, nj, ni)
    Returns cp: (nk, nj, ni, 2, 2, 2, 3)
    """
    nk, nj, ni1 = xe.shape
    nj1, ni2 = ye.shape[1], ye.shape[2]
    nk1, nj2, ni3 = ze.shape
    ni = ni1 - 1
    assert nj1 == nj + 1 and nk1 == nk + 1 and ni2 == ni and nj2 == nj and ni3 == ni

    cp = np.empty((nk, nj, ni, 2, 2, 2, 3), dtype=float)

    x0 = xe[:, :, 0:ni]     # (nk, nj, ni)
    x1 = xe[:, :, 1:ni+1]
    y0 = ye[:, 0:nj, :]     # (nk, nj, ni)
    y1 = ye[:, 1:nj+1, :]
    z0 = ze[0:nk, :, :]     # (nk, nj, ni)
    z1 = ze[1:nk+1, :, :]

    # kface, jface, iface in {0,1}
    cp[:, :, :, 0, 0, 0, 0] = x0; cp[:, :, :, 0, 0, 0, 1] = y0; cp[:, :, :, 0, 0, 0, 2] = z0
    cp[:, :, :, 0, 0, 1, 0] = x1; cp[:, :, :, 0, 0, 1, 1] = y0; cp[:, :, :, 0, 0, 1, 2] = z0
    cp[:, :, :, 0, 1, 0, 0] = x0; cp[:, :, :, 0, 1, 0, 1] = y1; cp[:, :, :, 0, 1, 0, 2] = z0
    cp[:, :, :, 0, 1, 1, 0] = x1; cp[:, :, :, 0, 1, 1, 1] = y1; cp[:, :, :, 0, 1, 1, 2] = z0

    cp[:, :, :, 1, 0, 0, 0] = x0; cp[:, :, :, 1, 0, 0, 1] = y0; cp[:, :, :, 1, 0, 0, 2] = z1
    cp[:, :, :, 1, 0, 1, 0] = x1; cp[:, :, :, 1, 0, 1, 1] = y0; cp[:, :, :, 1, 0, 1, 2] = z1
    cp[:, :, :, 1, 1, 0, 0] = x0; cp[:, :, :, 1, 1, 0, 1] = y1; cp[:, :, :, 1, 1, 0, 2] = z1
    cp[:, :, :, 1, 1, 1, 0] = x1; cp[:, :, :, 1, 1, 1, 1] = y1; cp[:, :, :, 1, 1, 1, 2] = z1
    return cp


def _get_corner_points_and_actnum(grid):
    """
    Returns (corner_points, actnum) for:
      - demo cartesian grid with scalar/1D/2D/3D spacings (dx, dy, dz)
      - xtgeo.Grid (if used)
    Falls back to unit edges if nothing found so the demo always runs.
    """
    nk, nj, ni = grid.nk, grid.nj, grid.ni

    def pick(obj, names, default=None):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                return v() if callable(v) else v
        return default

    # ---- Origins (accept scalar or arrays)
    origin = pick(grid, ["origin", "o", "org"], default=None)
    if origin is not None and len(origin) >= 3:
        ox, oy, oz = origin[0], origin[1], origin[2]
    else:
        ox = pick(grid, ["x0", "ox", "origin_x", "originx"], 0.0)
        oy = pick(grid, ["y0", "oy", "origin_y", "originy"], 0.0)
        oz = pick(grid, ["z0", "oz", "origin_z", "originz"], 0.0)

    # ---- Spacings (accept scalar/1D/2D/3D)
    spacing = pick(grid, ["spacing", "dxyz", "delta"], default=None)
    if spacing is not None and len(spacing) >= 3:
        dx, dy, dz = spacing[0], spacing[1], spacing[2]
    else:
        dx = pick(grid, ["dx", "delx"], 1.0)
        dy = pick(grid, ["dy", "dely"], 1.0)
        dz = pick(grid, ["dz", "delz"], 1.0)

    # Edges along each axis from (origin, spacing)
    # These handle scalar, 1D, 2D, 3D shapes
    xe = _edges_from_spacing(ox, dx, axis=2, nk=nk, nj=nj, ni=ni)      # (nk, nj, ni+1)
    ye = _edges_from_spacing(oy, dy, axis=1, nk=nk, nj=nj, ni=ni)      # (nk, nj+1, ni)
    ze = _edges_from_spacing(oz, dz, axis=0, nk=nk, nj=nj, ni=ni)      # (nk+1, nj, ni)

    # Corner points
    cp = _cartesian_corner_points_from_edges(xe, ye, ze)

    # ACTNUM: attribute or method; else ones
    act = pick(grid, ["actnum", "get_actnum"], default=None)
    if act is None:
        act = np.ones((nk, nj, ni), dtype=np.int32)
    else:
        act = np.asarray(act).reshape(nk, nj, ni).astype(np.int32, copy=False)
    return cp, act


# ============================================================================
# Internal helpers: ETP array registry
# ============================================================================

def _register_array(
    bundle: BuildBundle,
    *,
    uuid: _uuid.UUID,
    title: str,
    indexable: str,
    kind: str,
    array: np.ndarray,
    planned_path: str,
    uom: Optional[str],
    facet_type: Optional[str] = None,
    facet: Optional[str] = None,
):
    """Register an array for ETP streaming without writing HDF5."""
    binding = ArrayBinding(
        uuid=uuid,
        title=title,
        indexable=indexable,
        kind=kind,
        planned_hdf5_path=planned_path,
        shape=tuple(array.shape),
        dtype=str(array.dtype),
        uom=uom,
        facet_type=facet_type,
        facet=facet,
    )
    bundle.array_registry[uuid] = array
    bundle.array_meta[uuid] = binding
    if title not in bundle.property_index:
        bundle.property_index[title] = uuid


# ============================================================================
# Optional debug utility (call manually if needed)
# ============================================================================

def _debug_dump_grid_attrs(grid):
    keys = [k for k in dir(grid) if not k.startswith("_")]
    print("ATTRS:", keys)
    for k in (
        "nk", "nj", "ni", "geometry", "origin", "x0", "y0", "z0", "ox", "oy", "oz",
        "spacing", "dxyz", "dx", "dy", "dz", "actnum", "get_actnum"
    ):
        if hasattr(grid, k):
            v = getattr(grid, k)
            print(k, "=", v() if callable(v) else v)
