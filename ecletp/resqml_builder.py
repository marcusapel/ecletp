# resqml_builder.py
from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import xtgeo

# resqpy imports are optional in ETP mode (we import lazily)
# import resqpy.model as rq
# import resqpy.crs as rqc
# import resqpy.grid as rqq
# import resqpy.rq_import as rqi
# import resqpy.property as rqp

# ----------------------------
# Helpers & configuration
# ----------------------------

# Stable namespace for deterministic UUIDs (change if you want different stable IDs across envs)
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

# Minimal mapping from Eclipse keywords to RESQML property kinds & units
# Extend as needed
ECL_TO_RESQML = {
    "ACTNUM": ("discrete", None),
    "PORO": ("porosity", None),
    "PERMX": ("permeability rock", "m2"),  # RESQML kind name; uom in SI (m2); convert if needed
    "PERMY": ("permeability rock", "m2"),
    "PERMZ": ("permeability rock", "m2"),
    "NTG": ("net to gross ratio", None),
    "SATNUM": ("discrete", None),
    "PVTNUM": ("discrete", None),
}

def infer_kind_uom(keyword: str, values: np.ndarray) -> Tuple[str, Optional[str]]:
    kind, uom = ECL_TO_RESQML.get(keyword.upper(), (None, None))
    if kind is None:
        # fallback: continuous vs discrete by dtype
        kind = "continuous" if values.dtype.kind in ("f", "c") else "discrete"
    return kind, uom

# ----------------------------
# Public API
# ----------------------------

def build_in_memory(
    xtg_grid: xtgeo.Grid,
    *,
    title_prefix: str = "ECL2RESQML",
    dataspace: str = "default",
    epsg: Optional[int] = None,
    mode: str = "etp",  # "etp" (no disk write) or "resqpy" (canonical EPC+HDF5)
    properties: Optional[Dict[str, np.ndarray]] = None,  # extra properties by keyword
) -> BuildBundle:
    """
    Entry point. Builds a grid + properties mapping ready for either ETP streaming ('etp' mode)
    or a canonical RESQML dataset ('resqpy' mode).
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
    xtg_grid: xtgeo.Grid,
    *,
    bundle: BuildBundle,
    title_prefix: str,
    dataspace: str,
    epsg: Optional[int],
    properties: Dict[str, np.ndarray],
) -> BuildBundle:
    """Build from an XTGeo corner-point grid."""
    nk, nj, ni = xtg_grid.nk, xtg_grid.nj, xtg_grid.ni
    # Corner points: (nk, nj, ni, 2, 2, 2, 3)
    cp = xtg_grid.get_corner_points()

    # ACTNUM (bool) → int32 per cell
    act = xtg_grid.get_actnum()
    if act is None:
        act = np.ones((nk, nj, ni), dtype=np.int32)
    else:
        act = act.astype(np.int32, copy=False)

    # Deterministic grid & CRS UUIDs
    grid_title = f"{title_prefix}_Grid_{nk}x{nj}x{ni}"
    grid_uuid = stable_uuid(dataspace, "grid", grid_title)
    crs_uuid = stable_uuid(dataspace, "crs", str(epsg or "unknown"))
    bundle.grid_uuid = grid_uuid
    bundle.crs_uuid = crs_uuid

    # Prepare planned HDF5 dataset paths (used both by resqpy & ETP registry)
    points_path = f"/RESQML/IjkGridRepresentation/{grid_uuid}/Points"
    actnum_path = f"/RESQML/GridProperty/{grid_uuid}/ACTNUM"

    if bundle.mode == "etp":
        # --- ETP mode: do NOT write EPC/HDF5. Keep arrays in memory & record metadata.

        # Flatten points to (n_cells, 8, 3) if needed by your downstream
        cp_reg = cp  # keep original shape; ETP client can reshape
        _register_array(
            bundle,
            uuid=stable_uuid(dataspace, "grid", "points", str(grid_uuid)),
            title=f"{grid_title}_Points",
            indexable="cells",
            kind="points",
            array=cp_reg,
            planned_path=points_path,
            uom=None,
        )

        _register_array(
            bundle,
            uuid=stable_uuid(dataspace, "grid", "ACTNUM", str(grid_uuid)),
            title="ACTNUM",
            indexable="cells",
            kind="discrete",
            array=act,
            planned_path=actnum_path,
            uom=None,
        )

        # Extra cell properties
        for kw, arr in properties.items():
            kind, uom = infer_kind_uom(kw, arr)
            path = f"/RESQML/GridProperty/{grid_uuid}/{kw.upper()}"
            _register_array(
                bundle,
                uuid=stable_uuid(dataspace, "grid", kw.upper(), str(grid_uuid)),
                title=kw.upper(),
                indexable="cells",
                kind=kind,
                array=np.asarray(arr),
                planned_path=path,
                uom=uom,
            )

        return bundle
    # --- resqpy mode: build a canonical RESQML dataset (writes HDF5)
    import resqpy.model as rq
    import resqpy.crs as rqc
    import resqpy.rq_import as rqi
    import resqpy.property as rqp

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
    crs.create_xml()  # writes CRS xml

    # Build grid from corner points
    # resqpy.rq_import.grid_from_cp handles CP geometry
    grid = rqi.grid_from_cp(
        cp_array=cp,
        model=model,
        crs_uuid=crs.uuid,
        title=grid_title,
        max_z_void=0.1,  # typical default (introduces K-gaps if large voids)
    )
    # ACTNUM: set inactive mask on grid
    grid.set_inactive(act == 0)
    grid.create_xml()  # writes grid xml + geometry arrays to HDF5

    bundle.grid_uuid = grid.uuid
    bundle.crs_uuid = crs.uuid

    # Add ACTNUM as property
    add_property(
        model,
        support_uuid=grid.uuid,
        values=act.astype(np.int32, copy=False),
        title="ACTNUM",
        property_kind="discrete",
        indexable_element="cells",
    )

    # Add extra cell properties
    for kw, arr in properties.items():
        knd, uom = infer_kind_uom(kw, arr)
        add_property(
            model,
            support_uuid=grid.uuid,
            values=np.asarray(arr),
            title=kw.upper(),
            property_kind=knd,
            indexable_element="cells",
            uom=uom,
        )

    # Persist to EPC/HDF5 on disk
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
    """Create a RESQML Property (resqpy mode). In ETP mode we don't call this."""
    import resqpy.property as rqp

    if property_kind is None:
        property_kind = "continuous" if values.dtype.kind in ("f", "c") else "discrete"

    return rqp.Property.from_array(
        parent_model=model,
        support_uuid=support_uuid,
        values=values,
        title=title,  # <— 'title' is the citation title in resqpy API
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

# ----------------------------
# Internal helpers
# ----------------------------

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
