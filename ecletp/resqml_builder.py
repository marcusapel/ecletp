# ecletp/resqml_builder.py
from __future__ import annotations

import logging
import uuid as _uuid
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from .grdecl_reader import GrdeclGrid
from .osdu_mapping import OSDU_PROPERTY_MAP

logger = logging.getLogger(__name__)

# Deterministic UUID namespace for this module
_NAMESPACE_SEED = _uuid.UUID("00000000-0000-0000-0000-0000000002a1")


@dataclass
class InMemoryResqml:
    model: "resqpy.model.Model"
    crs_uuid: _uuid.UUID
    grid_uuid: _uuid.UUID
    property_uuids: Dict[str, _uuid.UUID]
    xml_by_uuid: Dict[_uuid.UUID, bytes]
    array_uris: Dict[str, Tuple[str, np.ndarray]]


def uuid5(name: str) -> _uuid.UUID:
    return _uuid.uuid5(_NAMESPACE_SEED, name)


def build_in_memory(
    grid: GrdeclGrid,
    title_prefix: str,
    dataspace: str = "maap/m25test",
    epsg: str = "2334",
) -> InMemoryResqml:
    """
    Build CRS, IJK grid and properties using resqpy with an in-memory HDF5 backend.

    Notes:
      • Properties are created with Property.from_array(..., support_uuid=...), not the constructor.  [resqpy docs]
      • For grid cell arrays use indexable_element='cells'; uom for continuous only.                 [resqpy docs]
      • RegularGrid: explicit geometry is not written by default; see docs for set_points_cached.    [resqpy docs]
      • Corner-point grids: use resqpy.rq_import.grid_from_cp(...) (expects Nexus-style cp_array).    [resqpy docs]
    """
    import h5py
    import lxml.etree as ET
    import resqpy.model as rq
    import resqpy.crs as rqc
    import resqpy.grid as rqq
    import resqpy.property as rqp
    import resqpy.rq_import as rqi

    # In-memory HDF5 bound to the model
    mem_h5 = h5py.File("inmem.h5", mode="w", driver="core", backing_store=False)
    model = rq.Model(new_epc=True, epc_file="inmem.epc")
    model.h5_file = mem_h5  # type: ignore[attr-defined]

    # CRS
    crs_uuid = uuid5(f"{title_prefix}:crs")
    crs = rqc.Crs(
        model,
        title=f"{title_prefix} CRS",
        xy_units="m",
        z_units="m",
        z_inc_down=True,
        uuid=crs_uuid,
        epsg_code=str(epsg)
    )
    crs.create_xml(add_as_part=True)

    # Grid
    grid_uuid = uuid5(f"{title_prefix}:ijkgrid")

    if grid.geometry_kind == "cartesian" and grid.dx is not None and grid.dy is not None and grid.dz is not None:
        # RegularGrid (documented signature; explicit points not written unless requested)
        reg = rqq.RegularGrid(
            parent_model=model,
            origin=(0.0, 0.0, 0.0 if grid.tops is None else float(grid.tops.min())),
            extent_kji=(grid.nk, grid.nj, grid.ni),
            dxyz=(float(grid.dx.mean()), float(grid.dy.mean()), float(grid.dz.mean())),
            crs_uuid=crs_uuid,
            set_points_cached=False,
            title=f"{title_prefix} IJK Grid",
            uuid=grid_uuid,
        )
        reg.write_hdf5()
        reg.create_xml(add_as_part=True)
        resq_grid = reg
    elif grid.geometry_kind == "corner-point":
        # We do NOT call a non-existent Grid.from_corner_points.
        # The documented creation route is rq_import.grid_from_cp(...), which expects a Nexus cp_array.
        # Ref: https://resqpy.readthedocs.io/en/latest/_autosummary/resqpy.rq_import.grid_from_cp.html
        raise NotImplementedError(
            "Corner-point grid creation requires a Nexus-style cp_array via "
            "resqpy.rq_import.grid_from_cp(...). Your GRDECL COORD/ZCORN must be converted "
            "to the expected cp_array beforehand (no guessing here). See resqpy docs."
        )
        # Example once cp_array is provided (pseudocode; do not enable without verified adapter):
        # cp_array = build_cp_array_from_coord_zcorn(grid.coord, grid.zcorn)  # verified adapter required
        # g = rqi.grid_from_cp(parent_model=model, cp_array=cp_array, crs_uuid=crs_uuid, title=f"{title_prefix} IJK Grid", grid_uuid=grid_uuid)
        # g.write_hdf5(); g.create_xml(add_as_part=True)
        # resq_grid = g
    else:
        raise ValueError(f"Unknown geometry_kind: {grid.geometry_kind}")

    # Properties
    property_uuids: Dict[str, _uuid.UUID] = {}
    array_uris: Dict[str, Tuple[str, np.ndarray]] = {}

    for kw, arr in grid.properties.items():
        np_arr = np.asarray(arr)
        kind = OSDU_PROPERTY_MAP.get(kw, {}).get("resqml_property_kind", "unknown")
        uom = OSDU_PROPERTY_MAP.get(kw, {}).get("uom", "Euc")
        # discrete if ACTNUM or integer dtype; omit uom for discrete properties
        is_discrete = (kw.upper() == "ACTNUM") or np.issubdtype(np_arr.dtype, np.integer)
        uom_for_resqpy: Optional[str] = None if is_discrete else uom

        prop = rqp.Property.from_array(
            parent_model=model,
            cached_array=np_arr,
            source_info=f"{kw} from GRDECL",
            keyword=kw,
            support_uuid=resq_grid.uuid,  # correct linking field
            property_kind=kind,
            indexable_element="cells",
            uom=uom_for_resqpy,
        )
        property_uuids[kw] = prop.uuid

        da_uri = f"eml:///dataspace('{dataspace}')/uuid({property_uuids[kw]})/path(values)"
        array_uris[kw] = (da_uri, np_arr)

        logger.debug(
            "Published property %s (uuid=%s, kind=%s, uom=%s, discrete=%s)",
            kw, str(prop.uuid), kind, uom_for_resqpy, is_discrete
        )

    # Serialize XML blobs (for ETP PutDataObjects)
    xml_by_uuid: Dict[_uuid.UUID, bytes] = {}
    for u in [crs_uuid, grid_uuid] + list(property_uuids.values()):
        root = model.root(uuid=u)
        xml_bytes = ET.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        xml_by_uuid[u] = xml_bytes

    return InMemoryResqml(
        model=model,
        crs_uuid=crs_uuid,
        grid_uuid=grid_uuid,
        property_uuids=property_uuids,
        xml_by_uuid=xml_by_uuid,
        array_uris=array_uris,
    )
