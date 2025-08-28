from __future__ import annotations

import logging
import uuid as _uuid
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from .grdecl_reader import GrdeclGrid
from .osdu_mapping import OSDU_PROPERTY_MAP

logger = logging.getLogger(__name__)

_NAMESPACE_SEED = _uuid.UUID("00000000-0000-0000-0000-0000000002a1")  # arbitrary constant seed for uuid5


@dataclass
class InMemoryResqml:
    model: "resqpy.model.Model"
    crs_uuid: _uuid.UUID
    grid_uuid: _uuid.UUID
    property_uuids: Dict[str, _uuid.UUID]
    xml_by_uuid: Dict[_uuid.UUID, bytes]  # serialized xml bytes for each object
    array_uris: Dict[str, Tuple[str, np.ndarray]]  # label -> (uuid-path-URI, np.ndarray)


# Alias to match the requested return annotation name
BuildBundle = InMemoryResqml


def uuid5(name: str) -> _uuid.UUID:
    return _uuid.uuid5(_NAMESPACE_SEED, name)


def build_in_memory(
    xtg_grid: GrdeclGrid,
    *,
    title_prefix: str = "ECL2RESQML",
    dataspace: str = "default",
    epsg: Optional[int] = None,
    mode: str = "etp",
    properties: Optional[Dict[str, np.ndarray]] = None,
) -> BuildBundle:
    """
    Build CRS, IJK grid and properties using resqpy with an in-memory HDF5 backend.
    Returns an InMemoryResqml bundle with xml and numpy arrays and URI suggestions for ETP PutDataArrays.

    Parameters
    ----------
    xtg_grid : GrdeclGrid
        Parsed grid structure from GRDECL reader.
    title_prefix : str
        Title prefix used to generate titles and stable UUID5 values.
    dataspace : str
        ETP dataspace name to embed in the suggested DataArray URIs.
    epsg : Optional[int]
        Optional EPSG code for CRS.
    mode : str
        Currently informational; 'etp' implies we prepare uuid-path URIs for arrays.
    properties : Optional[Dict[str, np.ndarray]]
        If provided, use this property dict instead of xtg_grid.properties.
    """
    import h5py
    import lxml.etree as ET
    import resqpy.model as rq
    import resqpy.crs as rqc
    import resqpy.grid as rqq
    import resqpy.property as rqp

    # Create a model with in-memory HDF5 file
    mem_h5 = h5py.File("inmem.h5", mode="w", driver="core", backing_store=False)
    model = rq.Model(new_epc=True, epc_file="inmem.epc")
    # Attach in-memory HDF5 handle
    model.h5_file = mem_h5  # type: ignore[attr-defined]

    # CRS: LocalDepth3dCrs (optionally set EPSG)
    crs_uuid = uuid5(f"{title_prefix}:crs")
    crs_kwargs = dict(
        model=model,
        title=f"{title_prefix} CRS",
        xy_units="m",
        z_units="m",
        z_inc_down=True,
        uuid=crs_uuid,
    )
    if epsg is not None:
        # resqpy supports epsg_code on Crs constructor
        crs_kwargs["epsg_code"] = int(epsg)
    crs = rqc.Crs(**crs_kwargs)
    crs.create_xml(add_as_part=True)

    # Grid
    grid_uuid = uuid5(f"{title_prefix}:ijkgrid")

    if xtg_grid.geometry_kind == "cartesian" and xtg_grid.dx is not None and xtg_grid.dy is not None and xtg_grid.dz is not None:
        reg = rqq.RegularGrid(
            parent_model=model,
            origin=(0.0, 0.0, 0.0 if xtg_grid.tops is None else float(xtg_grid.tops.min())),
            extent_kji=(xtg_grid.nk, xtg_grid.nj, xtg_grid.ni),
            dxyz=(float(xtg_grid.dx.mean()), float(xtg_grid.dy.mean()), float(xtg_grid.dz.mean())),
            crs_uuid=crs_uuid,
            set_points_cached=False,
            title=f"{title_prefix} IJK Grid",
            uuid=grid_uuid,
        )
        reg.write_hdf5()
        reg.create_xml(add_as_part=True)
        resq_grid = reg
    else:
        # Corner-point grid using COORD & ZCORN
        g = rqq.Grid.from_corner_points(
            parent_model=model,
            ni=xtg_grid.ni,
            nj=xtg_grid.nj,
            nk=xtg_grid.nk,
            zcorn=xtg_grid.zcorn.reshape((-1,)),  # resqpy expects shaped arrays
            coord=xtg_grid.coord.reshape((-1,)),
            crs_uuid=crs_uuid,
            title=f"{title_prefix} IJK Grid",
            uuid=grid_uuid,
        )
        g.write_hdf5()
        g.create_xml(add_as_part=True)
        resq_grid = g

    # Properties
    prop_dict: Dict[str, np.ndarray] = properties if properties is not None else dict(xtg_grid.properties)
    property_uuids: Dict[str, _uuid.UUID] = {}
    array_uris: Dict[str, Tuple[str, np.ndarray]] = {}

    for kw, arr in prop_dict.items():
        # Normalize array to C-order numpy array
        arr = np.asarray(arr)
        # Map to RESQML property kind & uom
        kind = OSDU_PROPERTY_MAP.get(kw, {}).get("resqml_property_kind", "unknown")
        uom = OSDU_PROPERTY_MAP.get(kw, {}).get("uom", "Euc")
        title = OSDU_PROPERTY_MAP.get(kw, {}).get("display", kw)

        puuid = uuid5(f"{title_prefix}:prop:{kw}")

        # Simple heuristic for discrete vs. continuous
        discrete = kw.upper() == "ACTNUM" or arr.dtype.kind in {"i", "u", "b"}

        p = rqp.Property(
            parent_model=model,
            support=resq_grid,
            property_kind=kind,
            indexable_element="cells",
            uom=(None if discrete else uom),
            discrete=discrete,
            title=title,
            uuid=puuid,
        )
        p.set_array_values(arr)
        p.write_hdf5()
        p.create_xml(add_as_part=True)

        property_uuids[kw] = puuid

        # Suggest a uuid-path DataArray URI under the property UUID (for ETP Protocol 9)
        if mode == "etp":
            da_uri = f"eml:///dataspace('{dataspace}')/uuid({puuid})/path(values)"
            array_uris[f"{kw}"] = (da_uri, arr)

    # Serialize XML for stored objects (for PutDataObjects payloads)
    xml_by_uuid: Dict[_uuid.UUID, bytes] = {}
    try:
        import lxml.etree as ET  # already imported, but guard for type-checkers
        for u in [crs_uuid, grid_uuid] + list(property_uuids.values()):
            root = model.root(uuid=u)
            xml_bytes = ET.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
            xml_by_uuid[u] = xml_bytes
    except Exception as exc:
        logger.exception("Failed to serialize XML: %s", exc)
        raise

    return InMemoryResqml(
        model=model,
        crs_uuid=crs_uuid,
        grid_uuid=grid_uuid,
        property_uuids=property_uuids,
        xml_by_uuid=xml_by_uuid,
        array_uris=array_uris,
    )
