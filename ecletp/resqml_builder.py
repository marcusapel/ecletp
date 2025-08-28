
from __future__ import annotations
import io
import logging
import uuid as _uuid
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from .grdecl_reader import GrdeclGrid
from .osdu_mapping import OSDU_PROPERTY_MAP

logger = logging.getLogger(__name__)

_NAMESPACE_SEED = _uuid.UUID('00000000-0000-0000-0000-0000000002a1')  # arbitrary constant seed for uuid5

@dataclass
class InMemoryResqml:
    model: 'resqpy.model.Model'
    crs_uuid: _uuid.UUID
    grid_uuid: _uuid.UUID
    property_uuids: Dict[str, _uuid.UUID]
    xml_by_uuid: Dict[_uuid.UUID, bytes]  # serialized xml bytes for each object
    array_uris: Dict[str, Tuple[str, np.ndarray]]  # label -> (uuid-path-URI, np.ndarray)


def uuid5(name: str) -> _uuid.UUID:
    return _uuid.uuid5(_NAMESPACE_SEED, name)


def build_in_memory(grid: GrdeclGrid, title_prefix: str, dataspace: str = "maap/m25test") -> InMemoryResqml:
    """Build CRS, IJK grid and properties using resqpy with an in-memory HDF5 backend.

    Returns an InMemoryResqml bundle with xml and numpy arrays and URI suggestions for ETP PutDataArrays.
    """
    import h5py
    import lxml.etree as ET
    import resqpy.model as rqm
    import resqpy.crs as rqc
    import resqpy.grid as rqq
    import resqpy.property as rqp

def build_in_memory(grid, title_prefix='model', dataspace=''):
    # Create a new in-memory model
    model = rqm.Model(new_epc_file='inmem.epc')

    # Define default CRS using EPSG:2334
    crs = rqc.Crs(
        model=model,
        title='EPSG:2334 CRS',
        origin=(0.0, 0.0, 0.0),
        rotation=0.0,
        z_inc_down=True,
        units='m',
        epsg_code=2334
    )
    crs.create_xml()

    # Grid
    grid_uuid = uuid5(f"{title_prefix}:ijkgrid")
    if grid.geometry_kind == 'cartesian' and grid.dx is not None and grid.dy is not None and grid.dz is not None:
        reg = rqq.RegularGrid(
            parent_model=model,
            origin=(0.0, 0.0, 0.0 if grid.tops is None else float(grid.tops.min())),
            extent_kji=(grid.nk, grid.nj, grid.ni),
            dxyz=(float(grid.dx.mean()), float(grid.dy.mean()), float(grid.dz.mean())),
            crs_uuid=crs_uuid,
            set_points_cached=False,
            title=f"{title_prefix} IJK Grid",
            uuid=grid_uuid
        )
        reg.write_hdf5()
        reg.create_xml(add_as_part=True)
        resq_grid = reg
    else:
        # Corner point grid using COORD & ZCORN
        g = rqq.Grid.from_corner_points(
            parent_model=model,
            ni=grid.ni, nj=grid.nj, nk=grid.nk,
            zcorn=grid.zcorn.reshape((-1,)),  # resqpy expects shaped arrays; from GRDECL layout
            coord=grid.coord.reshape((-1,)),
            crs_uuid=crs_uuid,
            title=f"{title_prefix} IJK Grid",
            uuid=grid_uuid
        )
        g.write_hdf5()
        g.create_xml(add_as_part=True)
        resq_grid = g

    # ACTNUM as discrete/categorical property
    property_uuids: Dict[str, _uuid.UUID] = {}
    array_uris: Dict[str, Tuple[str, np.ndarray]] = {}

    for kw, arr in grid.properties.items():
        kind = OSDU_PROPERTY_MAP.get(kw, {}).get('resqml_property_kind', 'unknown')
        uom = OSDU_PROPERTY_MAP.get(kw, {}).get('uom', 'Euc')
        title = OSDU_PROPERTY_MAP.get(kw, {}).get('display', kw)
        puuid = uuid5(f"{title_prefix}:prop:{kw}")
        discrete = True if kw.upper() == 'ACTNUM' or arr.dtype in (np.int32, np.int64, np.uint8) else False
        p = rqp.Property(
            parent_model=model,
            support=resq_grid,
            property_kind=kind,
            indexable_element='cells',
            uom=(uom if not discrete else None),
            discrete=discrete,
            title=title,
            uuid=puuid
        )
        p.set_array_values(arr)
        p.write_hdf5()
        p.create_xml(add_as_part=True)
        property_uuids[kw] = puuid
        # Suggest a uuid-path DataArray URI under the property UUID
        da_uri = f"eml:///dataspace('{dataspace}')/uuid({puuid})/path(values)"
        array_uris[f"{kw}"] = (da_uri, arr)

    # Serialize XML for objects (for PutDataObjects payloads)
    xml_by_uuid: Dict[_uuid.UUID, bytes] = {}
    for u in [crs_uuid, grid_uuid] + list(property_uuids.values()):
        root = model.root(uuid=u)
        xml_bytes = ET.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        xml_by_uuid[u] = xml_bytes

    return InMemoryResqml(model=model, crs_uuid=crs_uuid, grid_uuid=grid_uuid, property_uuids=property_uuids, xml_by_uuid=xml_by_uuid, array_uris=array_uris)
