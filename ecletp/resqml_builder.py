#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a RESQML model from an Eclipse corner-point grid and properties using resqpy.

Features:
- Creates EPC/HDF5 model and CRS
- Imports corner-point grid (7D array)
- Adds ACTNUM and other properties with correct metadata
- Supports categorical properties via StringLookup
- Optional GRDECL import using xtgeo

Author: Marcus Apel (Equinor) 
"""

import os
import numpy as np
import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.rq_import as rqimp
import resqpy.property as rqp

try:
    import xtgeo
    _HAS_XTGEO = True
except ImportError:
    _HAS_XTGEO = False


# ---------------- CRS ----------------
def create_crs(model, xy_units="m", z_units="m", z_inc_down=True,
               epsg_code=None, rotation=0.0, offsets=(0.0, 0.0, 0.0)):
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
        title="LocalDepth3dCrs"
    )
    crs.create_xml(add_as_part=True)
    return crs


# ---------------- Property Mapping ----------------
ECL_TO_RESQML = {
    "PORO": {"kind": "porosity", "uom": None},
    "NTG": {"kind": "net to gross ratio", "uom": None},
    "PERMX": {"kind": "rock permeability", "uom": "mD", "facet_type": "direction", "facet": "I"},
    "PERMY": {"kind": "rock permeability", "uom": "mD", "facet_type": "direction", "facet": "J"},
    "PERMZ": {"kind": "rock permeability", "uom": "mD", "facet_type": "direction", "facet": "K"},
    "ACTNUM": {"kind": "discrete", "uom": None}
}


def add_property(model, grid_uuid, array, keyword, string_lookup_uuid=None):
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
        citation_title=keyword.upper()
    )
    prop.write_hdf5()
    prop.create_xml(add_as_part=True)
    return prop


# ---------------- Main Builder ----------------
def build_from_corner_points(epc_path, cp_kjinxyz, actnum=None, properties=None,
                              epsg_code=None, facies_lookup=None):
    model = rq.Model(epc_file=epc_path, new_epc=True, create_basics=True, create_hdf5_ext=True)
    crs = create_crs(model, epsg_code=epsg_code)

    # Grid
    grid = rqimp.grid_from_cp(model, cp_kjinxyz, crs_uuid=crs.uuid, treat_as_nan=np.nan)
    grid.write_hdf5()
    grid.create_xml(add_as_part=True)

    # ACTNUM
    if actnum is not None:
        add_property(model, grid.uuid, actnum.astype(np.int32), "ACTNUM")

    # FACIES lookup
    facies_lookup_uuid = None
    if facies_lookup:
        sl = rqp.StringLookup(model, int_to_str_dict=facies_lookup, title="FACIES_LOOKUP")
        sl.create_xml(add_as_part=True)
        facies_lookup_uuid = sl.uuid

    # Other properties
    if properties:
        for kw, arr in properties.items():
            add_property(model, grid.uuid, arr, kw, string_lookup_uuid=facies_lookup_uuid if kw.upper() == "FACIES" else None)

    model.store_epc()
    return model


# ---------------- GRDECL Import ----------------
def build_from_grdecl(grdecl_path, epc_path, epsg_code=None):
    if not _HAS_XTGEO:
        raise ImportError("xtgeo is required for GRDECL import.")
    grid = xtgeo.grid_from_file(grdecl_path)
    cp = grid.corner_points()
    act = getattr(grid, "actnum", None)
    props = {p.name.upper(): np.asarray(p.values).reshape(grid.nk, grid.nj, grid.ni) for p in grid.properties}
    return build_from_corner_points(epc_path, cp, actnum=act, properties=props, epsg_code=epsg_code)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build RESQML EPC from Eclipse corner-point grid.")
    ap.add_argument("epc", help="Output EPC file path")
    ap.add_argument("--grdecl", help="Input GRDECL file (requires xtgeo)")
    ap.add_argument("--epsg", help="EPSG code", default=None)
    args = ap.parse_args()

    if args.grdecl:
        build_from_grdecl(args.grdecl, args.epc, epsg_code=args.epsg)
    else:
        print("Provide --grdecl or call build_from_corner_points() in your code.")

