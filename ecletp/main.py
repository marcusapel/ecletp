from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .grdecl_reader import read_grdecl
from .resqml_builder import build_in_memory
from . import config_etp as conf  # use in-repo configuration module
from .etp_client import (
    EtpConfig,
    connect_etp,
    ensure_transaction,
    commit_and_close,
    put_objects,
    put_data_arrays,
    CONTENT_TYPE_EML_CRS,
    CONTENT_TYPE_RESQML_GRID,
    CONTENT_TYPE_RESQML_PROPERTY,
    CONTENT_TYPE_RESQML_DISCRETE,
)

LOG = logging.getLogger("ecl2resqml_etp")


def build_and_publish(args) -> None:
    logging.basicConfig(
        level=(logging.DEBUG if args.log else logging.INFO),
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    # Parse GRDECL (or demo grid if --eclgrd omitted)
    grid, meta = read_grdecl(args.eclgrd)
    LOG.info(
        "Parsed grid: %dx%dx%d, geometry=%s via %s",
        grid.ni,
        grid.nj,
        grid.nk,
        grid.geometry_kind,
        meta.get("parser"),
    )

    title_prefix = args.title or (Path(args.eclgrd).stem if args.eclgrd else "demo")

    # Build in-memory RESQML + arrays using the canonical kwargs
    bundle = build_in_memory(
        title_prefix=title_prefix,
        dataspace=args.dataspace,
        epsg=args.epsg,
        mode="etp",
        properties=None,  # pass external properties dict here if needed
    )

    # Prepare PutDataObjects payloads
    ds_uri = args.dataspace

    def uri_for(u):
        return f"eml:///dataspace('{ds_uri}')/uuid({u})"

    objects = []

    # CRS
    objects.append((uri_for(bundle.crs_uuid), bundle.xml_by_uuid[bundle.crs_uuid], CONTENT_TYPE_EML_CRS))
    # Grid
    objects.append((uri_for(bundle.grid_uuid), bundle.xml_by_uuid[bundle.grid_uuid], CONTENT_TYPE_RESQML_GRID))
    # Properties (discrete/continuous)
    for kw, puuid in bundle.property_uuids.items():
        ctype = CONTENT_TYPE_RESQML_DISCRETE if kw.upper() == "ACTNUM" else CONTENT_TYPE_RESQML_PROPERTY
        objects.append((uri_for(puuid), bundle.xml_by_uuid[puuid], ctype))

    # Connect to ETP and publish (using in-repo config_etp.py)
    cfg = EtpConfig(
        ws_uri=conf.WS_URI,
        dataspace=args.dataspace,
        app_name=args.app_name,
        auth_token=getattr(conf, "AUTH_TOKEN", None),
    )

    client = connect_etp(cfg)
    try:
        ensure_transaction(client)
        put_objects(client, args.dataspace, objects)
        # Arrays via Protocol 9
        put_data_arrays(client, bundle.array_uris)
    finally:
        commit_and_close(client)


def cli():
    p = argparse.ArgumentParser(description="GRDECL -> RESQML -> ETP publisher (RESQML 2.0.1, ETP 1.2)")
    p.add_argument("--eclgrd", help="Path to Eclipse GRDECL .grdecl; if omitted, uses demo 2x2x3 grid", default=None)
    p.add_argument("--dataspace", help="ETP dataspace, e.g., maap/m25test", default="maap/m25test")
    p.add_argument("--app-name", default="ecl2resqml_etp")
    p.add_argument("--title", help="Title prefix for objects", default=None)
    p.add_argument("--epsg", type=int, default=None, help="Optional EPSG code for CRS (e.g., 32631)")
    p.add_argument("--log", action="store_true", help="Enable debug logging")
    args = p.parse_args()
    build_and_publish(args)


if __name__ == "__main__":
    cli()
