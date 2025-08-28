
from __future__ import annotations
import argparse
import importlib.util
import logging
from pathlib import Path

from .grdecl_reader import read_grdecl
from .resqml_builder import build_in_memory
from .etp_client import EtpConfig, connect_etp, ensure_transaction, commit_and_close, put_objects, CONTENT_TYPE_EML_CRS, CONTENT_TYPE_RESQML_GRID, CONTENT_TYPE_RESQML_PROPERTY, CONTENT_TYPE_RESQML_DISCRETE, put_data_arrays

LOG = logging.getLogger("ecl2resqml_etp")


def load_confid(conf_path: str):
    spec = importlib.util.spec_from_file_location("confid_etp", conf_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def build_and_publish(args):
    logging.basicConfig(level=(logging.DEBUG if args.log else logging.INFO), format='[%(levelname)s] %(name)s: %(message)s')

    grid, meta = read_grdecl(args.eclgrd)
    LOG.info("Parsed grid: %dx%dx%d, geometry=%s via %s", grid.ni, grid.nj, grid.nk, grid.geometry_kind, meta.get('parser'))

    title_prefix = args.title or (Path(args.eclgrd).stem if args.eclgrd else 'demo')
    bundle = build_in_memory(grid, title_prefix=title_prefix, dataspace=args.dataspace)

    # Prepare PutDataObjects
    ds_uri = args.dataspace
    def uri_for(uuid):
        return f"eml:///dataspace('{ds_uri}')/uuid({uuid})"

    objects = []
    # CRS first
    objects.append((uri_for(bundle.crs_uuid), bundle.xml_by_uuid[bundle.crs_uuid], CONTENT_TYPE_EML_CRS))
    # Grid then
    objects.append((uri_for(bundle.grid_uuid), bundle.xml_by_uuid[bundle.grid_uuid], CONTENT_TYPE_RESQML_GRID))
    # Properties (discrete/continuous)
    for kw, puuid in bundle.property_uuids.items():
        ct = CONTENT_TYPE_RESQML_DISCRETE if kw.upper()== 'ACTNUM' else CONTENT_TYPE_RESQML_PROPERTY
        objects.append((uri_for(puuid), bundle.xml_by_uuid[puuid], ct))

    # Connect ETP and publish
    conf = load_confid(args.confid)
    cfg = EtpConfig(ws_uri=conf.WS_URI, dataspace=args.dataspace, app_name=args.app_name, auth_token=getattr(conf, 'AUTH_TOKEN', None))
    client = connect_etp(cfg)
    try:
        ensure_transaction(client)
        put_objects(client, args.dataspace, objects)
        # Arrays via Protocol 9
        put_data_arrays(client, bundle.array_uris)
    finally:
        commit_and_close(client)


def cli():
    p = argparse.ArgumentParser(description='GRDECL -> RESQML -> ETP publisher (RESQML 2.0.1, ETP 1.2)')
    p.add_argument('--eclgrd', help='Path to Eclipse GRDECL .grdecl; if omitted, uses demo 2x2x3 grid', default=None)
    p.add_argument('--dataspace', help="ETP dataspace, e.g., maap/m25test", default='maap/m25test')
    p.add_argument('--config', help='Path to config_etp.py (connection, token)', required=False)
    p.add_argument('--app-name', default='ecl2resqml_etp')
    p.add_argument('--title', help='Title prefix for objects', default=None)
    p.add_argument('--log', action='store_true', help='Enable debug logging')
    args = p.parse_args()
    build_and_publish(args)

if __name__ == '__main__':
    cli()
