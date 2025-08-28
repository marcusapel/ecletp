
from __future__ import annotations
import gzip
import io
import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CONTENT_TYPE_RESQML_GRID = 'application/x-resqml+xml;version=2.0;type=resqml20.IjkGridRepresentation'
CONTENT_TYPE_EML_CRS = 'application/x-eml+xml;version=2.0;type=eml20.LocalDepth3dCrs'
CONTENT_TYPE_RESQML_PROPERTY = 'application/x-resqml+xml;version=2.0;type=resqml20.ContinuousProperty'
CONTENT_TYPE_RESQML_DISCRETE = 'application/x-resqml+xml;version=2.0;type=resqml20.DiscreteProperty'

@dataclass
class EtpConfig:
    ws_uri: str
    dataspace: str
    app_name: str = 'ecl2etp'
    auth_token: str | None = None


def connect_etp(cfg: EtpConfig):
    """Open a pyetp connection and return the client/session object."""
    from pyetp.client import connect
    headers = {
        'dataspace': cfg.dataspace,
        'ApplicationName': cfg.app_name,
    }
    if cfg.auth_token:
        headers['Authorization'] = cfg.auth_token  # do NOT log token
    logger.info("Connecting ETP: ws_uri=%s, headers={dataspace:%s, ApplicationName:%s}", cfg.ws_uri, cfg.dataspace, cfg.app_name)
    client = connect(cfg.ws_uri, headers=headers)
    return client


def ensure_transaction(client) -> str:
    """Start or reuse an ETP Protocol 18 transaction and return its transaction UUID string."""
    # pyetp exposes protocol handlers on client.protocol[18]
    tx = client.protocol[18]
    existing = getattr(client, 'transaction_id', None)
    if existing:
        logger.debug("Reusing existing transaction %s", existing)
        return existing
    resp = tx.start_transaction()
    tid = resp.transactionUuid
    logger.info("Started ETP transaction %s", tid)
    client.transaction_id = tid
    return tid


def commit_and_close(client):
    tx = client.protocol[18]
    tid = getattr(client, 'transaction_id', None)
    if tid:
        tx.commit_transaction(transactionUuid=tid)
        logger.info("Committed transaction %s", tid)
    client.close()


def put_objects(client, dataspace: str, items: Iterable[Tuple[str, bytes, str]]):
    """Put RESQML/EML XML objects with Protocol 3.

    items: iterable of (uri, xml_bytes, content_type)
    """
    from etptypes.energistics.etp.v12.datatypes.object.data_object import DataObject
    from etptypes.energistics.etp.v12.datatypes.object.data_object import DataObjectFormat as _DOF

    p3 = client.protocol[3]
    data_objects = []
    for uri, xml_bytes, content_type in items:
        data_objects.append(DataObject(resource={'uri': uri, 'contentType': content_type}, data=xml_bytes, format=_DOF.xml))
        logger.debug("Queue PutDataObjects: %s", uri)
    logger.info("PutDataObjects count=%d", len(data_objects))
    p3.put_data_objects(data_objects=data_objects)


def put_data_arrays(client, arrays: Dict[str, Tuple[str, np.ndarray]], gzip_over_bytes: int = 4*1024*1024, max_msg_bytes: int = 1024*1024):
    """Put arrays via Protocol 9 DataArray (chunked + gzip for large pieces).

    arrays: label -> (uuid-path-URI, np.ndarray)
    """
    p9 = client.protocol[9]
    for label, (uri, arr) in arrays.items():
        data = arr.tobytes(order='C')
        total = len(data)
        logger.info("PutDataArrays %s -> %s size=%d bytes", label, uri, total)
        if total <= 1024:
            # small enough to inline as one chunk
            p9.put_data_arrays(data_arrays=[{'uri': uri, 'data': data, 'dimensions': list(arr.shape)}])
            continue
        # chunking
        chunk_size = max_msg_bytes
        n_chunks = math.ceil(total / chunk_size)
        offset = 0
        for i in range(n_chunks):
            chunk = data[offset: offset+chunk_size]
            offset += len(chunk)
            if len(chunk) >= gzip_over_bytes:
                chunk = gzip.compress(chunk)
                compressed = True
            else:
                compressed = False
            p9.put_data_arrays(data_arrays=[{
                'uri': uri,
                'data': chunk,
                'dimensions': list(arr.shape),
                'compressed': compressed
            }])
