
# ecl2resqml_etp

Read an Eclipse GRDECL grid, build RESQML 2.0.1 objects fully in memory using `resqpy`, and publish them to an ETP 1.2 RDDMS store using `pyetp`.

## Features
- GRDECL reader using Equinor **resdata** when available; fallback lightweight parser.
- Builds **LocalDepth3dCrs**, **IjkGridRepresentation**, and **Property** objects (continuous or discrete) with deterministic UUIDv5.
- Stores arrays in an in-memory HDF5 file (h5py `driver='core', backing_store=False`).
- Publishes XML via Protocol 3 **PutDataObjects** and arrays via Protocol 9 **PutDataArrays** with chunking and gzip (>4 MiB).
- Manages Protocol 18 **Transaction** (start/commit) and logs URIs and content-types (never logs tokens).

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python -m ecl2resqml_etp.main   --eclgrd /path/to/model.GRDECL   --dataspace maap/m25test   --confid /path/to/confid_etp.py   --title SPE10B --log
```

`confid_etp.py` must define at least `WS_URI` and optionally `AUTH_TOKEN`.

Example template:
```python
# confid_etp.py (template)
WS_URI = 'wss://rddms.example.com/etp'
AUTH_TOKEN = 'Bearer eyJ...'
```

## Content types and URIs
- CRS: `application/x-eml+xml;version=2.0;type=eml20.LocalDepth3dCrs`
- Grid: `application/x-resqml+xml;version=2.0;type=resqml20.IjkGridRepresentation`
- Property: `application/x-resqml+xml;version=2.0;type=resqml20.ContinuousProperty` (or `DiscreteProperty`)
- Object URI format: `eml:///dataspace('<ds>')/uuid(<uuid>)`
- Array URI format (uuid-path style): `eml:///dataspace('<ds>')/uuid(<property_uuid>)/path(values)`

> Note: The ordering of **PutDataObjects** is CRS → Grid → Properties. Arrays are sent after objects.

## Requirements
See `requirements.txt` for pinned packages.

## Limitations / Notes
- `resqpy` typically writes to EPC/HDF5; here we keep everything in-memory using `h5py` core driver.
- The GRDECL lightweight parser supports a subset of keywords and repeat syntax; prefer installing **resdata**.
- Adjust OSDU property references in `osdu_mapping.py` as needed.
```
