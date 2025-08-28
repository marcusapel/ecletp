
"""ecl2resqml_etp: Read Eclipse GRDECL, build RESQML 2.0.1 objects (in-memory), and publish to an ETP 1.2 RDDMS store.

Modules
-------
- grdecl_reader: parse GRDECL (resdata if available) or produce a demo grid, return a structured Python dataclass with geometry + cell properties.
- resqml_builder: build in-memory RESQML objects (eml20 CRS, IjkGridRepresentation, properties) using resqpy; use an in-memory HDF5 file.
- etp_client: connect to ETP 1.2, manage Protocol 18 transactions, Protocol 3 PutDataObjects, Protocol 9 PutDataArrays with gzip chunking.
- osdu_mapping: Eclipse keyword -> OSDU PropertyNameType mapping and helpers.
- main: CLI entrypoint.
"""
__all__ = [
    'grdecl_reader',
    'resqml_builder',
    'etp_client',
    'osdu_mapping'
]
