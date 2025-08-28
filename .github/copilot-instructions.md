# Copilot Instructions for ecl2resqml_etp

## Project Overview
This project reads Eclipse GRDECL grid files, builds RESQML 2.0.1 objects in memory using `resqpy`, and publishes them to an ETP 1.2 RDDMS store via `pyetp`. It is designed for deterministic, in-memory conversion and transfer of reservoir grid data.

## Key Components
- `grdecl_reader.py`: Reads GRDECL files, using Equinor resdata if available, otherwise a lightweight parser.
- `resqml_builder.py`: Constructs RESQML objects (`LocalDepth3dCrs`, `IjkGridRepresentation`, `Property`) with deterministic UUIDv5.
- `etp_client.py`: Handles ETP 1.2 communication, including Protocols 3 (PutDataObjects), 9 (PutDataArrays), and 18 (Transaction).
- `main.py`: Orchestrates the workflow, parses CLI arguments, and manages logging.
- `config_etp.py` & external `confid_etp.py`: Store ETP connection details (WS_URI, AUTH_TOKEN).

## Data Flow
1. GRDECL file is parsed to extract grid and property data.
2. RESQML objects are built in memory, arrays stored in an in-memory HDF5 file (`h5py`, driver='core').
3. Objects and arrays are published to the ETP store in protocol order: CRS → Grid → Properties → Arrays.

## Developer Workflows
- **Install dependencies:**
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Run main workflow:**
  ```bash
  python -m ecl2resqml_etp.main --eclgrd <GRDECL> --dataspace <ds> --confid <confid_etp.py> --title <title> [--log]
  ```
- **Configuration:**
  - `confid_etp.py` must define `WS_URI` and optionally `AUTH_TOKEN`.

## Project-Specific Patterns
- **Deterministic UUIDs:** All RESQML objects use UUIDv5 for reproducibility.
- **In-memory HDF5:** Arrays are never written to disk; use `h5py` with `driver='core', backing_store=False`.
- **Protocol ordering:** Always publish CRS, then Grid, then Properties, then Arrays.
- **Logging:** URIs and content-types are logged, but never authentication tokens.

## Integration Points
- **External dependencies:**
  - `resqpy` for RESQML object creation
  - `pyetp` for ETP communication
  - `h5py` for in-memory HDF5
- **ETP Store:** Communicates with RDDMS via WebSocket (WS_URI, AUTH_TOKEN)

## Examples
- **Object URI:** `eml:///dataspace('<ds>')/uuid(<uuid>)`
- **Array URI:** `eml:///dataspace('<ds>')/uuid(<property_uuid>)/path(values)`
- **Content types:**
  - CRS: `application/x-eml+xml;version=2.0;type=eml20.LocalDepth3dCrs`
  - Grid: `application/x-resqml+xml;version=2.0;type=resqml20.IjkGridRepresentation`
  - Property: `application/x-resqml+xml;version=2.0;type=resqml20.ContinuousProperty` or `DiscreteProperty`

## Conventions
- No disk I/O for arrays; all data is transient and memory-resident.
- Use deterministic UUIDs for all objects.
- Follow protocol ordering strictly for publishing.

---

If any section is unclear or missing details, please provide feedback to improve these instructions.
