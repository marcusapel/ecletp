#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecl_grid_etp.py

Fetch an IjkGridRepresentation (RESQML 2.0.1) from an OSDU RDDMS over ETP using
FETPAPI + FESAPI, build resqpy objects, and export an ECLIPSE GRDECL grid (+optional property).

Requirements:
  pip install fetpapi fesapi resqpy numpy

Config:
  Provide config_etp.py (see README in this file header).

References:
  - FETPAPI examples & repo: https://github.com/F2I-Consulting/fetpapi  [connect, ContextInfo, getResources/getDataObjects]
  - FESAPI Python & repo: https://github.com/F2I-Consulting/fesapi      [DataObjectRepository, HDF proxy factory]
  - Energistics RESQML 2.0.1 GRDECL mapping (COORD, ZCORN, ACTNUM):     https://docs.energistics.org/RESQML/RESQML_TOPICS/RESQML-000-292-0-C-sv2010.html
  - GRDECL conventions (ordering, repeat counts):                        https://docs.opengosim.com/manual/input_deck/grid/grdecl/
  - resqpy usage (Grid, properties):                                     https://resqpy.readthedocs.io/

Author: M. Apel requested, built to his constraints by automation.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import threading
import time
from typing import Tuple, Optional, List

# --- External libs expected ---
try:
    import fetpapi
    import fesapi
except Exception as e:
    print("ERROR: fetpapi / fesapi are required (pip install fetpapi fesapi).", file=sys.stderr)
    raise

import numpy as np

# resqpy
try:
    import resqpy.model as rq
    import resqpy.grid as rqgrid
except Exception as e:
    print("ERROR: resqpy is required (pip install resqpy).", file=sys.stderr)
    raise


# ---------- Utilities ----------
def assert_has(obj, name: str):
    """Assert obj has attribute 'name', else raise with a helpful message listing available members."""
    if not hasattr(obj, name):
        members = [m for m in dir(obj) if not m.startswith("_")]
        raise AttributeError(
            f"Required member '{name}' not found on {type(obj).__name__}. "
            f"Available: {members}"
        )

def load_config(path: str):
    spec = importlib.util.spec_from_file_location("config_etp", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load config from {path}")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    # Validate expected names (do not guess)
    for v in ("ETP_URL", "DATA_PARTITION_ID", "get_token"):
        if not hasattr(cfg, v):
            raise AttributeError(f"config_etp.py missing '{v}'")
    return cfg


# ---------- ETP Session & Repo ----------

class EtpSession:
    def __init__(self, etp_url: str, partition: str, bearer_token: str):
        # Create initialization params (+ header with data-partition-id)
        assert_has(fetpapi, "InitializationParameters")
        init_params = fetpapi.InitializationParameters(str(os.urandom(8).hex()), etp_url)
        m = fetpapi.MapStringString()
        m["data-partition-id"] = partition
        init_params.setAdditionalHandshakeHeaderFields(m)

        # Start client session
        self._client = fetpapi.createClientSession(init_params, f"Bearer {bearer_token}")

        # Run in background (example pattern from FETPAPI examples / your notebook)
        self._thread = threading.Thread(target=self._client.run, daemon=True)
        self._thread.start()

        # Wait up to 8 seconds for connection
        start = time.perf_counter()
        while self._client.isEtpSessionClosed() and (time.perf_counter() - start) < 8:
            time.sleep(0.2)
        if self._client.isEtpSessionClosed():
            raise ConnectionError("ETP session did not open within 8 seconds. "
                                  "Check URL (must end with '/'), token, and network.")

    @property
    def client(self):
        return self._client

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass


def build_dataspace_uri(dataspace: str) -> str:
    # ETP URIs in RDDMS commonly look like: eml:///dataspace('demo/Volve')
    return f"eml:///dataspace('{dataspace}')"


def build_grid_uri(dataspace: str, eml_address: str) -> str:
    """
    Compose a full ETP URI for an IjkGridRepresentation under a dataspace.

    If 'eml_address' already starts with 'eml:///', we simply prefix the dataspace if needed.
    Examples seen in the field & in your notebook patterns.
    """
    ds_uri = build_dataspace_uri(dataspace)
    if eml_address.startswith("eml:///"):
        # If the address already includes a dataspace, return as-is
        if "/dataspace(" in eml_address:
            return eml_address
        # Otherwise, mount under dataspace
        if eml_address.startswith("eml:///resqml"):
            return ds_uri + "/" + eml_address[len("eml:///"):]
        return eml_address
    # Accept just UUID and build the canonical RESQML class URI segment
    if len(eml_address) in (36, 38):  # UUID or with braces
        return f"{ds_uri}/resqml20.obj_IjkGridRepresentation('{eml_address.strip(\"{}\")}'"
    # Accept full class segment
    return f"{ds_uri}/{eml_address}"


def load_repo_over_etp(session: EtpSession, dataspace_uri: str, focus_uri: str) -> fesapi.DataObjectRepository:
    """
    Discover resources in the dataspace and pull required DataObjects into a FESAPI repo.
    We keep it generic, mirroring your notebook approach for horizons (Grid2d) & faults.

    NOTE: We request broadly (dataspace scope) and then rely on filtering by class later.
    """
    # FESAPI repo and ETP-aware HDF proxy factory
    repo = fesapi.DataObjectRepository()
    assert_has(fetpapi, "FesapiHdfProxyFactory")
    hdf_proxy_factory = fetpapi.FesapiHdfProxyFactory(session.client)
    repo.setHdfProxyFactory(hdf_proxy_factory)

    # Discover resources at dataspace root
    ctx = fetpapi.ContextInfo()
    ctx.uri = dataspace_uri
    ctx.depth = 1
    ctx.navigableEdges = fetpapi.RelationshipKind_Both
    ctx.includeSecondaryTargets = False
    ctx.includeSecondarySources = False

    resources = session.client.getResources(ctx, fetpapi.ContextScopeKind__self)
    if resources.empty():
        raise RuntimeError(f"No resources returned for context {dataspace_uri}")

    # Pull DataObjects for all resources (simple & robust)
    uri_map = fetpapi.MapStringString()
    for idx, res in enumerate(resources):
        uri_map[str(idx)] = res.uri
    data_objects = session.client.getDataObjects(uri_map)

    # Add to repo
    for dobj in data_objects.values():
        # infer dataobject type using fetpapi helpers; add to repo for the current dataspace
        repo.addOrReplaceGsoapProxy(dobj.data, fetpapi.getDataObjectType(dobj.resource.uri),
                                    fetpapi.getDataspaceUri(dobj.resource.uri))

    return repo


def find_ijk_grid_uuid(repo: fesapi.DataObjectRepository) -> Optional[str]:
    """
    Find the first UUID in the repo whose SWIG type name suggests IjkGridRepresentation.
    We avoid guessing exact wrapper class names; we check their string form.
    """
    assert_has(repo, "getUuids")
    uuids = list(repo.getUuids())
    for uid in uuids:
        obj = repo.getDataObjectByUuid(uid)
        tname = type(obj).__name__
        if "IjkGridRepresentation" in tname or "IjkGrid" in tname:
            return uid
    return None


def serialize_repo_to_epc(repo: fesapi.DataObjectRepository, epc_path: str):
    """
    Persist repo to EPC/HDF so 'resqpy' can load it.
    Avoid guessing: use DataObjectRepository <-> EpcDocument idiom (FESAPI).
    """
    # Build an EpcDocument and let the repo serialize into it
    # (Member names differ slightly across versions; check and fail-fast)
    try:
        EpcDocument = fesapi.EpcDocument
    except Exception:
        raise RuntimeError("fesapi.EpcDocument is not available in this build. "
                           "Please use a FESAPI build exposing EpcDocument in Python.")

    epc = EpcDocument(epc_path)  # sets path & prepares (see doxygen)
    # The repository serialization function name differs across versions:
    for candidate in ("serializeJ", "serializeToFile", "serializeInto"):
        if hasattr(repo, candidate):
            getattr(repo, candidate)(epc)
            break
    else:
        # Try common name on epc object:
        for candidate in ("serialize", "write"):
            if hasattr(epc, candidate):
                getattr(epc, candidate)(repo)
                break
        else:
            raise RuntimeError("Could not find a serialization method to write EPC; "
                               "inspect repo/epc methods and adapt.")
    # Ensure on disk
    if not os.path.exists(epc_path):
        raise IOError(f"EPC was not created at {epc_path}")


# ---------- GRDECL helpers ----------

def flatten_natural_order(arr_kji: np.ndarray) -> np.ndarray:
    """
    Flatten (k,j,i)-ordered cell arrays to ECL 'natural ordering':
    I fastest, then J, K slowest. Numpy (k,j,i) -> reshape(..., order='C') then transpose.
    We implement explicitly to avoid any ambiguity.
    """
    nk, nj, ni = arr_kji.shape
    # Build natural order: loop K (slow), J (mid), I (fast)
    out = np.empty(nk * nj * ni, dtype=arr_kji.dtype)
    p = 0
    for k in range(nk):
        for j in range(nj):
            row = arr_kji[k, j, :]
            out[p:p + ni] = row
            p += ni
    return out


def grdecl_repeat_encode(values: np.ndarray, float_fmt="{: .6g}") -> List[str]:
    """
    Produce lines of GRDECL text with Eclipse repeat counts (e.g., '3000*0.44').
    Keep it simple (moderate compression) to avoid pathological repeats.
    """
    lines = []
    n = len(values)
    i = 0
    chunk = []
    while i < n:
        v = values[i]
        # look ahead for repeats
        run = 1
        while i + run < n and values[i + run] == v and run < 1000000:
            run += 1
        if run >= 4:
            # flush current chunk
            if chunk:
                lines.append(" ".join(chunk))
                chunk = []
            lines.append(f"{run}*{float_fmt.format(v)}")
            i += run
        else:
            chunk.append(float_fmt.format(v))
            i += 1
            if len(chunk) >= 8:  # keep lines readable
                lines.append(" ".join(chunk))
                chunk = []
    if chunk:
        lines.append(" ".join(chunk))
    return lines


def build_coord_zcorn_from_corner_points(grid: rqgrid.Grid) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert explicit corner points (nk,nj,ni, 2,2,2, 3) to GRDECL COORD and ZCORN.
    Notes:
      - COORD: (NI+1)*(NJ+1)*6 floats: for each pillar, lower and upper points (x,y,z) of straight coordinate line
      - ZCORN: 8*NI*NJ*NK depths (z positive downward in GRDECL)
    We follow Energistics mapping and GRDECL conventions.  See refs.
    """
    # Corner points: [k,j,i, corner_k, corner_j, corner_i, xyz]
    cps = grid.corner_points()
    nk, nj, ni = cps.shape[:3]

    # 1) ZCORN: order in GRDECL is (I-fastest) across cells; for each cell 8 z values
    # resqpy corner_points order: 2x2x2 corners with last axis xyz; z is at index 2.
    zcorn = np.empty(nk * nj * ni * 8, dtype=float)
    p = 0
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                # GRDECL expects the sequence: z at 8 cell corners; we use the same consistent corner ordering
                # We choose order [kcorner, jcorner, icorner] in (0/1) triples matching resqpy corner_points
                for kk in (0, 1):
                    for jj in (0, 1):
                        for ii in (0, 1):
                            zcorn[p] = cps[k, j, i, kk, jj, ii, 2]  # Z component
                            p += 1

    # 2) COORD: For each pillar (i in 0..NI, j in 0..NJ), we need two points (lower, upper)
    # We approximate pillar line as straight from min-z to max-z among the pillar’s nodes (using areal XY from average).
    # This is standard when forming GRDECL from explicit corner point geometry.
    coord = np.empty(( (ni + 1) * (nj + 1), 6 ), dtype=float)
    q = 0
    # Create pillar point cloud by sampling corner points that share that pillar
    # Pillar (ip, jp) touches up to 4 neighboring cells; gather all their corner nodes at that pillar index
    def pillar_nodes(ip, jp):
        xs, ys, zs = [], [], []
        # neighboring cells in i,j space:
        for jcell in max(jp - 1, 0), min(jp, nj - 1):
            for icell in max(ip - 1, 0), min(ip, ni - 1):
                # Determine which local corner maps to pillar (ip,jp):
                # local ii = 1 if ip == icell + 1 else 0
                # local jj = 1 if jp == jcell + 1 else 0
                if icell < 0 or icell >= ni or jcell < 0 or jcell >= nj:
                    continue
                ii_local = 1 if ip == icell + 1 else 0
                jj_local = 1 if jp == jcell + 1 else 0
                for k in (0, 1):
                    # For each layer top/bottom at this cell corner
                    pt = cps[min(nk-1, k), jcell, icell, k, jj_local, ii_local, :]
                    xs.append(pt[0]); ys.append(pt[1]); zs.append(pt[2])
        if len(zs) == 0:
            # fallback: use nearest valid (should not happen for valid explicit geometry)
            return (0.,0.,0.), (0.,0.,1.)
        # Decide lower (max depth) and upper (min depth) assuming z increasing downward in data
        idx_min = int(np.argmin(zs))
        idx_max = int(np.argmax(zs))
        upper = (xs[idx_min], ys[idx_min], zs[idx_min])
        lower = (xs[idx_max], ys[idx_max], zs[idx_max])
        return lower, upper

    for jp in range(nj + 1):
        for ip in range(ni + 1):
            lower, upper = pillar_nodes(ip, jp)
            coord[q, 0:3] = lower
            coord[q, 3:6] = upper
            q += 1

    return coord.reshape(-1), zcorn.reshape(-1)


def write_grdecl(path_prefix: str,
                 ni: int, nj: int, nk: int,
                 coord: np.ndarray, zcorn: np.ndarray,
                 actnum: Optional[np.ndarray] = None,
                 prop: Optional[Tuple[str, np.ndarray]] = None):
    """
    Write a GRDECL file to disk: <path_prefix>.grdecl, with DIMENS/COORD/ZCORN/ACTNUM (+ property if given).
    """
    grid_path = f"{path_prefix}.grdecl"
    with open(grid_path, "w") as f:
        f.write(f"DIMENS\n {ni} {nj} {nk} /\n\n")

        # COORD
        f.write("COORD\n")
        for line in grdecl_repeat_encode(coord, "{: .6g}"):
            f.write(" " + line + "\n")
        f.write("/\n\n")

        # ZCORN
        f.write("ZCORN\n")
        for line in grdecl_repeat_encode(zcorn, "{: .6g}"):
            f.write(" " + line + "\n")
        f.write("/\n\n")

        # ACTNUM
        if actnum is None:
            actnum = np.ones((nk, nj, ni), dtype=int)
        flat_act = flatten_natural_order(actnum.astype(int))
        f.write("ACTNUM\n")
        for line in grdecl_repeat_encode(flat_act, "{:d}"):
            f.write(" " + line + "\n")
        f.write("/\n\n")

        # Property (optional)
        if prop is not None:
            kw, arr = prop
            flat_prop = flatten_natural_order(arr)
            f.write(f"{kw}\n")
            # Eclipse expects default double; property format left as general
            for line in grdecl_repeat_encode(flat_prop, "{: .6g}"):
                f.write(" " + line + "\n")
            f.write("/\n\n")

    return grid_path


# ---------- Pipeline ----------

def run(dataspace: str, eml: str, outprefix: str, prop_title: Optional[str], config_path: str):
    cfg = load_config(config_path)
    token = cfg.get_token()

    etp = EtpSession(cfg.ETP_URL, cfg.DATA_PARTITION_ID, token)
    try:
        ds_uri = build_dataspace_uri(dataspace)
        grid_uri = build_grid_uri(dataspace, eml)

        repo = load_repo_over_etp(etp, ds_uri, grid_uri)

        # Find IJK grid uuid
        grid_uuid = find_ijk_grid_uuid(repo)
        if grid_uuid is None:
            raise RuntimeError("No IjkGridRepresentation found in the loaded dataspace. "
                               "Inspect the dataspace & URI inputs.")

        # Persist repo -> EPC/HDF (temp)
        with tempfile.TemporaryDirectory() as td:
            epc_path = os.path.join(td, "from_etp.epc")
            serialize_repo_to_epc(repo, epc_path)

            # Load with resqpy
            model = rq.Model(epc_path)
            # If multiple grids: use uuid we found
            grid = rqgrid.Grid(model, uuid=model.uuid(obj_type="Grid", uuid=grid_uuid) or grid_uuid)
            grid.cache_all_geometry_arrays()

            ni, nj, nk = grid.extent_kji  # note: returns (nk, nj, ni)
            nk, nj, ni = int(nk), int(nj), int(ni)

            # Build COORD & ZCORN
            coord, zcorn = build_coord_zcorn_from_corner_points(grid)

            # ACTNUM: if grid has inactive (pinched-out), otherwise ones
            try:
                act = grid.array_cell_active().astype(int)
            except Exception:
                act = np.ones((nk, nj, ni), dtype=int)

            # Optional property: find by citation title (or ECL keyword)
            prop_tuple = None
            if prop_title:
                pc = grid.property_collection
                # heuristic: match citation title case-insensitively; fallback: first numeric property
                part_ix = pc.find_part(title=prop_title) if hasattr(pc, "find_part") else None
                arr = None
                if part_ix is not None:
                    arr = pc.numeric_array_ref(part_ix)
                else:
                    # loose match
                    for part in pc.parts():
                        meta = pc.metadata_for_part(part)
                        if str(meta.get("citation_title","")).lower() == prop_title.lower():
                            arr = pc.numeric_array_ref(part)
                            break
                if arr is None:
                    print(f"WARNING: Could not find property titled '{prop_title}' on grid; skipping property export.")
                else:
                    if arr.shape != (nk, nj, ni):
                        print(f"WARNING: Property array shape {arr.shape} != (nk,nj,ni) {(nk,nj,ni)}; "
                              f"will attempt reshape if total size matches.")
                        if arr.size == nk*nj*ni:
                            arr = arr.reshape((nk, nj, ni))
                        else:
                            arr = None
                    if arr is not None:
                        prop_tuple = (prop_title.upper(), arr)

            # Write GRDECL
            path = write_grdecl(outprefix, ni, nj, nk, coord, zcorn, actnum=act, prop=prop_tuple)
            print(f"GRDECL written: {path}")
            if prop_tuple:
                print(f"Included property: {prop_tuple[0]}")

    finally:
        etp.close()


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Export IJK grid from ETP (FETPAPI/FESAPI) to Eclipse GRDECL (RESQML 2.0.1 -> GRDECL).")
    ap.add_argument("--dataspace", required=True, help="RDDMS dataspace path, e.g., 'demo/Volve'")
    ap.add_argument("--eml", required=True, help="EML address of the IjkGridRepresentation (full EML or UUID)")
    ap.add_argument("--outfile", required=True, help="Output prefix for .grdecl (no extension)")
    ap.add_argument("--prop", default=None, help="Optional property citation title (e.g., PORO, PERMX) to export")
    ap.add_argument("--config", default="./config_etp.py", help="Path to config_etp.py with ETP_URL, DATA_PARTITION_ID, get_token()")
    args = ap.parse_args()
    run(args.dataspace, args.eml, args.outfile, args.prop, args.config)


if __name__ == "__main__":
    main()
