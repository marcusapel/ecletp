
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class GrdeclGrid:
    # Logical sizes
    ni: int
    nj: int
    nk: int
    # geometry kind: 'corner-point' or 'cartesian'
    geometry_kind: str
    # For corner-point grids
    coord: Optional[np.ndarray] = None  # ( (nj+1)*(ni+1) , 6 ) or (n_pillars, 6)
    zcorn: Optional[np.ndarray] = None  # (nk*2, nj*2, ni*2) style, flattened per GRDECL
    actnum: Optional[np.ndarray] = None # (nk, nj, ni)
    # For cartesian grids
    dx: Optional[np.ndarray] = None
    dy: Optional[np.ndarray] = None
    dz: Optional[np.ndarray] = None
    tops: Optional[np.ndarray] = None
    # Cell properties keyed by Eclipse keyword, arrays shaped (nk, nj, ni)
    properties: Dict[str, np.ndarray] = field(default_factory=dict)


def _demo_grid() -> GrdeclGrid:
    ni, nj, nk = 2, 2, 3
    # Simple cartesian block 100x100x10m cells, top at 0, z increases downward
    dx = np.full((nk, nj, ni), 100.0, dtype=float)
    dy = np.full((nk, nj, ni), 100.0, dtype=float)
    dz = np.full((nk, nj, ni), 10.0, dtype=float)
    tops = np.zeros((nj, ni), dtype=float)
    actnum = np.ones((nk, nj, ni), dtype=np.int32)
    props = {
        'PORO': np.full((nk, nj, ni), 0.2, dtype=float),
        'PERMX': np.full((nk, nj, ni), 100.0, dtype=float),
        'PRESSURE': np.full((nk, nj, ni), 1.0e7, dtype=float),
        'SWAT': np.full((nk, nj, ni), 0.3, dtype=float),
        'SOIL': np.full((nk, nj, ni), 0.7, dtype=float),
        'ACTNUM': actnum
    }
    return GrdeclGrid(ni=ni, nj=nj, nk=nk, geometry_kind='cartesian', dx=dx, dy=dy, dz=dz, tops=tops, actnum=actnum, properties=props)


def read_grdecl(path: Optional[str]) -> Tuple[GrdeclGrid, Dict[str, str]]:
    """Read a GRDECL file using resdata if available; fallback to PyGRDECL; or return demo grid if path is None.

    Returns: (grid, meta)
    meta: dict with keys {'source','parser','notes'}
    """
    if path is None:
        logger.warning("No --eclgrd provided; using built-in 2x2x3 demo grid")
        return _demo_grid(), {'source':'demo','parser':'demo','notes':'synthetic cartesian grid'}

    # Try Equinor resdata first
    try:
        import resdata.grid as rdg  # type: ignore
        from resdata.grid import rd_grid  # noqa
        logger.info("Parsing GRDECL with resdata")
        grid = rdg.RdGrid(path)  # RdGrid can read GRDECL/EGRID
        dims = grid.dimensions  # (ni, nj, nk) or similar
        ni, nj, nk = int(dims[0]), int(dims[1]), int(dims[2])
        act = np.array(grid.actnum(), dtype=np.int32).reshape(nk, nj, ni)
        props = {}
        for kw in grid.properties():
            try:
                arr = np.array(grid.property(kw)).reshape(nk, nj, ni)
                props[kw.upper()] = arr
            except Exception as ex:
                logger.debug("Skip property %s: %s", kw, ex)
        # Geometry: prefer corner-point if COORD/ZCORN are present
        if hasattr(grid, 'zcorn') and hasattr(grid, 'coord'):
            zcorn = np.array(grid.zcorn())
            coord = np.array(grid.coord())
            gg = GrdeclGrid(ni=ni, nj=nj, nk=nk, geometry_kind='corner-point', coord=coord, zcorn=zcorn, actnum=act, properties=props)
        else:
            # Attempt to derive cartesian from DX/DY/DZ/TOPS
            dx = np.array(getattr(grid, 'dx')()).reshape(nk, nj, ni) if hasattr(grid, 'dx') else None
            dy = np.array(getattr(grid, 'dy')()).reshape(nk, nj, ni) if hasattr(grid, 'dy') else None
            dz = np.array(getattr(grid, 'dz')()).reshape(nk, nj, ni) if hasattr(grid, 'dz') else None
            tops = np.array(getattr(grid, 'tops')()).reshape(nj, ni) if hasattr(grid, 'tops') else None
            gg = GrdeclGrid(ni=ni, nj=nj, nk=nk, geometry_kind='cartesian', dx=dx, dy=dy, dz=dz, tops=tops, actnum=act, properties=props)
        return gg, {'source': path, 'parser':'resdata', 'notes':'Read using Equinor resdata'}
    except Exception as ex:
        logger.info("resdata unavailable or failed: %s", ex)

    # Try PyGRDECL simple parser
    try:
        import re
        logger.info("Parsing GRDECL with lightweight regex (PyGRDECL-like)")
        txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
        def read_kw(name):
            m = re.search(rf"{name}\s*(.*?)/", txt, flags=re.S|re.I)
            if not m: return None
            raw = m.group(1)
            # Expand repeat syntax like 3*1.0
            vals = []
            for token in re.findall(r"[-+\d\.Ee]+\*?[-+\d\.Ee]*", raw):
                if '*' in token:
                    n, v = token.split('*')
                    vals.extend([float(v)]*int(float(n)))
                elif token.strip():
                    vals.append(float(token))
            return np.array(vals)
        dims = read_kw('DIMENS')
        ni, nj, nk = map(int, dims[:3]) if dims is not None else (10,10,3)
        act = read_kw('ACTNUM')
        act = act.reshape(nk, nj, ni) if act is not None else np.ones((nk,nj,ni), dtype=int)
        zcorn = read_kw('ZCORN')
        coord = read_kw('COORD')
        dx = read_kw('DX'); dy = read_kw('DY'); dz = read_kw('DZ'); tops = read_kw('TOPS')
        props = {}
        for kw in ['PORO','PERMX','PERMY','PERMZ','PRESSURE','SWAT','SOIL','SGAS']:
            a = read_kw(kw)
            if a is not None:
                props[kw] = a.reshape(nk, nj, ni)
        if zcorn is not None and coord is not None:
            gg = GrdeclGrid(ni=ni,nj=nj,nk=nk, geometry_kind='corner-point', coord=coord, zcorn=zcorn, actnum=act, properties=props)
        else:
            dx = dx.reshape(nk,nj,ni) if dx is not None else None
            dy = dy.reshape(nk,nj,ni) if dy is not None else None
            dz = dz.reshape(nk,nj,ni) if dz is not None else None
            tops = tops.reshape(nj,ni) if tops is not None else None
            gg = GrdeclGrid(ni=ni,nj=nj,nk=nk, geometry_kind='cartesian', dx=dx, dy=dy, dz=dz, tops=tops, actnum=act, properties=props)
        return gg, {'source': path, 'parser':'regex', 'notes':'Simplified GRDECL read'}
    except Exception as ex:
        logger.exception("Failed to parse GRDECL: %s", ex)
        return _demo_grid(), {'source': path, 'parser':'demo', 'notes':f'Fallback demo due to error: {ex}'}
