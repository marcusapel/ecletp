#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strat_column_mapper_pro.py

Full stratigraphic model conversion:
- RESQML 2.0.1 (+ 2.2-compatible options)  <-> Domain
- OSDU WPC StratigraphicColumnRankInterpretation (+ reference-data ChronoStrat) <-> Domain
- OpenWorks (SMDA Excel / JSON) <-> Domain

Usage: 

1) OpenWorks Excel -> OSDU (column + ranks + chronostrat reference-data)
python strat_column_mapper_pro.py ow-to-osdu \
  --excel smda-api_strat-units.xlsx \
  --emit-refdata \
  -o out/osdu_strat_full.json

2) Split ranks into separate WPC records and back-link them from the column
python strat_column_mapper_pro.py ow-to-osdu \
  --excel smda-api_strat-units.xlsx \
  --split-ranks --emit-refdata \
  -o out/osdu_strat_split.json

3) OSDU -> RESQML EPC (optionally add over/underburden endcaps per v2.2 usage)
python strat_column_mapper_pro.py osdu-to-resqml \
  --manifest out/osdu_strat_full.json \
  --add-over-under-burden \
  -o out/strat_column.epc

4) RESQML EPC -> OSDU (preserve rank order and unit names)
python strat_column_mapper_pro.py resqml-to-osdu \
  --epc input_column.epc \
  --emit-refdata \
  -o out/osdu_from_resqml.json


References:
- RESQML v2.0.1 docs (Energistics): StratigraphicColumnRankInterpretation, FIRP, examples.  # [7](https://docs.energistics.org/RESQML/RESQML_TOPICS/RESQML-000-000-titlepage.html)[1](https://docs.energistics.org/RESQML/RESQML_TOPICS/RESQML-500-110-0-R-sv2010.html)
- resqpy strata classes used for IO.                                                           # [2](https://resqpy.readthedocs.io/en/latest/_autosummary/resqpy.strata.StratigraphicColumnRank.html)
- OSDU Worked Example (Stratigraphy) and ChronoStrat reference-data manifests.                # [5](https://github.com/jonslo/osdu-data-data-definitions/blob/master/Examples/WorkedExamples/Reservoir%20Data/Stratigraphy/README.md)[6](https://community.opengroup.org/osdu/data/data-definitions/-/blob/master/Examples/reference-data/ChronoStratigraphy.1.1.0.json)
- RESQML v2.2 usage notes (over/underburden dummy units).                                     # [8](https://github.com/bp/resqpy/issues/843)
"""

from __future__ import annotations
import argparse
import dataclasses as dc
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

# Optional deps
try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore
# resqpy is optional but required for EPC I/O
try:
    import resqpy.model as rq
    from resqpy.strata import (
        StratigraphicColumn as RqStratColumn,
        StratigraphicColumnRank as RqStratRank,
        StratigraphicUnitInterpretation as RqSUI,
        StratigraphicUnitFeature as RqSUF,
    )
    _HAS_RESQPY = True
except Exception:
    _HAS_RESQPY = False

# ------------- Domain -------------------------------------------------------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _uu(u: Optional[str]) -> str:
    try:
        return str(uuid.UUID(str(u))) if u else str(uuid.uuid4())
    except Exception:
        return str(uuid.uuid4())

@dc.dataclass
class StratRefItem:
    """Reference-data item (e.g., ChronoStrat: System/Series)."""
    scheme: str                   # e.g., 'ChronoStratigraphy'
    rank: Optional[str] = None    # e.g., 'System', 'Series'
    code: Optional[str] = None    # controlled code if applicable
    name: Optional[str] = None
    id: Optional[str] = None      # OSDU RD id: 'osdu:reference-data--ChronoStratigraphy:...'
    uri: Optional[str] = None     # external URI

@dc.dataclass
class StratUnitFeature:
    name: str
    uuid: str

@dc.dataclass
class StratUnitInterp:
    """Interpretation (I in FIRP) of a StratUnitFeature."""
    name: str
    uuid: str
    feature_uuid: str
    unit_type: Optional[str] = None        # group/formation/member/zone/subzone
    top_age_ma: Optional[float] = None
    base_age_ma: Optional[float] = None
    parent_name: Optional[str] = None
    color_html: Optional[str] = None
    extras: Dict[str, Any] = dc.field(default_factory=dict)

@dc.dataclass
class StratRank:
    level: Optional[int]
    rank_name: Optional[str]            # lithostrat/chronostrat label (e.g., 'group','formation','system','series')
    ordering_criteria: str = "older-to-younger"
    units: List[StratUnitInterp] = dc.field(default_factory=list)
    # Optional links to reference-data items (ChronoStrat hierarchy)
    chrono_refs: List[StratRefItem] = dc.field(default_factory=list)

@dc.dataclass
class StratColumn:
    name: str
    uuid: str
    ranks: List[StratRank]
    source_system: Optional[str] = None
    extras: Dict[str, Any] = dc.field(default_factory=dict)

@dc.dataclass
class StratOccurrence:
    """Local/occurrence interpretation (e.g., along a well)."""
    name: str
    uuid: str
    column_uuid: str                      # global column this occurrence references
    rank_uuids: List[str] = dc.field(default_factory=list)
    context: Optional[str] = None         # e.g., wellbore id, section id

@dc.dataclass
class StratColumnSet:
    """Optional grouping/versioning for columns."""
    name: str
    uuid: str
    column_uuids: List[str] = dc.field(default_factory=list)

# ------------- OpenWorks Adapter ---------------------------------------------------------------
class OpenWorksAdapter:
    """Parse SMDA Excel/JSON to domain, and export back as JSON for audit."""

    @staticmethod
    def from_excel(path: str, sheet: str = "ApiStratUnit") -> StratColumn:
        if pd is None:
            raise RuntimeError("pandas required for Excel reading")
        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = [c.strip().lower() for c in df.columns]

        col_name = str(df['strat_column_identifier'].iloc[0])
        col_uuid = _uu(df.get('strat_column_uuid', pd.Series([None]*len(df))).iloc[0])

        # Minimal feature/interpretation split: create a feature per unit name
        features: Dict[str, StratUnitFeature] = {}
        def _feature_for(name: str) -> StratUnitFeature:
            if name not in features:
                features[name] = StratUnitFeature(name=name, uuid=_uu(None))
            return features[name]

        # Build units
        units: List[StratUnitInterp] = []
        for _, r in df.iterrows():
            nm = str(r.get('identifier'))
            feat = _feature_for(nm)
            units.append(StratUnitInterp(
                name=nm,
                uuid=_uu(r.get('uuid')),
                feature_uuid=feat.uuid,
                unit_type=r.get('strat_unit_type'),
                top_age_ma=float(r['top_age']) if r.get('top_age') == r.get('top_age') else None,
                base_age_ma=float(r['base_age']) if r.get('base_age') == r.get('base_age') else None,
                parent_name=r.get('strat_unit_parent') if str(r.get('strat_unit_parent')).lower() != 'null' else None,
                color_html=r.get('color_html'),
                extras={
                    "color_rgb": (r.get('color_r'), r.get('color_g'), r.get('color_b')),
                    "top_label": r.get('top'),
                    "base_label": r.get('base'),
                    "source": r.get('source'),
                    "strat_column_type": r.get('strat_column_type'),
                    "sequence": r.get('strat_unit_sequence')
                }
            ))

        # Build ranks by level and sort by age
        def safe_int(x): 
            try: return int(x)
            except: return None
        df['level'] = df.get('strat_unit_level', None).apply(safe_int)
        rank_name_map = {1:'group', 2:'formation', 3:'subzone'}
        ranks: List[StratRank] = []
        for lvl, sdf in sorted(df.groupby('level'), key=lambda kv: (kv[0] if kv[0] is not None else 9999)):
            sdf = sdf.sort_values(by=['top_age', 'base_age'], ascending=[True, True])
            nm_to_level = set(sdf['identifier'].tolist())
            rank_units = [u for u in units if u.name in nm_to_level]
            ranks.append(StratRank(level=lvl, rank_name=rank_name_map.get(lvl, f"level{lvl}"), units=rank_units))

        return StratColumn(name=col_name, uuid=col_uuid, ranks=ranks, source_system="OpenWorks")

# ------------- OSDU Adapter --------------------------------------------------------------------

class OsduAdapter:
    """
    Build OSDU manifests:
      - Column record with nested ranks/units  (default)
      - Optional: split ranks into separate WPC records and link them from the column (--split-ranks)
      - Optional: emit reference-data (ChronoStratigraphy) manifests (--emit-refdata)

    OSDU examples show that ChronoStrat System/Series are stored as reference-data and referenced from the rank interpretations.  # [5](https://github.com/jonslo/osdu-data-data-definitions/blob/master/Examples/WorkedExamples/Reservoir%20Data/Stratigraphy/README.md)
    """

    DEFAULT_KIND = "osdu:wks:work-product-component--StratigraphicColumnRankInterpretation:1.3.0"
    RD_CHRONO_KIND = "osdu:wks:reference-data--ChronoStratigraphy:1.1.0"

    @staticmethod
    def to_manifests(
        column: StratColumn,
        split_ranks: bool = False,
        emit_refdata: bool = False,
        viewers: Optional[List[str]] = None,
        owners: Optional[List[str]] = None,
        legaltags: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        kind: str = DEFAULT_KIND
    ) -> Dict[str, Any]:

        viewers = viewers or ["data.default.viewers@opendes.dataservices.energy"]
        owners  = owners  or ["data.default.owners@opendes.dataservices.energy"]
        legaltags = legaltags or ["opendes-public-usa-dataset-0001"]
        countries = countries or ["US"]

        def base_record():
            return {
                "acl": {"viewers": viewers, "owners": owners},
                "legal": {"legaltags": legaltags, "otherRelevantDataCountries": countries}
            }

        records: List[Dict[str, Any]] = []
        ref_records: List[Dict[str, Any]] = []

        # Assemble rank payloads & optional refdata
        def build_rank_payload(r: StratRank) -> Dict[str, Any]:
            p = {
                "rankLevel": r.level,
                "rankName": r.rank_name,
                "orderingCriteria": r.ordering_criteria,
                "units": [{
                    "name": u.name,
                    "uuid": u.uuid,
                    "type": u.unit_type,
                    "topAgeMa": u.top_age_ma,
                    "baseAgeMa": u.base_age_ma,
                    "parent": u.parent_name,
                    "colorHtml": u.color_html
                } for u in r.units]
            }
            if r.chrono_refs:
                p["chronoStratRefs"] = [{
                    "id": it.id, "scheme": it.scheme, "rank": it.rank, "name": it.name, "code": it.code, "uri": it.uri
                } for it in r.chrono_refs]
            return p

        ranks_payload = [build_rank_payload(r) for r in column.ranks]

        # Strategy A: single record with embedded ranks (as per Worked Example narrative)
        col_record = {
            **base_record(),
            "id": f"wpc:{column.uuid}",
            "kind": kind,
            "data": {
                "name": column.name,
                "stratigraphicColumnUuid": column.uuid,
                "sourceSystem": column.source_system,
                "createdTime": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "ranks": ranks_payload
            }
        }
        records.append(col_record)

        # Strategy B: optional split ranks
        if split_ranks:
            rank_ids = []
            for idx, rp in enumerate(ranks_payload, start=1):
                rid = f"wpc:{column.uuid}:rank:{idx}"
                rank_ids.append(rid)
                rank_rec = {
                    **base_record(),
                    "id": rid,
                    "kind": kind,
                    "data": {**rp, "name": f"{column.name} Rank {idx}", "stratigraphicColumnUuid": column.uuid}
                }
                records.append(rank_rec)

            # backlink: store rank ids on the column record (relationships)
            col_record.setdefault("data", {}).setdefault("relationships", {})
            col_record["data"]["relationships"]["rankInterpretations"] = [{"id": rid} for rid in rank_ids]

        # Optional: emit reference-data for ChronoStrat
        if emit_refdata:
            seen = set()
            for r in column.ranks:
                for it in r.chrono_refs:
                    key = (it.scheme, it.rank, it.code, it.name)
                    if key in seen: 
                        continue
                    seen.add(key)
                    rd_id = it.id or f"reference-data:{it.scheme}:{(it.code or it.name or str(uuid.uuid4()))}"
                    ref_records.append({
                        "id": rd_id,
                        "kind": OsduAdapter.RD_CHRONO_KIND,
                        "acl": {"viewers": viewers, "owners": owners},
                        "legal": {"legaltags": legaltags, "otherRelevantDataCountries": countries},
                        "data": {
                            "name": it.name,
                            "code": it.code,
                            "rank": it.rank,
                            "scheme": it.scheme,
                            "uri": it.uri
                        }
                    })

        bundle = {"records": records}
        if ref_records:
            bundle["referenceData"] = ref_records
        return bundle

    @staticmethod
    def from_manifest(path: str) -> StratColumn:
        with open(path, 'r') as f:
            payload = json.load(f)
        rec = payload["records"][0] if "records" in payload else payload
        data = rec["data"]
        ranks: List[StratRank] = []
        for r in data.get("ranks", []):
            units = [StratUnitInterp(
                name=u.get("name"), uuid=u.get("uuid") or _uu(None),
                feature_uuid=_uu(None),
                unit_type=u.get("type"),
                top_age_ma=u.get("topAgeMa"), base_age_ma=u.get("baseAgeMa"),
                parent_name=u.get("parent"), color_html=u.get("colorHtml")
            ) for u in r.get("units", [])]
            chrono = [StratRefItem(
                scheme=cr.get("scheme"), rank=cr.get("rank"), code=cr.get("code"),
                name=cr.get("name"), id=cr.get("id"), uri=cr.get("uri")
            ) for cr in r.get("chronoStratRefs", [])]
            ranks.append(StratRank(
                level=r.get("rankLevel"), rank_name=r.get("rankName"),
                ordering_criteria=r.get("orderingCriteria", "older-to-younger"),
                units=units, chrono_refs=chrono
            ))
        return StratColumn(
            name=data.get("name"), uuid=data.get("stratigraphicColumnUuid") or rec.get("id", _uu(None)),
            ranks=ranks, source_system=data.get("sourceSystem")
        )

# ------------- RESQML Adapter ------------------------------------------------------------------

class ResqmlAdapter:
    """
    RESQML 2.0.1 IO using resqpy (global column + ranks + unit feature/interpretation).
    Optional flags:
     - add_over_under_burden: add dummy end units per v2.2 usage guidance.  # [8](https://github.com/bp/resqpy/issues/843)
    """

    @staticmethod
    def from_epc(epc_path: str, column_uuid: Optional[str] = None) -> StratColumn:
        if not _HAS_RESQPY:
            raise RuntimeError("resqpy is required for RESQML I/O")
        model = rq.Model(epc_path)

        # Choose column
        cuuid = column_uuid or (model.uuids(obj_type='StratigraphicColumn')[0])
        sc = RqStratColumn(model, uuid=cuuid)
        name = getattr(sc, 'title', f"StratColumn-{str(sc.uuid)[:8]}")

        ranks: List[StratRank] = []
        # resqpy StratColumn keeps ordered rank uuids
        rank_uuids = getattr(sc, 'rank_uuids', None) or getattr(sc, 'column_rank_uuids', None) or []
        for r_uuid in rank_uuids:
            r = RqStratRank(model, uuid=r_uuid)
            # Ordered unit interpretation uuids (oldest->youngest)
            unit_uuids = getattr(r, 'unit_interpretation_uuids', None) or []
            units: List[StratUnitInterp] = []
            for uu in unit_uuids:
                sui = RqSUI(model, uuid=uu)
                nm = getattr(sui, 'title', None) or f"Unit-{str(uu)[:8]}"
                # We do not necessarily have ages/colors in the XML; keep placeholders in extras.
                units.append(StratUnitInterp(name=nm, uuid=str(uu), feature_uuid=_uu(None)))
            ranks.append(StratRank(level=None, rank_name=getattr(r, 'title', None), units=units))

        return StratColumn(name=name, uuid=str(sc.uuid), ranks=ranks, source_system="RESQML")

    @staticmethod
    def to_epc(column: StratColumn, epc_out: str, add_over_under_burden: bool = False) -> str:
        if not _HAS_RESQPY:
            raise RuntimeError("resqpy is required for RESQML I/O")
        model = rq.Model(new_epc_file=epc_out)

        # Create features & interpretations
        feat_by_name: Dict[str, RqSUF] = {}
        interp_by_name: Dict[str, RqSUI] = {}

        def ensure_feature(nm: str) -> RqSUF:
            if nm not in feat_by_name:
                feat_by_name[nm] = RqSUF(model, title=nm)
            return feat_by_name[nm]

        def ensure_interp(u: StratUnitInterp) -> RqSUI:
            if u.name not in interp_by_name:
                ensure_feature(u.name)
                interp_by_name[u.name] = RqSUI(model, title=u.name)
            return interp_by_name[u.name]

        rq_ranks: List[RqStratRank] = []
        for rank in column.ranks:
            rq_units = [ensure_interp(u) for u in rank.units]
            if add_over_under_burden and rq_units:
                # prepend/append dummy end units per usage guidance
                ensure_feature("Overburden"); ensure_feature("Underburden")
                rq_units = [RqSUI(model, title="Overburden")] + rq_units + [RqSUI(model, title="Underburden")]
            rq_ranks.append(RqStratRank(model, title=(rank.rank_name or "Rank"), unit_interpretations=rq_units))

        RqStratColumn(model, title=column.name, rank_interpretations=rq_ranks)
        model.store_epc()
        return epc_out

# ------------- CLI -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Full Strat Column Mapper (RESQML/OSDU/OpenWorks)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # OpenWorks -> OSDU
    p1 = sub.add_parser("ow-to-osdu", help="OpenWorks (SMDA Excel/JSON) -> OSDU")
    src = p1.add_mutually_exclusive_group(required=True)
    src.add_argument("--excel", help="SMDA Excel path")
    src.add_argument("--json", help="OpenWorks JSON path")
    p1.add_argument("--sheet", default="ApiStratUnit")
    p1.add_argument("--output", "-o", required=True)
    p1.add_argument("--split-ranks", action="store_true")
    p1.add_argument("--emit-refdata", action="store_true")
    p1.set_defaults(func=lambda a: ow_to_osdu(a))

    # RESQML -> OSDU
    p2 = sub.add_parser("resqml-to-osdu", help="RESQML EPC -> OSDU")
    p2.add_argument("--epc", required=True)
    p2.add_argument("--uuid", help="StratigraphicColumn UUID")
    p2.add_argument("--output", "-o", required=True)
    p2.add_argument("--split-ranks", action="store_true")
    p2.add_argument("--emit-refdata", action="store_true")
    p2.set_defaults(func=lambda a: resqml_to_osdu(a))

    # OSDU -> RESQML
    p3 = sub.add_parser("osdu-to-resqml", help="OSDU manifest -> RESQML EPC")
    p3.add_argument("--manifest", required=True)
    p3.add_argument("--output", "-o", required=True)
    p3.add_argument("--add-over-under-burden", action="store_true")
    p3.set_defaults(func=lambda a: osdu_to_resqml(a))

    # OpenWorks -> RESQML
    p4 = sub.add_parser("ow-to-resqml", help="OpenWorks (SMDA Excel/JSON) -> RESQML EPC")
    src2 = p4.add_mutually_exclusive_group(required=True)
    src2.add_argument("--excel", help="SMDA Excel path")
    src2.add_argument("--json", help="OpenWorks JSON path")
    p4.add_argument("--sheet", default="ApiStratUnit")
    p4.add_argument("--output", "-o", required=True)
    p4.add_argument("--add-over-under-burden", action="store_true")
    p4.set_defaults(func=lambda a: ow_to_resqml(a))

    args = ap.parse_args()
    args.func(args)

def ow_to_osdu(a):
    col = OpenWorksAdapter.from_excel(a.excel, a.sheet) if a.excel else _load_ow_json(a.json)
    bundle = OsduAdapter.to_manifests(col, split_ranks=a.split_ranks, emit_refdata=a.emit_refdata)
    with open(a.output, "w") as f: json.dump(bundle, f, indent=2)
    print(f"[ok] OSDU manifest written: {a.output}")

def resqml_to_osdu(a):
    col = ResqmlAdapter.from_epc(a.epc, a.uuid)
    bundle = OsduAdapter.to_manifests(col, split_ranks=a.split_ranks, emit_refdata=a.emit_refdata)
    with open(a.output, "w") as f: json.dump(bundle, f, indent=2)
    print(f"[ok] OSDU manifest written: {a.output}")

def osdu_to_resqml(a):
    col = OsduAdapter.from_manifest(a.manifest)
    epc = ResqmlAdapter.to_epc(col, a.output, add_over_under_burden=a.add_over_under_burden)
    print(f"[ok] RESQML EPC written: {epc}")

def ow_to_resqml(a):
    col = OpenWorksAdapter.from_excel(a.excel, a.sheet) if a.excel else _load_ow_json(a.json)
    epc = ResqmlAdapter.to_epc(col, a.output, add_over_under_burden=a.add_over_under_burden)
    print(f"[ok] RESQML EPC written: {epc}")

def _load_ow_json(path: str) -> StratColumn:
    with open(path, "r") as f:
        d = json.load(f)
    # Simplified: expect similar shape to Excel-derived domain (can be extended per your OW export)
    ranks = []
    for rd in d.get("ranks", []):
        units = [StratUnitInterp(
            name=u["name"], uuid=_uu(u.get("uuid")), feature_uuid=_uu(None),
            unit_type=u.get("type"),
            top_age_ma=u.get("topAgeMa"), base_age_ma=u.get("baseAgeMa"),
            parent_name=u.get("parent"), color_html=u.get("colorHtml")
        ) for u in rd.get("units", [])]
        ranks.append(StratRank(level=rd.get("rankLevel"), rank_name=rd.get("rankName"), units=units))
    return StratColumn(name=d["name"], uuid=_uu(d.get("uuid")), ranks=ranks, source_system="OpenWorks-JSON")

if __name__ == "__main__":
    main()
