#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strat_column_mapper_pro.py

Full stratigraphic model conversion utilities:
- RESQML 2.0.1 I/O (with optional v2.2-style over/underburden endcaps)  <-> Domain model
- OSDU WPC StratigraphicColumnRankInterpretation (+ optional ChronoStrat reference-data) <-> Domain model
- OpenWorks (SMDA Excel / JSON) <-> Domain model
- RESQML obj_StratigraphicOccurrenceInterpretation XML writer (RESQML 2.0.1-compliant)

CLI Commands:
  * resqml2osdu    : Convert a RESQML EPC to an OSDU manifest
  * osdu2resqml    : Convert an OSDU manifest to a RESQML EPC
  * ow2resqml      : Convert OpenWorks (Excel/JSON) to a RESQML EPC
  * ow2osdu        : Convert OpenWorks (Excel/JSON) to an OSDU manifest
Extras:
  * make-occurrence: Create a RESQML obj_StratigraphicOccurrenceInterpretation XML

Dependencies:
  - pandas  (optional; required only for --excel inputs)
  - resqpy  (optional; required for RESQML EPC read/write)
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Optional dependency: pandas (Excel I/O)
try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

# Optional dependency: resqpy (RESQML EPC I/O)
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

# For Occurrence XML writing
import xml.etree.ElementTree as ET


# ======================================================================================
# Utilities
# ======================================================================================

def _now_utc_iso() -> str:
    """Return current UTC timestamp in ISO-8601 without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_uuid(u: Optional[str]) -> str:
    """Return a valid UUID string; generate one if not provided or invalid."""
    try:
        return str(uuid.UUID(str(u))) if u else str(uuid.uuid4())
    except Exception:
        return str(uuid.uuid4())


# ======================================================================================
# Domain Model
# ======================================================================================

@dc.dataclass
class StratRefItem:
    """
    Reference-data item (e.g., OSDU ChronoStratigraphy entry).
    """
    scheme: str                   # e.g., 'ChronoStratigraphy'
    rank: Optional[str] = None    # e.g., 'System', 'Series'
    code: Optional[str] = None
    name: Optional[str] = None
    id: Optional[str] = None      # OSDU RD id: osdu:wks:reference-data--ChronoStratigraphy:...
    uri: Optional[str] = None     # Optional external URI


@dc.dataclass
class StratUnitFeature:
    """
    RESQML Feature object representing a geologic unit (F in FIRP).
    """
    name: str
    uuid: str


@dc.dataclass
class StratUnitInterp:
    """
    RESQML StratigraphicUnitInterpretation (I in FIRP), tied to a feature.
    """
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
    """
    A rank in the column (e.g., group/formation or system/series) with ordered units (oldest→youngest).
    """
    level: Optional[int]
    rank_name: Optional[str]                   # 'group', 'formation', 'system', 'series', ...
    ordering_criteria: str = "older-to-younger"
    units: List[StratUnitInterp] = dc.field(default_factory=list)
    chrono_refs: List[StratRefItem] = dc.field(default_factory=list)  # optional OSDU reference-data links


@dc.dataclass
class StratColumn:
    """
    Global stratigraphic column (RESQML StratigraphicColumn + ordered ranks).
    """
    name: str
    uuid: str
    ranks: List[StratRank]
    source_system: Optional[str] = None
    extras: Dict[str, Any] = dc.field(default_factory=dict)


@dc.dataclass
class StratOccurrence:
    """
    Local/occurrence stratigraphic interpretation (e.g., along a well/section).
    """
    name: str
    uuid: str
    column_uuid: str                      # global column this occurrence references
    rank_uuids: List[str] = dc.field(default_factory=list)
    context: Optional[str] = None         # e.g., wellbore id, section id


@dc.dataclass
class StratColumnSet:
    """
    Optional grouping/versioning for multiple columns.
    """
    name: str
    uuid: str
    column_uuids: List[str] = dc.field(default_factory=list)


# ======================================================================================
# OpenWorks Adapter
# ======================================================================================

class OpenWorksAdapter:
    """Parse OpenWorks/SMDA Excel or JSON into the neutral domain model."""

    @staticmethod
    def from_excel(path: str, sheet: str = "ApiStratUnit") -> StratColumn:
        if pd is None:
            raise RuntimeError("pandas is required to read Excel. Please install pandas.")

        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = [c.strip().lower() for c in df.columns]

        col_name = str(df["strat_column_identifier"].iloc[0])
        col_uuid = _ensure_uuid(df.get("strat_column_uuid", pd.Series([None] * len(df))).iloc[0])

        # Build Feature/Interpretation pairs (one feature per unit name)
        features: Dict[str, StratUnitFeature] = {}

        def _feature_for(name: str) -> StratUnitFeature:
            if name not in features:
                features[name] = StratUnitFeature(name=name, uuid=_ensure_uuid(None))
            return features[name]

        units: List[StratUnitInterp] = []
        for _, r in df.iterrows():
            nm = str(r.get("identifier"))
            feat = _feature_for(nm)
            units.append(
                StratUnitInterp(
                    name=nm,
                    uuid=_ensure_uuid(r.get("uuid")),
                    feature_uuid=feat.uuid,
                    unit_type=r.get("strat_unit_type"),
                    top_age_ma=float(r["top_age"]) if r.get("top_age") == r.get("top_age") else None,
                    base_age_ma=float(r["base_age"]) if r.get("base_age") == r.get("base_age") else None,
                    parent_name=r.get("strat_unit_parent")
                    if str(r.get("strat_unit_parent")).lower() != "null"
                    else None,
                    color_html=r.get("color_html"),
                    extras={
                        "color_rgb": (r.get("color_r"), r.get("color_g"), r.get("color_b")),
                        "top_label": r.get("top"),
                        "base_label": r.get("base"),
                        "source": r.get("source"),
                        "strat_column_type": r.get("strat_column_type"),
                        "sequence": r.get("strat_unit_sequence"),
                    },
                )
            )

        def _safe_int(x) -> Optional[int]:
            try:
                return int(x)
            except Exception:
                return None

        df["level"] = df.get("strat_unit_level", None).apply(_safe_int)
        rank_name_map = {1: "group", 2: "formation", 3: "subzone"}

        ranks: List[StratRank] = []
        for lvl, sdf in sorted(
            df.groupby("level"), key=lambda kv: (kv[0] if kv[0] is not None else 9999)
        ):
            sf = sdf.sort_values(by=["top_age", "base_age"], ascending=[True, True])
            names_at_level = set(sf["identifier"].tolist())
            rank_units = [u for u in units if u.name in names_at_level]
            ranks.append(
                StratRank(level=lvl, rank_name=rank_name_map.get(lvl, f"level{lvl}"), units=rank_units)
            )

        return StratColumn(
            name=col_name, uuid=col_uuid, ranks=ranks, source_system="OpenWorks/SMDA"
        )

    @staticmethod
    def from_json(path: str) -> StratColumn:
        with open(path, "r") as f:
            d = json.load(f)

        # Expect shape akin to the Excel-derived domain; adjust as needed
        ranks: List[StratRank] = []
        for rd in d.get("ranks", []):
            units = [
                StratUnitInterp(
                    name=u["name"],
                    uuid=_ensure_uuid(u.get("uuid")),
                    feature_uuid=_ensure_uuid(None),
                    unit_type=u.get("type"),
                    top_age_ma=u.get("topAgeMa"),
                    base_age_ma=u.get("baseAgeMa"),
                    parent_name=u.get("parent"),
                    color_html=u.get("colorHtml"),
                )
                for u in rd.get("units", [])
            ]
            ranks.append(
                StratRank(level=rd.get("rankLevel"), rank_name=rd.get("rankName"), units=units)
            )

        return StratColumn(
            name=d.get("name", "OpenWorks Column"),
            uuid=_ensure_uuid(d.get("uuid")),
            ranks=ranks,
            source_system="OpenWorks-JSON",
        )


# ======================================================================================
# OSDU Adapter
# ======================================================================================

class OsduAdapter:
    """
    Build OSDU manifests for WPC StratigraphicColumnRankInterpretation:
      - By default: one record containing embedded ranks/units.
      - Optionally: split ranks into separate WPC records and relate them to the column (--split-ranks).
    Optionally emit reference-data (ChronoStratigraphy) manifests (--emit-refdata).
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
        kind: str = DEFAULT_KIND,
    ) -> Dict[str, Any]:
        # Defaults
        viewers = viewers or ["data.default.viewers@opendes.dataservices.energy"]
        owners = owners or ["data.default.owners@opendes.dataservices.energy"]
        legaltags = legaltags or ["opendes-public-usa-dataset-0001"]
        countries = countries or ["US"]

        def base_record() -> Dict[str, Any]:
            return {
                "acl": {"viewers": viewers, "owners": owners},
                "legal": {"legaltags": legaltags, "otherRelevantDataCountries": countries},
            }

        records: List[Dict[str, Any]] = []
        ref_records: List[Dict[str, Any]] = []

        # Build rank payload
        def build_rank_payload(r: StratRank) -> Dict[str, Any]:
            payload = {
                "rankLevel": r.level,
                "rankName": r.rank_name,
                "orderingCriteria": r.ordering_criteria,
                "units": [
                    {
                        "name": u.name,
                        "uuid": u.uuid,
                        "type": u.unit_type,
                        "topAgeMa": u.top_age_ma,
                        "baseAgeMa": u.base_age_ma,
                        "parent": u.parent_name,
                        "colorHtml": u.color_html,
                    }
                    for u in r.units
                ],
            }
            if r.chrono_refs:
                payload["chronoStratRefs"] = [
                    {
                        "id": it.id,
                        "scheme": it.scheme,
                        "rank": it.rank,
                        "name": it.name,
                        "code": it.code,
                        "uri": it.uri,
                    }
                    for it in r.chrono_refs
                ]
            return payload

        ranks_payload = [build_rank_payload(r) for r in column.ranks]

        # Column record (with embedded ranks)
        col_record = {
            **base_record(),
            "id": f"wpc:{column.uuid}",
            "kind": kind,
            "data": {
                "name": column.name,
                "stratigraphicColumnUuid": column.uuid,
                "sourceSystem": column.source_system,
                "createdTime": _now_utc_iso(),
                "ranks": ranks_payload,
            },
        }
        records.append(col_record)

        # Optional: split ranks into separate WPC records, linked from the column
        if split_ranks:
            rank_ids: List[str] = []

            for idx, rp in enumerate(ranks_payload, start=1):
                rid = f"wpc:{column.uuid}:rank:{idx}"
                rank_ids.append(rid)

                rank_rec = {
                    **base_record(),
                    "id": rid,
                    "kind": kind,
                    "data": {
                        **rp,
                        "name": f"{column.name} Rank {idx}",
                        "stratigraphicColumnUuid": column.uuid,
                    },
                }
                records.append(rank_rec)

            col_record.setdefault("data", {}).setdefault("relationships", {})
            col_record["data"]["relationships"]["rankInterpretations"] = [{"id": rid} for rid in rank_ids]

        # Optional: emit reference-data records (ChronoStrat)
        if emit_refdata:
            seen = set()
            for r in column.ranks:
                for it in r.chrono_refs:
                    key = it.id or (it.scheme, it.rank, it.code, it.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    rd_id = it.id or f"reference-data:{it.scheme}:{(it.code or it.name or str(uuid.uuid4()))}"
                    ref_records.append(
                        {
                            "id": rd_id,
                            "kind": OsduAdapter.RD_CHRONO_KIND,
                            "acl": {"viewers": viewers, "owners": owners},
                            "legal": {"legaltags": legaltags, "otherRelevantDataCountries": countries},
                            "data": {
                                "name": it.name,
                                "code": it.code,
                                "rank": it.rank,
                                "scheme": it.scheme,
                                "uri": it.uri,
                            },
                        }
                    )

        bundle = {"records": records}
        if ref_records:
            bundle["referenceData"] = ref_records
        return bundle

    @staticmethod
    def from_manifest(path: str) -> StratColumn:
        with open(path, "r") as f:
            payload = json.load(f)

        rec = payload["records"][0] if "records" in payload else payload
        data = rec["data"]

        ranks: List[StratRank] = []
        for r in data.get("ranks", []):
            units = [
                StratUnitInterp(
                    name=u.get("name"),
                    uuid=_ensure_uuid(u.get("uuid")),
                    feature_uuid=_ensure_uuid(None),
                    unit_type=u.get("type"),
                    top_age_ma=u.get("topAgeMa"),
                    base_age_ma=u.get("baseAgeMa"),
                    parent_name=u.get("parent"),
                    color_html=u.get("colorHtml"),
                )
                for u in r.get("units", [])
            ]
            chrono = [
                StratRefItem(
                    scheme=cr.get("scheme"),
                    rank=cr.get("rank"),
                    code=cr.get("code"),
                    name=cr.get("name"),
                    id=cr.get("id"),
                    uri=cr.get("uri"),
                )
                for cr in r.get("chronoStratRefs", [])
            ]
            ranks.append(
                StratRank(
                    level=r.get("rankLevel"),
                    rank_name=r.get("rankName"),
                    ordering_criteria=r.get("orderingCriteria", "older-to-younger"),
                    units=units,
                    chrono_refs=chrono,
                )
            )

        return StratColumn(
            name=data.get("name", "StratColumn"),
            uuid=data.get("stratigraphicColumnUuid") or _ensure_uuid(rec.get("id")),
            ranks=ranks,
            source_system=data.get("sourceSystem"),
        )


# ======================================================================================
# RESQML Adapter
# ======================================================================================

class ResqmlAdapter:
    """
    RESQML 2.0.1 EPC I/O using resqpy (global column + ranks + unit feature/interpretation).
    Optional:
      - add_over_under_burden: include dummy end units (overburden/underburden) per common v2.2 usage guidance.
    """

    @staticmethod
    def from_epc(epc_path: str, column_uuid: Optional[str] = None) -> StratColumn:
        if not _HAS_RESQPY:
            raise RuntimeError("resqpy is required for RESQML I/O. Please install resqpy.")

        model = rq.Model(epc_path)
        cuuid = column_uuid or (model.uuids(obj_type="StratigraphicColumn")[0])

        sc = RqStratColumn(model, uuid=cuuid)
        name = getattr(sc, "title", f"StratColumn-{str(sc.uuid)[:8]}")

        ranks: List[StratRank] = []
        rank_uuids = getattr(sc, "rank_uuids", None) or getattr(sc, "column_rank_uuids", None) or []

        for r_uuid in rank_uuids:
            r = RqStratRank(model, uuid=r_uuid)
            unit_uuids = getattr(r, "unit_interpretation_uuids", None) or []

            units: List[StratUnitInterp] = []
            for uu in unit_uuids:
                sui = RqSUI(model, uuid=uu)
                nm = getattr(sui, "title", None) or f"Unit-{str(uu)[:8]}"
                # Ages/colors usually not present in XML; keep in extras if needed
                units.append(StratUnitInterp(name=nm, uuid=str(uu), feature_uuid=_ensure_uuid(None)))

            ranks.append(StratRank(level=None, rank_name=getattr(r, "title", None), units=units))

        return StratColumn(name=name, uuid=str(sc.uuid), ranks=ranks, source_system="RESQML")

    @staticmethod
    def to_epc(column: StratColumn, epc_out: str, add_over_under_burden: bool = False) -> str:
        if not _HAS_RESQPY:
            raise RuntimeError("resqpy is required for RESQML I/O. Please install resqpy.")

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
                ensure_feature("Overburden")
                ensure_feature("Underburden")
                rq_units = [RqSUI(model, title="Overburden")] + rq_units + [RqSUI(model, title="Underburden")]
            rq_ranks.append(RqStratRank(model, title=(rank.rank_name or "Rank"), unit_interpretations=rq_units))

        RqStratColumn(model, title=column.name, rank_interpretations=rq_ranks)
        model.store_epc()
        return epc_out


# ======================================================================================
# OSDU ChronoStrat reference-data loader & autolinker
# ======================================================================================

def load_chronostrat_reference_data(path: str) -> List[StratRefItem]:
    """
    Load OSDU reference-data ChronoStratigraphy manifest (JSON), return list of StratRefItem.
    Accepts {"records":[...]} or a bare list of records (each with .data fields).
    """
    with open(path, "r") as f:
        payload = json.load(f)

    recs = payload.get("records", payload)
    out: List[StratRefItem] = []
    for r in recs:
        d = r.get("data", {})
        out.append(
            StratRefItem(
                scheme=d.get("scheme") or "ChronoStratigraphy",
                rank=d.get("rank"),
                code=d.get("code"),
                name=d.get("name"),
                id=r.get("id"),
                uri=d.get("uri"),
            )
        )
    return out


def autolink_chronostrat(column: StratColumn, chrono_items: List[StratRefItem]) -> None:
    """
    Attach ChronoStrat references to ranks by simple name matching (case-insensitive).
    Extendable to age-overlap logic if ages are present in column units and in Chrono RD.
    """
    by_name: Dict[str, List[StratRefItem]] = {}
    for it in chrono_items:
        if it.name:
            by_name.setdefault(it.name.lower(), []).append(it)

    for r in column.ranks:
        linked: List[StratRefItem] = []
        for u in r.units:
            key = (u.name or "").lower()
            if key in by_name:
                linked.extend(by_name[key])

        # De-duplicate by id (or fallback compound key)
        seen = set()
        r.chrono_refs = []
        for it in linked:
            k = it.id or (it.scheme, it.rank, it.code, it.name)
            if k in seen:
                continue
            seen.add(k)
            r.chrono_refs.append(it)


# ======================================================================================
# Minimal RESQML 2.0.1 Occurrence writer (XML part)
# ======================================================================================

def write_resqml_occurrence_xml(output_path: str, title: str, rank_uuid: str, ordering: str = "depth") -> str:
    """
    Minimal RESQML 2.0.1 obj_StratigraphicOccurrenceInterpretation writer.
    Produces a standalone XML part referencing a StratigraphicColumnRankInterpretation UUID.
    """
    NS = {
        "resqml2": "http://www.energistics.org/energyml/data/resqmlv2",
        "eml": "http://www.energistics.org/energyml/data/commonv2",
    }
    for k, v in NS.items():
        try:
            ET.register_namespace(k, v)
        except Exception:
            pass

    occ_uuid = str(uuid.uuid4())
    root = ET.Element("{%s}obj_StratigraphicOccurrenceInterpretation" % NS["resqml2"])

    cite = ET.SubElement(root, "{%s}Citation" % NS["eml"])
    ET.SubElement(cite, "{%s}Title" % NS["eml"]).text = title
    ET.SubElement(cite, "{%s}Originator" % NS["eml"]).text = "strat_column_mapper_pro"
    ET.SubElement(cite, "{%s}Creation" % NS["eml"]).text = _now_utc_iso()
    ET.SubElement(cite, "{%s}UUID" % NS["eml"]).text = occ_uuid

    ET.SubElement(root, "{%s}OrderingCriteria" % NS["resqml2"]).text = ordering

    link = ET.SubElement(root, "{%s}StratigraphicColumnRankInterpretation" % NS["resqml2"])
    dor = ET.SubElement(link, "{%s}DataObjectReference" % NS["eml"])
    ET.SubElement(dor, "{%s}ContentType" % NS["eml"]).text = (
        "application/x-resqml+xml;version=2.0;type=obj_StratigraphicColumnRankInterpretation"
    )
    ET.SubElement(dor, "{%s}UUID" % NS["eml"]).text = rank_uuid
    ET.SubElement(dor, "{%s}Title" % NS["eml"]).text = "Referenced Rank"

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


# ======================================================================================
# CLI Handlers (renamed commands)
# ======================================================================================

def _cmd_resqml2osdu(args: argparse.Namespace) -> None:
    """Convert RESQML EPC -> OSDU manifest."""
    col = ResqmlAdapter.from_epc(args.epc, args.uuid)

    if args.chronostrat_rd:
        items = load_chronostrat_reference_data(args.chronostrat_rd)
        autolink_chronostrat(col, items)

    bundle = OsduAdapter.to_manifests(
        column=col,
        split_ranks=args.split_ranks,
        emit_refdata=args.emit_refdata,
        kind=args.kind,
        viewers=args.viewers,
        owners=args.owners,
        legaltags=args.legaltags,
        countries=args.countries,
    )
    with open(args.output, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"[ok] OSDU manifest written: {args.output}")


def _cmd_osdu2resqml(args: argparse.Namespace) -> None:
    """Convert OSDU manifest -> RESQML EPC."""
    col = OsduAdapter.from_manifest(args.manifest)
    epc = ResqmlAdapter.to_epc(col, args.output, add_over_under_burden=args.add_over_under_burden)
    print(f"[ok] RESQML EPC written: {epc}")


def _cmd_ow2resqml(args: argparse.Namespace) -> None:
    """Convert OpenWorks (SMDA Excel/JSON) -> RESQML EPC."""
    if args.excel:
        col = OpenWorksAdapter.from_excel(args.excel, args.sheet)
    else:
        col = OpenWorksAdapter.from_json(args.json)

    epc = ResqmlAdapter.to_epc(col, args.output, add_over_under_burden=args.add_over_under_burden)
    print(f"[ok] RESQML EPC written: {epc}")


def _cmd_ow2osdu(args: argparse.Namespace) -> None:
    """Convert OpenWorks (SMDA Excel/JSON) -> OSDU manifest."""
    if args.excel:
        col = OpenWorksAdapter.from_excel(args.excel, args.sheet)
    else:
        col = OpenWorksAdapter.from_json(args.json)

    if args.chronostrat_rd:
        items = load_chronostrat_reference_data(args.chronostrat_rd)
        autolink_chronostrat(col, items)

    bundle = OsduAdapter.to_manifests(
        column=col,
        split_ranks=args.split_ranks,
        emit_refdata=args.emit_refdata,
        kind=args.kind,
        viewers=args.viewers,
        owners=args.owners,
        legaltags=args.legaltags,
        countries=args.countries,
    )
    with open(args.output, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"[ok] OSDU manifest written: {args.output}")


# ======================================================================================
# Main CLI
# ======================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Stratigraphic Column Mapper (RESQML/OSDU/OpenWorks)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # resqml2osdu
    # ------------------------------------------------------------------
    p_r2o = sub.add_parser("resqml2osdu", help="Convert RESQML EPC -> OSDU manifest")
    p_r2o.add_argument("--epc", required=True, help="Input RESQML EPC path")
    p_r2o.add_argument("--uuid", help="(Optional) StratigraphicColumn UUID inside the EPC")
    p_r2o.add_argument("--output", "-o", required=True, help="Output OSDU manifest JSON path")
    p_r2o.add_argument("--split-ranks", action="store_true", help="Split ranks to separate WPC records")
    p_r2o.add_argument("--emit-refdata", action="store_true", help="Emit ChronoStrat reference-data records")
    p_r2o.add_argument("--chronostrat-rd", help="OSDU ChronoStrat reference-data manifest JSON to auto-link")
    p_r2o.add_argument("--kind", default=OsduAdapter.DEFAULT_KIND, help="OSDU kind for WPC")
    p_r2o.add_argument("--viewers", nargs="*")
    p_r2o.add_argument("--owners", nargs="*")
    p_r2o.add_argument("--legaltags", nargs="*")
    p_r2o.add_argument("--countries", nargs="*")
    p_r2o.set_defaults(func=_cmd_resqml2osdu)

    # ------------------------------------------------------------------
    # osdu2resqml
    # ------------------------------------------------------------------
    p_o2r = sub.add_parser("osdu2resqml", help="Convert OSDU manifest -> RESQML EPC")
    p_o2r.add_argument("--manifest", required=True, help="Input OSDU manifest JSON path")
    p_o2r.add_argument("--output", "-o", required=True, help="Output RESQML EPC path")
    p_o2r.add_argument(
        "--add-over-under-burden",
        action="store_true",
        help="Add dummy Over/Underburden units at rank boundaries",
    )
    p_o2r.set_defaults(func=_cmd_osdu2resqml)

    # ------------------------------------------------------------------
    # ow2resqml
    # ------------------------------------------------------------------
    p_ow2r = sub.add_parser("ow2resqml", help="Convert OpenWorks (Excel/JSON) -> RESQML EPC")
    g_ow2r = p_ow2r.add_mutually_exclusive_group(required=True)
    g_ow2r.add_argument("--excel", help="OpenWorks/SMDA Excel path (e.g., smda-api_strat-units.xlsx)")
    g_ow2r.add_argument("--json", help="OpenWorks JSON path")
    p_ow2r.add_argument("--sheet", default="ApiStratUnit", help="Excel sheet name")
    p_ow2r.add_argument("--output", "-o", required=True, help="Output RESQML EPC path")
    p_ow2r.add_argument(
        "--add-over-under-burden",
        action="store_true",
        help="Add dummy Over/Underburden units at rank boundaries",
    )
    p_ow2r.set_defaults(func=_cmd_ow2resqml)

    # ------------------------------------------------------------------
    # ow2osdu
    # ------------------------------------------------------------------
    p_ow2o = sub.add_parser("ow2osdu", help="Convert OpenWorks (Excel/JSON) -> OSDU manifest")
    g_ow2o = p_ow2o.add_mutually_exclusive_group(required=True)
    g_ow2o.add_argument("--excel", help="OpenWorks/SMDA Excel path (e.g., smda-api_strat-units.xlsx)")
    g_ow2o.add_argument("--json", help="OpenWorks JSON path")
    p_ow2o.add_argument("--sheet", default="ApiStratUnit", help="Excel sheet name")
    p_ow2o.add_argument("--output", "-o", required=True, help="Output OSDU manifest JSON path")
    p_ow2o.add_argument("--split-ranks", action="store_true", help="Split ranks to separate WPC records")
    p_ow2o.add_argument("--emit-refdata", action="store_true", help="Emit ChronoStrat reference-data records")
    p_ow2o.add_argument("--chronostrat-rd", help="OSDU ChronoStrat reference-data manifest JSON to auto-link")
    p_ow2o.add_argument("--kind", default=OsduAdapter.DEFAULT_KIND, help="OSDU kind for WPC")
    p_ow2o.add_argument("--viewers", nargs="*")
    p_ow2o.add_argument("--owners", nargs="*")
    p_ow2o.add_argument("--legaltags", nargs="*")
    p_ow2o.add_argument("--countries", nargs="*")
    p_ow2o.set_defaults(func=_cmd_ow2osdu)

    # ------------------------------------------------------------------
    # make-occurrence (bonus)
    # ------------------------------------------------------------------
    p_occ = sub.add_parser(
        "make-occurrence",
        help="Create a RESQML obj_StratigraphicOccurrenceInterpretation XML (standalone part)",
    )
    p_occ.add_argument("--rank-uuid", required=True, help="UUID of StratigraphicColumnRankInterpretation to reference")
    p_occ.add_argument("--title", default="Stratigraphic Occurrence")
    p_occ.add_argument("--ordering", default="depth", choices=["depth", "age", "custom"])
    p_occ.add_argument("--output", "-o", required=True, help="Output XML path")
    p_occ.set_defaults(
        func=lambda a: print(
            f"[ok] Occurrence XML written: {write_resqml_occurrence_xml(a.output, a.title, a.rank_uuid, a.ordering)}"
        )
    )

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
