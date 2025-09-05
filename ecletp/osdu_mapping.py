
"""OSDU reference-data mapping for common Eclipse property keywords.

The mapping can be extended; each entry carries a canonical OSDU reference, a display name,
preferred property kind (RESQML), and a default unit of measure where applicable.
"""
from __future__ import annotations
from typing import Dict

OSDU_PROPERTY_MAP: Dict[str, dict] = {
    # Porosity (fraction)
    'PORO': {
        'osdu_ref': 'osdu:reference-data--PropertyNameType:Porosity:1.0.0',
        'osdu_name': 'Porosity',
        'display': 'Porosity',
        'resqml_property_kind': 'porosity',
        'uom': 'frac'
    },
    # Permeability in X (md)
    'PERMX': {
        'osdu_ref': 'osdu:reference-data--PropertyNameType:PermeabilityX:1.0.0',
        'osdu_name': 'Permeability X',
        'display': 'PERMX',
        'resqml_property_kind': 'permeability rock',
        'uom': 'mD'
    },
    'PERMY': {
        'osdu_ref': 'osdu:reference-data--PropertyNameType:PermeabilityY:1.0.0',
        'osdu_name': 'Permeability Y',
        'display': 'PERMY',
        'resqml_property_kind': 'permeability rock',
        'uom': 'mD'
    },
    'PERMZ': {
        'osdu_ref': 'osdu:reference-data--PropertyNameType:PermeabilityZ:1.0.0',
        'osdu_name': 'Permeability Z',
        'display': 'PERMZ',
        'resqml_property_kind': 'permeability rock',
        'uom': 'mD'
    },
    'PRESSURE': {
        'osdu_ref': 'osdu:reference-data--PropertyNameType:Pressure:1.0.0',
        'osdu_name': 'Pressure',
        'display': 'Pressure',
        'resqml_property_kind': 'pressure',
        'uom': 'Pa'
    },
    'SWAT': {
        'osdu_ref': 'osdu:reference-data--PropertyNameType:WaterSaturation:1.0.0',
        'osdu_name': 'Water Saturation',
        'display': 'SWAT',
        'resqml_property_kind': 'saturation',
        'uom': 'frac'
    },
    'SOIL': {
        'osdu_ref': 'osdu:reference-data--PropertyNameType:OilSaturation:1.0.0',
        'osdu_name': 'Oil Saturation',
        'display': 'SOIL',
        'resqml_property_kind': 'saturation',
        'uom': 'frac'
    },
    'SGAS': {
        'osdu_ref': 'osdu:reference-data--PropertyNameType:GasSaturation:1.0.0',
        'osdu_name': 'Gas Saturation',
        'display': 'SGAS',
        'resqml_property_kind': 'saturation',
        'uom': 'frac'
    },
    'ACTNUM': {
        'osdu_ref': 'osdu:reference-data--PropertyNameType:ActiveCell:1.0.0',
        'osdu_name': 'Active Cell',
        'display': 'ACTNUM',
        'resqml_property_kind': 'bool',
        'uom': 'Euc'
    }
}

DEFAULT_PROPERTY_ORDER = ['PORO','PERMX','PERMY','PERMZ','PRESSURE','SWAT','SOIL','SGAS','ACTNUM']
