#!/usr/bin/env python3
"""
Land use integration module for preprocessing pipeline
Generated: 2025-09-22 19:23:47
"""

import json
import numpy as np
from pathlib import Path

# Load land use data
LAND_USE_FILE = Path(__file__).parent / 'land_use_lookup.json'

def load_land_use_data():
    """Load land use lookup data"""
    if LAND_USE_FILE.exists():
        with open(LAND_USE_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_land_use_features(site_key):
    """Get land use features for a specific site"""
    data = load_land_use_data()
    if site_key in data:
        lu = data[site_key]
        return [
            lu['urban_percent'] / 100.0,      # Normalize to 0-1
            lu['forest_percent'] / 100.0,
            lu['agriculture_percent'] / 100.0,
            lu['impervious_percent'] / 100.0
        ]
    return [0.0, 0.0, 0.0, 0.0]  # Default if not found

# Feature names for documentation
LAND_USE_FEATURE_NAMES = [
    'urban_percent_norm',
    'forest_percent_norm', 
    'agriculture_percent_norm',
    'impervious_percent_norm'
]

# Integration with existing pipeline:
# Add these 4 features to your existing feature set
# Total features: 33 (current) + 4 (land use) = 37 features
