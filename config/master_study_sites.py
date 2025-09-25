"""
Master list of study sites. Keys are USGS site IDs; values hold metadata.
Add additional sites as needed following the same schema.
"""

MASTER_STUDY_SITES = {
    '03479000': {
        'name': 'Watauga River, NC',
        'full_name': 'Watauga River, NC',
        'state': 'NC',
        'lat': 36.215,
        'lon': -81.687,
        'biome': 'Appalachian mixed forest',
        'nwm_comid': 19743430,  # From previous configurations
        'usgs_id': '03479000',
        'region': 'Southeast',
        'regulation_status': 'Unregulated',
        'data_sources': ['ERA5', 'NWM', 'USGS', 'NLCD'],
    },
}
