# config.py

ZONE_COORDS = {
    "AT": [13.404, 47.346], "BE": [4.476, 50.720], "CH": [8.035, 46.994],
    "CZ": [14.974, 49.736], "DE": [10.434, 51.493], "DK1": [9.959, 55.992],
    "FR": [2.000, 46.010], "NL": [5.898, 52.759], "PL": [19.981, 51.985],
}

COUNTRY_NAMES = {
    "AT": "Austria", 
    "BE": "Belgium", 
    "CH": "Switzerland",
    "CZ": "Czech Republic", 
    "DE": "Germany", 
    "DK1": "Denmark (Zealand)",
    "FR": "France", 
    "NL": "Netherlands", 
    "PL": "Poland"
}

CATEGORY_MAP = {
    'wind': 'Wind Onshore',
    'wind onshore': 'Wind Onshore',
    'wind offshore': 'Wind Offshore',
    'solar': 'Solar',
    'solar pv': 'Solar',
    'nuclear': 'Nuclear',
    'hydroelectric': 'Hydro',
    'hydro': 'Hydro',
    'pumped hydro': 'Hydro',
    'coal': 'Fossil Fuels',
    'gas': 'Fossil Fuels',
    'oil': 'Fossil Fuels',
    'biomass': 'Bioenergy',
    'geothermal': 'Other',
    'other-renewable': 'Other',
    'other': 'Other'
}

# Paul Tol's "Muted" Palette (Luminance-constrained & CVD safe)
GENERATION_COLORS = {
    'Wind Onshore': '#88CCEE',  # Cyan (Airy/Light)
    'Wind Offshore': '#44AA99', # Teal (Oceanic)
    'Solar': '#DDCC77',         # Sand (Muted Yellow, no glare)
    'Nuclear': '#CC6677',       # Rose (Soft Red)
    'Hydro': '#332288',         # Indigo (Deep Water)
    'Fossil Fuels': '#999933',  # Olive (Earth/Muted)
    'Bioenergy': '#AA4499',     # Purple (Distinct)
    'Other': '#888888'          # Neutral Grey
}