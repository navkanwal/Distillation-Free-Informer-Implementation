"""Exact allowlist for the separate public-release bundle; no identifier mappings."""
ENERGY_TARGET = 'Energy (kWh)'
WINDOW = 24
PROFILE = 'energy_public_v1'
FEATURE_COLUMNS = [
    'Station Name', ENERGY_TARGET, 'Connection Duration (seconds)',
    'Charging Duration (seconds)', 'Port Type', 'Plug Type',
    'Completion Hour Sin', 'Completion Hour Cos',
    'Completion Weekday Sin', 'Completion Weekday Cos',
]
CATEGORICAL_COLUMNS = ['Station Name', 'Port Type', 'Plug Type']
SEMANTICS = 'Energy (kWh) of the next completed session at one known station, using its previous 24 consecutive completed sessions; not next-hour load or instantaneous power.'
SEQUENCE_RULE = '24 consecutive public-complete station histories, sorted by End Date then source record; target is following Energy. Invalid histories break windows; target features are not inputs.'
UNKNOWN = '__UNKNOWN__'
MISSING = '__MISSING__'
