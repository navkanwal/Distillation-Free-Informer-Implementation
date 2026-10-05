# Public Energy feature audit and schema decision

Phase A completed read-only on 2026-10-05 before implementation. Inspected the canonical trainer, saved encoders/scalers/metadata/checkpoint/report, raw dataset, API contract, frontend catalog, field groups, form validation and sample helper. Dataset: 259,415 rows, SHA-256 `e65f5f5d3861cdd6bf3da2de97aabe590d7708e001db1532202eec52081ba82c`. No source identifier values are reproduced here.

All canonical inputs are historical completed-session measurements, available only after completion. None is an attribute of the future target session. Every canonical field is requested once per observation in `ForecastView.jsx`, validated in `predictionInputs.js`, and exported through `inputCatalog.json` and `/prediction-contract`. The table's UI group specifies the dependent frontend field; the label is exactly the feature name. Station Name additionally controls the station selector and sample lookup. Station/equipment fields may receive catalog defaults. All fields flow into the 24-row POST. Tests and sample generation also depend on this contract.

Categorical preprocessing: missing string becomes `__MISSING__`, LabelEncoder uses training-period vocabulary plus missing/unknown tokens, then MinMaxScaler; station vocabulary uses all 47 registry names. Numeric preprocessing: coercion to float64, complete-row filtering and training-only MinMaxScaler. Unique counts below are full-source categorical counts (including empty string), followed by saved encoder class counts including sentinels. Numeric unique counts are not categorical vocabularies.

| Feature / frontend field | Canonical type | Preprocessing | Categorical uniques source / encoder | Publication-sensitive? | Real inference meaning | Leakage / memorization risk | Recommendation |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| Station Name | string | LabelEncoder + minmax; registry | 47 / 47 | Institutional infrastructure label, not driver ID; review deployment inventory | Historical station identity; station selector, equipment group | Station-specific memorization; no unseen-station evaluation | KEEP |
| MAC Address | string | LabelEncoder + minmax | 83 / 44 | Yes; device/network identifier, even if charger-associated | Hardware identity; equipment group | Stable unique device lookup/proxy | REMOVE FROM PUBLIC MODEL |
| Start Time Zone | string | LabelEncoder + minmax | 3 / 5 | No | Historical zone; session details | Low; redundant after time conversion | REVIEW |
| Total Duration (hh:mm:ss) | string | LabelEncoder + minmax | 31,025 / 22,864 | Operational value; exact histories can be linkable | Historical connection duration; measurements | Huge vocabulary, false ordinal distances, unseen durations | KEEP, convert to seconds |
| Charging Time (hh:mm:ss) | string | LabelEncoder + minmax | 22,473 / 17,974 | Operational value; exact histories can be linkable | Historical active charging duration; measurements | Huge vocabulary, false ordinal distances | KEEP, convert to seconds |
| Energy (kWh) | float64 | numeric + minmax | — | Operational value; do not publish session histories | Previous delivered Energy; measurements | Legitimate autoregression; leaks target if future row is supplied | KEEP |
| GHG Savings (kg) | float64 | numeric + minmax | — | No direct identifier | Historical derived savings; measurements | Often redundant Energy-derived proxy | REVIEW |
| Gasoline Savings (gallons) | float64 | numeric + minmax | — | No direct identifier | Historical derived savings; measurements | Often redundant Energy-derived proxy | REVIEW |
| Port Type | string | LabelEncoder + minmax | 3 / 4 | Public equipment class | Historical charger type; equipment | Low cardinality; ordinal encoding limitation | KEEP |
| Port Number | float64 | numeric + minmax | — | Station-local equipment label; not driver ID | Historical port index; equipment | Confounded with station/device | REVIEW |
| Plug Type | string | LabelEncoder + minmax | 2 / 4 | Public connector class | Historical plug specification; equipment | Low cardinality | KEEP |
| EVSE ID | float64 | numeric + minmax | — | Unique equipment identifier; not proven necessary/public | Historical device key; equipment | Equipment lookup proxy; numeric distances meaningless | REMOVE FROM PUBLIC MODEL |
| Address 1 | string | LabelEncoder + minmax | 20 / 17 | Station infrastructure address, distinct from driver address | Station location; equipment | Station proxy; duplicate aliases | REVIEW |
| City | string | LabelEncoder + minmax | 1 / 3 | Public station city | Station location; equipment | Constant, no useful variation | REVIEW |
| State/Province | string | LabelEncoder + minmax | 1 / 3 | Public station region | Station location; equipment | Constant | REVIEW |
| Postal Code | float64 | numeric + minmax | — | Station postal code, distinct from Driver Postal Code | Station location; equipment | Ordinal postcode distances inappropriate; redundant station proxy | REVIEW |
| Country | string | LabelEncoder + minmax | 1 / 3 | Public station country | Station location; equipment | Constant | REVIEW |
| Latitude | float64 | numeric + minmax | — | Station coordinates; review public registry | Station location; equipment | Fine spatial proxy; new-station transfer untested | REVIEW |
| Longitude | float64 | numeric + minmax | — | Station coordinates; review public registry | Station location; equipment | Fine spatial proxy; new-station transfer untested | REVIEW |
| Currency | string | LabelEncoder + minmax | 5 / 5 | No direct identifier | Historical billing currency; session details | Constant/dirty categories, little signal | REVIEW |
| Fee | float64 | numeric + minmax | — | Individual historical transaction amount; linkable | Historical charge; measurements | Energy/tariff-derived proxy, rare values | REVIEW |
| Ended By | string | LabelEncoder + minmax | 17 / 14 | Termination category may encode user/operator provenance | Historical termination; session details | User-behavior proxy; future termination would leak outcome | REVIEW |
| Plug In Event Id | float64 | numeric + minmax | — | Yes; session/event identifier | Historical database key; session details | Near-unique session lookup/time-order proxy | REMOVE FROM PUBLIC MODEL |
| Driver Postal Code | float64 | numeric + minmax | — | Yes; personal geographic attribute | Historical driver's home area; session details | Driver geography/proxy memorization | REMOVE FROM PUBLIC MODEL |
| User ID | string | LabelEncoder + minmax | 21,442 / 10,590 | Yes; persistent individual identifier | Historical user identity; session details | User lookup/memorization and unseen-user issues | REMOVE FROM PUBLIC MODEL |
| County | string | LabelEncoder + minmax | 3 / 4 | Station region; blanks common | Station region; equipment | Redundant/missing constant metadata | REVIEW |
| System S/N | float64 | numeric + minmax | — | Unique equipment serial, not proven public/needed | Historical hardware identity; equipment | Unique equipment lookup proxy | REMOVE FROM PUBLIC MODEL |
| Model Number | string | LabelEncoder + minmax | 11 / 11 | Equipment model class; not inherently personal | Charger model; equipment | Station-generation proxy; many missing values | REVIEW |

Station assessment: the City of Palo Alto officially lists the matching dataset as infrastructure data: https://data.paloalto.gov/dataviews/257812/electric-vehicle-charging-station-usage-july-2011-dec-2020/ . All local station labels share an institutional namespace. This supports keeping institutional station labels separately from driver identifiers, but does not establish that every historical charger was publicly accessible, or grant a license for all derived artifacts. No person/device identifiers are recoded, hashed, or retained as public predictors.

## Phase B: selected public schema

Exactly 10 fields in saved order: `Station Name`, `Energy (kWh)`, `Connection Duration (seconds)`, `Charging Duration (seconds)`, `Port Type`, `Plug Type`, `Completion Hour Sin`, `Completion Hour Cos`, `Completion Weekday Sin`, `Completion Weekday Cos`.

Station/port/plug classes use fresh training vocabularies and fresh scalers. Durations are parsed arithmetically, never label encoded. Hour/weekday are coarse cyclic features from the recorded End Date wall clock, at minute resolution for hour. They are not UTC-normalized because the canonical End Date ordering is naive; this limitation is explicit. No exact session date or source-row key is exported as an input. Operational values still require access-controlled handling in production; no real per-session histories will ship in public examples/catalogs.

Removed identifier fields: User ID, Driver Postal Code, Plug In Event Id, MAC Address, EVSE ID, System S/N. Removed optional fields: savings proxies duplicate historical Energy; Fee adds transaction detail; Ended By adds termination/behavior information; Currency and the city/state/country/county fields are constant or redundant; addresses/postcodes/coordinates duplicate the retained station identity and need finer registry review; Port Number and Model Number add station/device proxies; Start Time Zone is not needed for the chosen wall-clock cyclic representation. Station geography and equipment model can be reconsidered in a later version with independent usefulness and disclosure evidence.

## Sequence and evaluation decision

Target remains next completed-session Energy (kWh), following 24 consecutive complete rows of one station, sorted by End Date then source record as the canonical pipeline does. Invalid public measurements break windows; no skipped/imputed rows. Target-row features are never included. Only target-row Energy and completion ordering/completeness determine eligibility; future duration/port information is not required. This deliberate change avoids requiring future/session attributes at forecast time.

Chronological cutoffs remain the canonical 70%/85% timestamp quantiles over the exact source dataset; vocabulary/scalers fit complete public training-history rows only. Completeness now depends on public fields rather than missing identifiers, increasing the usable cohort. Report realized window proportions separately from nominal 70/15/15 record-time proportions. Adjacent windows overlap and later evaluation histories may include earlier observed test sessions: rolling one-session forecasting, not a fixed-origin multi-step forecast. Compare the public and frozen internal models on exactly the canonical eligible test windows in a separate local-only evaluation; do not copy old metrics or mix cohort scores.

