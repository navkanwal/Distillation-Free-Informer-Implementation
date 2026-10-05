// Source-defined bootstrap schema, with no fitted categories or recorded histories.
// The interface obtains category choices from the validated public API contract.
const featureColumns = [
  'Station Name', 'Energy (kWh)', 'Connection Duration (seconds)',
  'Charging Duration (seconds)', 'Port Type', 'Plug Type',
  'Completion Hour Sin', 'Completion Hour Cos',
  'Completion Weekday Sin', 'Completion Weekday Cos',
]
const categoricalColumns = ['Station Name', 'Port Type', 'Plug Type']

export const publicSchema = {
  inputWindow: 24,
  inputDim: featureColumns.length,
  featureColumns,
  numericColumns: featureColumns.filter((field) => !categoricalColumns.includes(field)),
  categoricalColumns,
  categoryOptions: Object.fromEntries(categoricalColumns.map((field) => [field, []])),
  stationOptions: [],
  fixedValues: {},
  stationValues: {},
  targetColumn: 'Energy (kWh)',
  predictionHorizon: 1,
  horizonLabel: 'Next completed session',
  energyPredictionAvailable: false,
  releaseProfile: 'energy_public_v1',
  sampleAvailable: false,
  sampleStations: [],
  sampleSource: null,
}
