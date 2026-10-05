import { publicSchema as inputCatalog } from '../data/publicSchema.js'

export const predictionInputs = inputCatalog
export const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value)
const numericText = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i

export class PredictionValidationError extends Error {
  constructor(errors) {
    super(errors[0]?.message || 'Check the highlighted observations.')
    this.name = 'PredictionValidationError'
    this.errors = errors
  }
}

export function fieldMessage(row, feature, code) {
  const location = Number.isInteger(row) ? `Step T-${24 - row} → ` : ''
  const label = `${location}${feature || 'Historical input'}`
  if (code === 'missing') return `${label} is required.`
  if (code === 'category') return `${label} must be one of the supported values.`
  if (code === 'station') return `${label} must match the selected charging station.`
  if (code === 'number') return `${label} must be a finite number.`
  if (code === 'sequence_length') return 'Provide exactly 24 observations.'
  return `${label} has an invalid feature structure.`
}

// Only verified fixed or station metadata is prefilled. Measurements start blank.
export function createForecastForm(station = '', catalog = inputCatalog) {
  const defaults = { ...catalog.fixedValues, ...catalog.stationValues?.[station], 'Station Name': station }
  return {
    station,
    rows: Array.from({ length: catalog.inputWindow }, () => Object.fromEntries(
      catalog.featureColumns.map((feature) => [feature, defaults[feature] ?? '']),
    )),
  }
}

export function validateSequence(rows, station, catalog = inputCatalog, parseNumbers = false) {
  const errors = []
  const add = (row, feature, code) => errors.push({ row, feature, code, message: fieldMessage(row, feature, code) })
  if (!catalog.stationOptions.includes(station)) add(null, 'Station Name', station ? 'category' : 'missing')
  if (!Array.isArray(rows) || rows.length !== catalog.inputWindow) {
    add(null, null, 'sequence_length')
    throw new PredictionValidationError(errors)
  }
  const numeric = new Set(catalog.numericColumns)
  const columns = new Set(catalog.featureColumns)
  const features = rows.map((row, index) => {
    if (!isObject(row)) { add(index, null, 'structure'); return {} }
    if (Object.keys(row).some((key) => !columns.has(key))) add(index, null, 'structure')
    // Reconstruct in the saved feature order, never in caller object-key order.
    return Object.fromEntries(catalog.featureColumns.map((feature) => {
      let value = row[feature]
      if (value === undefined || value === null || (typeof value === 'string' && !value.trim())) {
        add(index, feature, 'missing')
      } else if (numeric.has(feature)) {
        if (parseNumbers && typeof value === 'string' && numericText.test(value.trim())) value = Number(value)
        if (typeof value !== 'number' || !Number.isFinite(value)) add(index, feature, 'number')
      } else if (typeof value !== 'string' || !catalog.categoryOptions[feature]?.includes(value)) {
        add(index, feature, 'category')
      } else if (feature === 'Station Name' && value !== station) {
        add(index, feature, 'station')
      }
      return [feature, value]
    }))
  })
  features.forEach((row, index) => {
    for (const field of catalog.numericColumns) {
      const value = row[field]
      if (typeof value === 'number' && Number.isFinite(value)) {
        if (field.endsWith(' Sin') || field.endsWith(' Cos')) {
          if (value < -1 || value > 1) add(index, field, 'number')
        } else if (value < 0) add(index, field, 'number')
      }
    }
    if (Number(row['Charging Duration (seconds)']) > Number(row['Connection Duration (seconds)'])) add(index, 'Charging Duration (seconds)', 'number')
    for (const prefix of ['Completion Hour', 'Completion Weekday']) {
      const sin = row[prefix + ' Sin'], cos = row[prefix + ' Cos']
      if (typeof sin === 'number' && typeof cos === 'number' && Math.abs(sin * sin + cos * cos - 1) > .002) add(index, prefix + ' Sin', 'number')
    }
  })
  if (errors.length) throw new PredictionValidationError(errors)
  return { features }
}

export function buildForecastFormPayload(form = {}, catalog = inputCatalog) {
  return validateSequence(form.rows, form.station, catalog, true)
}

export function formFromSample(sample, catalog = inputCatalog) {
  const station = sample?.features?.[0]?.['Station Name']
  const payload = validateSequence(sample?.features, station, catalog)
  return { station, rows: payload.features }
}
