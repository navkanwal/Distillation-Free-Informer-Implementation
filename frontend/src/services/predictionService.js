import { fieldMessage, isObject, predictionInputs, PredictionValidationError, validateSequence } from './predictionInputs.js'

const apiBaseUrl = (import.meta.env?.VITE_API_BASE_URL || 'http://127.0.0.1:8000').trim().replace(/\/+$/, '')

export function buildPredictionPayload(input, catalog = predictionInputs) {
  if (!isObject(input) || Object.keys(input).length !== 1 || !Array.isArray(input.features)) {
    throw new PredictionValidationError([{ row: null, feature: null, code: 'structure', message: 'Provide the complete 24-observation history.' }])
  }
  return validateSequence(input.features, input.features[0]?.['Station Name'], catalog)
}

function apiError(status, body, catalog) {
  // Build messages from known codes/feature names; never display arbitrary server text.
  const codes = new Set(['missing', 'category', 'station', 'number', 'structure', 'sequence_length'])
  const fieldErrors = Array.isArray(body?.errors) ? body.errors.flatMap((error) => {
    if (!isObject(error) || !codes.has(error.code)) return []
    const row = Number.isInteger(error.row) && error.row >= 0 && error.row < 24 ? error.row : null
    const feature = catalog.featureColumns.includes(error.feature) ? error.feature : null
    return [{ row, feature, code: error.code, message: fieldMessage(row, feature, error.code) }]
  }) : []
  if (status === 422 && fieldErrors.length) return new PredictionValidationError(fieldErrors)
  if (body?.code === 'target_mismatch') return new Error('Energy prediction is unavailable. The installed model was trained for a different target.')
  if (body?.code === 'sample_unavailable') return new Error('No complete, compatible 24-observation sample is available for this station. Enter observations manually or supply matching data and encoders.')
  if (status === 422) return new Error('The observations could not be accepted. Check all required values and try again.')
  if (status === 503) return new Error('The forecasting service is not ready. Please try again later.')
  return new Error('The prediction could not be completed. Please try again later.')
}

async function request(path, options = {}, catalog = predictionInputs) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 30000)
  try {
    const response = await fetch(`${apiBaseUrl}${path}`, { ...options, signal: controller.signal })
    let body
    try { body = await response.json() } catch (error) {
      if (error.name === 'AbortError' || error instanceof TypeError) throw error
      if (!response.ok) throw apiError(response.status, null, catalog)
      throw new Error('A valid response was not received. Please try again later.')
    }
    if (!response.ok) throw apiError(response.status, body, catalog)
    return body
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The forecasting service took too long to respond. Please try again later.')
    if (error instanceof TypeError) throw new Error('The forecasting service could not be reached. Please try again when it is available.')
    throw error
  } finally { clearTimeout(timeout) }
}

export const predictionService = {
  async getContract() {
    const body = await request('/prediction-contract')
    if (!isObject(body) || body.releaseProfile !== 'energy_public_v1' || body.inputWindow !== 24 || body.inputDim !== predictionInputs.inputDim
      || !Array.isArray(body.featureColumns) || body.featureColumns.length !== predictionInputs.inputDim
      || body.featureColumns.some((column, index) => column !== predictionInputs.featureColumns[index])
      || !Array.isArray(body.numericColumns) || !Array.isArray(body.categoricalColumns)
      || !Array.isArray(body.stationOptions) || !isObject(body.categoryOptions)
      || body.featureColumns.some((column) => {
        const number = body.numericColumns.includes(column)
        const category = body.categoricalColumns.includes(column)
        return number === category || (category && (!Array.isArray(body.categoryOptions[column])
          || body.categoryOptions[column].some((value) => typeof value !== 'string')))
      })) throw new Error('The forecasting service returned an incompatible input specification.')
    return body
  },

  async loadSampleInput(station) {
    return request(`/sample-input?station=${encodeURIComponent(station)}`)
  },

  async predict(input, catalog = predictionInputs) {
    const payload = buildPredictionPayload(input, catalog)
    const body = await request('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    }, catalog)
    if (!isObject(body) || typeof body.predicted_energy_kwh !== 'number' || !Number.isFinite(body.predicted_energy_kwh)
      || body.target_column !== 'Energy (kWh)' || body.unit !== 'kWh' || body.prediction_horizon !== 1) {
      throw new Error('A verified Energy prediction was not received. Please check the forecasting service.')
    }
    return {
      predicted_energy_kwh: body.predicted_energy_kwh,
      target_column: body.target_column, unit: body.unit, prediction_horizon: body.prediction_horizon,
    }
  },
}
