import { useEffect, useState } from 'react'
import { buildForecastFormPayload, createForecastForm, predictionInputs } from '../services/predictionInputs'
import { predictionService } from '../services/predictionService'

const measurements = ['Energy (kWh)', 'Connection Duration (seconds)', 'Charging Duration (seconds)']
const equipment = ['Port Type', 'Plug Type']

export function ForecastView() {
  const [catalog, setCatalog] = useState(predictionInputs)
  const [form, setForm] = useState(() => createForecastForm('', predictionInputs))
  const [page, setPage] = useState(0)
  const [ready, setReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  useEffect(() => {
    let active = true
    predictionService.getContract().then((next) => {
      if (active) { setCatalog(next); setForm(createForecastForm('', next)); setReady(true) }
    }).catch((failure) => { if (active) setError(failure.message) })
    return () => { active = false }
  }, [])
  function update(index, field, value) {
    setForm((current) => ({ ...current, rows: current.rows.map((row, i) => i === index ? { ...row, [field]: value } : row) }))
    setResult(null)
  }
  function changeStation(station) {
    setForm((current) => ({ station, rows: current.rows.map((row) => ({ ...row, 'Station Name': station })) }))
    setResult(null)
  }
  function time(index, prefix, value, period) {
    if (value === '') { update(index, `${prefix} Sin`, ''); update(index, `${prefix} Cos`, ''); return }
    if (!Number.isFinite(Number(value)) || Number(value) < 0 || Number(value) >= period) {
      setError('Use a completion hour from 0 up to 24, and a supported weekday.')
      update(index, `${prefix} Sin`, ''); update(index, `${prefix} Cos`, '')
      return
    }
    const angle = 2 * Math.PI * Number(value) / period
    update(index, `${prefix} Sin`, Math.sin(angle))
    update(index, `${prefix} Cos`, Math.cos(angle))
  }
  function timeValue(row, prefix, period) {
    if (row[`${prefix} Sin`] === '' || row[`${prefix} Cos`] === '') return ''
    const angle = Math.atan2(row[`${prefix} Sin`], row[`${prefix} Cos`])
    const value = (angle * period / (2 * Math.PI) + period) % period
    return prefix === 'Completion Weekday' ? Math.round(value) % 7 : Number(value.toFixed(6))
  }
  async function submit(event) {
    event.preventDefault(); setError(''); setResult(null); setBusy(true)
    try { setResult(await predictionService.predict(buildForecastFormPayload(form, catalog), catalog)) }
    catch (failure) { setError(failure.message) }
    finally { setBusy(false) }
  }
  return <>
    <div className="page-header"><div><div className="eyebrow">Public Energy model · Experimental</div><h1>Next-session Energy forecast</h1><p>Enter the previous 24 completed sessions at one station, oldest first. The result estimates Energy for the next completed session.</p></div></div>
    <p className="research-note">This single-seed research model is evaluated against persistence and station-mean baselines. See the released training report for results. It does not predict hourly load.</p>
    <form className="panel prediction-form" onSubmit={submit} noValidate>
      <div className="control-group"><label htmlFor="forecast-station">Charging station</label><select id="forecast-station" value={form.station} disabled={!ready || busy} onChange={(e) => changeStation(e.target.value)}><option value="">Choose a station</option>{catalog.stationOptions.map((s) => <option key={s} value={s}>{s}</option>)}</select></div>
      <p className="field-help">Use authorized completed-session observations. Recorded histories and sample autofill are excluded from the public release. Changing station preserves entered measurements; verify they belong to the selected station.</p>
      {error && <div role="alert" className="prediction-error">{error}</div>}
      <div className="sequence-pages">{[0, 1, 2].map((p) => <button key={p} type="button" aria-current={page === p ? 'page' : undefined} onClick={() => setPage(p)}>Sessions {p * 8 + 1}–{p * 8 + 8}</button>)}</div>
      {form.rows.slice(page * 8, page * 8 + 8).map((row, local) => {
        const i = page * 8 + local
        return <fieldset className="observation" key={i}><legend>Session {i + 1} · T-{24 - i}</legend><div className="feature-grid">
          {measurements.map((field) => <div className="feature-control" key={field}><label htmlFor={`row-${i}-${field}`}>{field}</label><input id={`row-${i}-${field}`} type="number" min="0" step="any" value={row[field]} disabled={busy} onChange={(e) => update(i, field, e.target.value)} /></div>)}
          {equipment.map((field) => <div className="feature-control" key={field}><label htmlFor={`row-${i}-${field}`}>{field}</label><select id={`row-${i}-${field}`} value={row[field]} disabled={busy} onChange={(e) => update(i, field, e.target.value)}><option value="">Choose a value</option>{catalog.categoryOptions[field].map((v) => <option key={v} value={v}>{v}</option>)}</select></div>)}
          <div className="feature-control"><label htmlFor={`hour-${i}`}>Completion hour (local wall clock, 0–23.99)</label><input id={`hour-${i}`} type="number" min="0" max="23.999" step="any" value={timeValue(row, 'Completion Hour', 24)} disabled={busy} onChange={(e) => time(i, 'Completion Hour', e.target.value, 24)} /></div>
          <div className="feature-control"><label htmlFor={`weekday-${i}`}>Completion weekday</label><select id={`weekday-${i}`} value={timeValue(row, 'Completion Weekday', 7)} disabled={busy} onChange={(e) => time(i, 'Completion Weekday', e.target.value, 7)}><option value="">Choose a day</option>{['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].map((d,n) => <option key={d} value={n}>{d}</option>)}</select></div>
        </div></fieldset>
      })}
      <button className="primary-button" disabled={!ready || busy} type="submit">{busy ? 'Predicting…' : 'Predict Energy'}</button>
    </form>
    <section className="panel energy-result" aria-live="polite"><h2>Predicted Energy</h2>{result ? <p className="energy-value">{result.predicted_energy_kwh.toFixed(2)} <span>kWh</span></p> : <p>No forecast generated.</p>}<p>Next completed session · 24 completed sessions of history</p></section>
  </>
}
