import assert from 'node:assert/strict'
import test from 'node:test'
import { predictionInputs, createForecastForm, buildForecastFormPayload } from '../src/services/predictionInputs.js'
import { predictionService } from '../src/services/predictionService.js'

// Explicit structural test categories; not fitted vocabularies or real sessions.
const catalog = {
  ...predictionInputs,
  categoryOptions: { 'Station Name': ['test-station'], 'Port Type': ['test-port'], 'Plug Type': ['test-plug'] },
  stationOptions: ['test-station'],
  energyPredictionAvailable: true,
}

test('source bootstrap has no fitted categories and cannot submit a forecast', async(t) => {
  assert.deepEqual(predictionInputs.stationOptions, [])
  for (const options of Object.values(predictionInputs.categoryOptions)) assert.deepEqual(options, [])
  assert.equal(predictionInputs.energyPredictionAvailable, false)
  assert.equal(createForecastForm().rows.length, 24)
  const fetch = t.mock.method(globalThis, 'fetch', () => { throw new Error('unexpected request') })
  await assert.rejects(predictionService.predict(buildForecastFormPayload(constructed(), catalog)))
  assert.equal(fetch.mock.callCount(), 0)
})

test('categories come from a compatible public API contract', async(t) => {
  t.mock.method(globalThis, 'fetch', async() => ({ ok: true, json: async() => catalog }))
  assert.deepEqual(await predictionService.getContract(), catalog)
})

function constructed() {
  const form=createForecastForm(catalog.stationOptions[0],catalog)
  for (const [i,row] of form.rows.entries()) {
    Object.assign(row,{'Energy (kWh)':String(i),'Connection Duration (seconds)':'7200',
      'Charging Duration (seconds)':'3600','Port Type':catalog.categoryOptions['Port Type'][0],
      'Plug Type':catalog.categoryOptions['Plug Type'][0],
      'Completion Hour Sin':0,'Completion Hour Cos':1,'Completion Weekday Sin':0,'Completion Weekday Cos':1})
  }
  return form
}

test('public contract contains exactly ten allowlisted predictors and no recorded samples',()=>{
  assert.equal(catalog.inputDim,10)
  assert.equal(catalog.releaseProfile,'energy_public_v1')
  assert.equal(catalog.sampleAvailable,false)
  assert.deepEqual(catalog.sampleStations,[])
  for(const field of ['User ID','MAC Address','Driver Postal Code','Plug In Event Id','EVSE ID','System S/N']) {
    assert.ok(!catalog.featureColumns.includes(field))
    assert.ok(!Object.hasOwn(catalog.categoryOptions,field))
  }
})

test('measurements start empty and a complete constructed request preserves zero Energy',()=>{
  const empty=createForecastForm(catalog.stationOptions[0],catalog)
  assert.equal(empty.rows[0]['Energy (kWh)'],'')
  assert.throws(()=>buildForecastFormPayload(empty,catalog))
  const payload=buildForecastFormPayload(constructed(),catalog)
  assert.equal(payload.features.length,24)
  assert.equal(payload.features[0]['Energy (kWh)'],0)
  assert.deepEqual(Object.keys(payload.features[0]),catalog.featureColumns)
})

test('identifier injection and missing histories fail before transport',async(t)=>{
  const fetch=t.mock.method(globalThis,'fetch',()=>{throw new Error('unexpected request')})
  const form=constructed(); form.rows[0]['User ID']='forbidden-test-value'
  assert.throws(()=>buildForecastFormPayload(form,catalog))
  await assert.rejects(predictionService.predict({features:[]},catalog))
  assert.equal(fetch.mock.callCount(),0)
})

test('internal API contract rejected by public transport',async(t)=>{
  t.mock.method(globalThis,'fetch',async()=>({ok:true,json:async()=>({...catalog,releaseProfile:'energy_v1',inputDim:28})}))
  await assert.rejects(predictionService.getContract(),/incompatible/)
})

test('invalid operational ranges and incompatible cyclic pairs rejected before POST',()=>{
  for (const change of [
    (r)=>{r['Energy (kWh)']=-1},
    (r)=>{r['Charging Duration (seconds)']=9000},
    (r)=>{r['Completion Hour Cos']=.5},
    (r)=>{r['Completion Weekday Sin']=2},
  ]) {
    const form=constructed(); change(form.rows[0])
    assert.throws(()=>buildForecastFormPayload(form,catalog))
  }
})

test('public transport submits typed 24 by 10 inputs and requires Energy units',async(t)=>{
  const payload=buildForecastFormPayload(constructed(),catalog)
  t.mock.method(globalThis,'fetch',async(url,opts)=>{
    assert.ok(url.endsWith('/predict')); assert.equal(opts.method,'POST')
    assert.deepEqual(JSON.parse(opts.body),payload)
    return {ok:true,json:async()=>({predicted_energy_kwh:1,target_column:'Energy (kWh)',unit:'kWh',prediction_horizon:1})}
  })
  assert.equal((await predictionService.predict(payload,catalog)).predicted_energy_kwh,1)
})
