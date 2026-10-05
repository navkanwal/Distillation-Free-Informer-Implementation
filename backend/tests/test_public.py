"""Public-only inference/validation tests; constructed inputs are not dataset histories."""
import json
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.public_inference import PublicInferenceService
from app.public_main import app, inference_service
from app.public_schema import FEATURE_COLUMNS, PROFILE
from app.preprocessing import SequenceValidationError

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=inference_service.model_path.parent
pytestmark=pytest.mark.artifacts


@pytest.fixture(scope='module')
def service():
    required = ['manifest.json','informer_checkpoint.pth','feature_metadata.json','minmax_scaler.joblib',
                'target_scaler.joblib','label_encoders.joblib','training_report.json','prediction_catalog.json']
    if any(not (BUNDLE/name).is_file() for name in required):
        pytest.skip('Regenerate a complete energy_public_v1 bundle before artifact integration tests.')
    obj=PublicInferenceService(BUNDLE/'informer_checkpoint.pth','cpu'); obj.load()
    return obj


def constructed_rows(service):
    # Arbitrary structural measurements, never represented as real sessions.
    p=service.preprocessing
    return [{'Station Name':p.category_options['Station Name'][0], 'Energy (kWh)':float(i),
             'Connection Duration (seconds)':7200.,'Charging Duration (seconds)':3600.,
             'Port Type':p.category_options['Port Type'][0],'Plug Type':p.category_options['Plug Type'][0],
             'Completion Hour Sin':0.,'Completion Hour Cos':1.,
             'Completion Weekday Sin':0.,'Completion Weekday Cos':1.} for i in range(24)]


def test_public_schema_and_fresh_artifacts(service):
    assert service.preprocessing.feature_columns==FEATURE_COLUMNS
    assert service.preprocessing.input_dim==10
    assert set(service.preprocessing.label_encoders)=={'Station Name','Port Type','Plug Type'}
    assert service.preprocessing.metadata['target_column']=='Energy (kWh)'
    report=json.loads((BUNDLE/'training_report.json').read_text())
    assert report['data_sha256']==service.preprocessing.metadata['training_data_sha256']
    assert report['package_versions'] and report['random_seeds']
    assert np.isfinite(service.predict(constructed_rows(service)))


@pytest.mark.parametrize('field',['User ID','MAC Address','Driver Postal Code','Plug In Event Id','System S/N','EVSE ID'])
def test_sensitive_predictors_rejected(service,field):
    rows=constructed_rows(service); rows[0][field]='forbidden-test-value'
    with pytest.raises(SequenceValidationError): service.predict(rows)


@pytest.mark.parametrize('change',[
    lambda r:r.pop(),
    lambda r:r[0].pop('Energy (kWh)'),
    lambda r:r[0].update({'Energy (kWh)':True}),
    lambda r:r[0].update({'Energy (kWh)':float('inf')}),
    lambda r:r[0].update({'Energy (kWh)':-1}),
    lambda r:r[0].update({'Charging Duration (seconds)':9000}),
    lambda r:r[0].update({'Completion Hour Cos':.5}),
    lambda r:r[0].update({'Port Type':'unsupported-test-class'}),
])
def test_invalid_operational_inputs_rejected(service,change):
    rows=constructed_rows(service); change(rows)
    with pytest.raises(SequenceValidationError): service.predict(rows)


def test_cross_station_history_rejected(service):
    rows=constructed_rows(service); rows[-1]['Station Name']=service.preprocessing.category_options['Station Name'][1]
    with pytest.raises(SequenceValidationError): service.predict(rows)


def test_internal_bundle_and_missing_public_bundle_fail_closed(tmp_path):
    obj=PublicInferenceService(tmp_path/'energy_v1/informer_checkpoint.pth','cpu')
    with pytest.raises(ValueError,match='only accepts'): obj.load()
    assert not obj.loaded and obj.preprocessing is None
    obj=PublicInferenceService(tmp_path/PROFILE/'informer_checkpoint.pth','cpu')
    with pytest.raises(FileNotFoundError): obj.load()
    assert not obj.loaded and obj.preprocessing is None


def test_manifest_corruption_fail_closed(tmp_path,service):
    target=tmp_path/PROFILE; shutil.copytree(BUNDLE,target)
    (target/'feature_metadata.json').write_text('{}')
    obj=PublicInferenceService(target/'informer_checkpoint.pth','cpu')
    with pytest.raises(ValueError,match='integrity'): obj.load()
    assert not obj.loaded and obj.preprocessing is None


def test_public_api_no_identifier_contract_no_samples(service):
    with TestClient(app) as client:
        contract=client.get('/prediction-contract').json()
        assert contract['featureColumns']==FEATURE_COLUMNS and contract['releaseProfile']==PROFILE
        assert not contract['sampleAvailable'] and not contract['sampleStations']
        station=contract['stationOptions'][0]
        assert client.get('/sample-input',params={'station':station}).json()['code']=='sample_unavailable'
        rows=constructed_rows(service)
        response=client.post('/predict',json={'features':rows})
        assert response.status_code==200 and response.json()['unit']=='kWh'
        rows[0]['User ID']='forbidden-test-value'
        assert client.post('/predict',json={'features':rows}).status_code==422
        response=client.post('/predict',json={'features':constructed_rows(service),'User ID':'forbidden-test-value'})
        assert response.status_code==422 and 'forbidden-test-value' not in response.text


def test_api_unavailable_never_serves_catalog_from_internal_bundle(monkeypatch,service):
    with TestClient(app) as client:
        monkeypatch.setattr(inference_service,'model',None)
        monkeypatch.setattr(inference_service,'preprocessing',None)
        assert client.get('/prediction-contract').status_code==503
        assert client.get('/sample-input',params={'station':'anything'}).status_code==503
