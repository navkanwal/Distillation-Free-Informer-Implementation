"""Source-only checks: no released weights, fitted objects, or raw CSV required."""
import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from app import public_main
from app.inference import InferenceService
from app.preprocessing import SequenceValidationError
from app.public_inference import PublicInferenceService
from app.public_preprocessing import PublicPreprocessingArtifacts
from app.public_schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, PROFILE, WINDOW


MEMBERS = ['informer_checkpoint.pth', 'feature_metadata.json', 'minmax_scaler.joblib',
           'target_scaler.joblib', 'label_encoders.joblib', 'training_report.json',
           'prediction_catalog.json']


@pytest.fixture
def validator():
    # Exercise row validation alone. Explicit test labels are not fitted vocabularies.
    obj = PublicPreprocessingArtifacts.__new__(PublicPreprocessingArtifacts)
    obj.feature_columns = FEATURE_COLUMNS.copy()
    obj.numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    obj.input_window = WINDOW
    obj.category_sets = {'Station Name': {'test-station', 'other-test-station'},
                         'Port Type': {'test-port'}, 'Plug Type': {'test-plug'}}
    return obj


def rows():
    return [{'Station Name': 'test-station', 'Energy (kWh)': float(i),
             'Connection Duration (seconds)': 7200., 'Charging Duration (seconds)': 3600.,
             'Port Type': 'test-port', 'Plug Type': 'test-plug',
             'Completion Hour Sin': 0., 'Completion Hour Cos': 1.,
             'Completion Weekday Sin': 0., 'Completion Weekday Cos': 1.} for i in range(WINDOW)]


def test_complete_structural_history_accepts_zero(validator):
    validator.validate_rows(rows())


@pytest.mark.parametrize('field', ['User ID', 'MAC Address', 'Driver Postal Code',
                                  'Plug In Event Id', 'System S/N', 'EVSE ID'])
def test_identifiers_rejected_without_artifacts(validator, field):
    history = rows()
    history[0][field] = 'forbidden-test-value'
    with pytest.raises(SequenceValidationError):
        validator.validate_rows(history)


@pytest.mark.parametrize('change', [
    lambda r: r.pop(),
    lambda r: r[0].pop('Energy (kWh)'),
    lambda r: r[0].update({'Energy (kWh)': True}),
    lambda r: r[0].update({'Energy (kWh)': float('inf')}),
    lambda r: r[0].update({'Energy (kWh)': -1}),
    lambda r: r[0].update({'Charging Duration (seconds)': 9000}),
    lambda r: r[0].update({'Completion Hour Cos': .5}),
    lambda r: r[0].update({'Port Type': 'unsupported-test-class'}),
    lambda r: r[-1].update({'Station Name': 'other-test-station'}),
])
def test_invalid_history_rejected_without_artifacts(validator, change):
    history = rows()
    change(history)
    with pytest.raises(SequenceValidationError):
        validator.validate_rows(history)


def invalid_bundle(tmp_path):
    # Integrity-only test bytes, never deserialized or used as a forecasting model.
    bundle = tmp_path / PROFILE
    bundle.mkdir()
    for name in MEMBERS:
        (bundle / name).write_bytes(b'invalid-test-artifact')
    manifest = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in MEMBERS}
    (bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return bundle


@pytest.mark.parametrize('missing', ['manifest.json', *MEMBERS])
def test_missing_member_prevents_startup_with_training_instructions(tmp_path, monkeypatch, missing):
    bundle = invalid_bundle(tmp_path)
    (bundle / missing).unlink()
    service = PublicInferenceService(bundle / 'informer_checkpoint.pth', 'cpu')
    monkeypatch.setattr(public_main, 'inference_service', service)
    with pytest.raises(RuntimeError, match='initial source-only release') as error:
        with TestClient(public_main.app):
            pytest.fail('Startup must fail before requests can be served.')
    assert 'training/train.py' in str(error.value)
    assert 'PUBLIC_MODEL_PATH' in str(error.value)
    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert not service.loaded and service.preprocessing is None


@pytest.mark.parametrize('problem', ['missing', 'internal', 'corrupt', 'incomplete_manifest'])
def test_startup_never_calls_internal_loader(tmp_path, monkeypatch, problem):
    def forbidden_load(_):
        pytest.fail('The public service must never invoke the internal loader.')
    monkeypatch.setattr(InferenceService, 'load', forbidden_load)
    bundle = tmp_path / PROFILE
    if problem == 'internal':
        bundle = tmp_path / 'energy_v1'
    elif problem in {'corrupt', 'incomplete_manifest'}:
        bundle = invalid_bundle(tmp_path)
        if problem == 'corrupt':
            (bundle / 'feature_metadata.json').write_text('{}', encoding='utf-8')
        else:
            (bundle / 'manifest.json').write_text('{}', encoding='utf-8')
    service = PublicInferenceService(bundle / 'informer_checkpoint.pth', 'cpu')
    monkeypatch.setattr(public_main, 'inference_service', service)
    with pytest.raises(RuntimeError, match='No internal/private model fallback'):
        with TestClient(public_main.app):
            pytest.fail('Invalid artifacts must prevent startup.')
    assert not service.loaded and service.preprocessing is None


def test_unloaded_api_never_serves_categories_or_forecasts(tmp_path, monkeypatch):
    service = PublicInferenceService(tmp_path / PROFILE / 'informer_checkpoint.pth', 'cpu')
    monkeypatch.setattr(public_main, 'inference_service', service)
    # No lifespan here: exercise the request guard independently of startup failure.
    client = TestClient(public_main.app)
    assert client.get('/prediction-contract').status_code == 503
    assert client.get('/sample-input', params={'station': 'test-station'}).status_code == 503
    assert client.post('/predict', json={'features': rows()}).status_code == 503


def test_invalid_request_does_not_echo_identifier():
    response = TestClient(public_main.app).post('/predict', json={'User ID': 'forbidden-test-value'})
    assert response.status_code == 422
    assert 'forbidden-test-value' not in response.text
