"""Public API entrypoint. An absent/invalid public bundle prevents startup."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .public_inference import PublicInferenceService
from .public_schema import PROFILE, SEMANTICS
from .preprocessing import SequenceValidationError
from .schemas import PredictionRequest, PredictionResponse

BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_PATH=Path(os.getenv('PUBLIC_MODEL_PATH',str(BASE_DIR.parent/'models'/PROFILE/'informer_checkpoint.pth')))
inference_service=PublicInferenceService(MODEL_PATH,os.getenv('DEVICE','cpu'))
logger=logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_):
    try: inference_service.load()
    except Exception as exc:
        message = (
            f'Cannot start inference: the complete, valid {PROFILE} bundle is required at '
            f'{inference_service.model_path.parent}. Trained/fitted artifacts are not '
            'redistributed in the initial source-only release. From the repository root, '
            'follow README.md Training: run training/train.py --csv '
            '"data/ChargePoint Data CY20Q4.csv" --output local/retrain/energy_public_v1 '
            '--seed 42, then set PUBLIC_MODEL_PATH to the absolute path of that '
            'bundle\'s informer_checkpoint.pth. No internal/private model fallback is permitted. '
            f'Artifact error: {exc}'
        )
        logger.error(message)
        raise RuntimeError(message) from exc
    yield


app=FastAPI(title='Public EV next-session Energy forecast',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=[v.strip() for v in os.getenv(
    'FRONTEND_ORIGINS','http://localhost:5173,http://127.0.0.1:5173').split(',') if v.strip()],
    allow_methods=['GET','POST'],allow_headers=['Content-Type'])


@app.exception_handler(RequestValidationError)
async def invalid_request(_,exc):
    # Do not echo incoming values, including mistakenly submitted identifiers.
    return JSONResponse(status_code=422,content={'detail':'Provide exactly 24 complete public-schema observations.',
        'errors':[{'row':None,'feature':None,'code':'structure','message':'The request has an invalid structure.'}]})


def ready():
    if not inference_service.loaded or inference_service.preprocessing is None:
        raise HTTPException(503,'The public Energy model is unavailable.')
    return inference_service.preprocessing


@app.get('/health')
def health():
    return {'status':'ok','model_loaded':inference_service.loaded,'release_profile':PROFILE,
            'energy_prediction_available':inference_service.loaded}


@app.get('/prediction-contract')
def contract():
    result=ready().contract()
    result.update(stationValues={},sampleAvailable=False,sampleStations=[],sampleSource=None,
                  sampleUnavailableReason='Recorded session histories are excluded from this public release.')
    return result


@app.get('/sample-input')
def sample(station: str=Query(...)):
    if station not in ready().category_sets['Station Name']:
        raise HTTPException(422,'Unsupported station.')
    return JSONResponse(status_code=503,content={'code':'sample_unavailable',
        'detail':'Recorded histories are excluded; enter authorized completed-session observations.'})


@app.post('/predict',response_model=PredictionResponse)
def predict(request: PredictionRequest):
    ready()
    try: energy=inference_service.predict(request.features)
    except SequenceValidationError as exc:
        return JSONResponse(status_code=422,content={'detail':str(exc),'errors':exc.errors})
    return PredictionResponse(predicted_energy_kwh=energy)
