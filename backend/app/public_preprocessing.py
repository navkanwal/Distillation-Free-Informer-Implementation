"""Public-only integrity, schema and value checks. No internal artifact fallback."""
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np

from .preprocessing import PreprocessingArtifacts, SequenceValidationError
from .public_schema import CATEGORICAL_COLUMNS, ENERGY_TARGET, FEATURE_COLUMNS, PROFILE, SEMANTICS, WINDOW


class PublicPreprocessingArtifacts(PreprocessingArtifacts):
    def __init__(self, models_dir: Path):
        if models_dir.name != PROFILE:
            raise ValueError('The public API only accepts energy_public_v1 artifacts.')
        required = {'informer_checkpoint.pth','feature_metadata.json','minmax_scaler.joblib',
                    'target_scaler.joblib','label_encoders.joblib','training_report.json','prediction_catalog.json'}
        manifest = json.loads((models_dir/'manifest.json').read_text(encoding='utf-8'))
        if set(manifest) != required:
            raise ValueError('The public model manifest is incomplete.')
        for name in required:
            if hashlib.sha256((models_dir/name).read_bytes()).hexdigest() != manifest[name]:
                raise ValueError('The public model failed its integrity check.')
        self.metadata = json.loads((models_dir/'feature_metadata.json').read_text(encoding='utf-8'))
        if (self.metadata.get('schema_version') != 2 or self.metadata.get('release_profile') != PROFILE
                or self.metadata.get('feature_columns') != FEATURE_COLUMNS
                or self.metadata.get('categorical_columns') != CATEGORICAL_COLUMNS
                or self.metadata.get('input_dim') != len(FEATURE_COLUMNS)
                or self.metadata.get('input_window') != WINDOW
                or self.metadata.get('target_column') != ENERGY_TARGET
                or self.metadata.get('target_unit') != 'kWh'
                or self.metadata.get('prediction_horizon') != 1
                or self.metadata.get('prediction_semantics') != SEMANTICS):
            raise ValueError('The public Energy feature contract is invalid.')
        report = json.loads((models_dir/'training_report.json').read_text(encoding='utf-8'))
        if (report.get('release_profile') != PROFILE or report.get('target_column') != ENERGY_TARGET
                or report.get('unit') != 'kWh' or report.get('completed_epochs',0) < 1
                or report.get('data_sha256') != self.metadata.get('training_data_sha256')
                or [f['name'] for f in report.get('feature_schema',[])] != FEATURE_COLUMNS):
            raise ValueError('The public Energy training provenance is invalid.')
        catalog = json.loads((models_dir/'prediction_catalog.json').read_text(encoding='utf-8'))
        if catalog.get('samples') != [] or catalog.get('stationValues') != {}:
            raise ValueError('Public artifacts must not export recorded histories or unreviewed station attributes.')
        self.scaler = joblib.load(models_dir/'minmax_scaler.joblib')
        self.target_scaler = joblib.load(models_dir/'target_scaler.joblib')
        self.label_encoders = joblib.load(models_dir/'label_encoders.joblib')
        if (list(self.scaler.feature_names_in_) != FEATURE_COLUMNS
                or list(self.target_scaler.feature_names_in_) != [ENERGY_TARGET]
                or self.target_scaler.n_features_in_ != 1
                or set(self.label_encoders) != set(CATEGORICAL_COLUMNS)):
            raise ValueError('Public preprocessing artifacts do not match the allowlist.')
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.categorical_columns = CATEGORICAL_COLUMNS.copy()
        self.numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
        self.input_window,self.input_dim,self.target_column = WINDOW,len(FEATURE_COLUMNS),ENERGY_TARGET
        self.category_options = {c:[str(v) for v in e.classes_] for c,e in self.label_encoders.items()}
        self.category_sets = {c:set(v) for c,v in self.category_options.items()}
        self.verified_energy_bundle = False

    def validate_rows(self, rows):
        super().validate_rows(rows)
        errors=[]
        for i,row in enumerate(rows):
            for c in self.numeric_columns:
                v=row[c]
                legal = -1 <= v <= 1 if c.endswith((' Sin',' Cos')) else v >= 0
                if not legal:
                    errors.append(self.issue(i,c,'number','Use nonnegative measurements and cyclic values between -1 and 1.'))
            if row['Charging Duration (seconds)'] > row['Connection Duration (seconds)']:
                errors.append(self.issue(i,'Charging Duration (seconds)','number','Charging duration cannot exceed connection duration.'))
            for prefix in ['Completion Hour','Completion Weekday']:
                if not math.isclose(row[prefix+' Sin']**2+row[prefix+' Cos']**2,1,abs_tol=.002):
                    errors.append(self.issue(i,prefix+' Sin','number','The sine and cosine must represent the same cyclic time.'))
        if errors: raise SequenceValidationError(errors)

    def contract(self):
        result=super().contract()
        result.update(releaseProfile=PROFILE,predictionSemantics=SEMANTICS,
                      horizonLabel='Next completed session')
        return result
