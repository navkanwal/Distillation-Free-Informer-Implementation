"""Separate public Energy training. Never writes to the internal Energy bundle."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset

from app.inference import Informer
from app.public_schema import (CATEGORICAL_COLUMNS, ENERGY_TARGET, FEATURE_COLUMNS,
                               MISSING, PROFILE, SEMANTICS, SEQUENCE_RULE, UNKNOWN, WINDOW)

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, predicted):
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Evaluation must contain only finite actuals and predictions.')
    return {'mae_kwh': float(np.mean(np.abs(actual-predicted))),
            'rmse_kwh': float(np.sqrt(np.mean((actual-predicted)**2)))}


def duration_seconds(values):
    parts = values.str.extract(r'^(\d+):([0-5]\d):([0-5]\d)$').astype(float)
    return parts[0]*3600 + parts[1]*60 + parts[2]


class PublicWindows(Dataset):
    def __init__(self, features, targets, positions):
        self.features, self.targets, self.positions = features, targets, positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        p = self.positions[index]
        return self.features[p-WINDOW:p], self.targets[p]


def prepare_data(csv_path):
    # Read only needed source columns. No personal/session/equipment IDs are loaded.
    usecols = ['Station Name', ENERGY_TARGET, 'Total Duration (hh:mm:ss)',
               'Charging Time (hh:mm:ss)', 'Port Type', 'Plug Type', 'End Date']
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, usecols=usecols)
    raw['_source_record'] = np.arange(1, len(raw)+1)  # private ordering only, never exported
    raw['_time'] = pd.to_datetime(raw['End Date'], format='%m/%d/%Y %H:%M', errors='coerce')
    raw = raw.sort_values(['Station Name', '_time', '_source_record'], kind='stable').reset_index(drop=True)
    frame = pd.DataFrame(index=raw.index)
    for c in CATEGORICAL_COLUMNS:
        frame[c] = raw[c].replace('', MISSING)
    frame[ENERGY_TARGET] = pd.to_numeric(raw[ENERGY_TARGET], errors='coerce')
    frame['Connection Duration (seconds)'] = duration_seconds(raw['Total Duration (hh:mm:ss)'])
    frame['Charging Duration (seconds)'] = duration_seconds(raw['Charging Time (hh:mm:ss)'])
    hour = raw['_time'].dt.hour + raw['_time'].dt.minute/60
    weekday = raw['_time'].dt.dayofweek
    for prefix, value, period in [('Completion Hour', hour, 24), ('Completion Weekday', weekday, 7)]:
        frame[prefix+' Sin'] = np.sin(2*np.pi*value/period)
        frame[prefix+' Cos'] = np.cos(2*np.pi*value/period)
    frame = frame[FEATURE_COLUMNS]
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    valid = np.isfinite(frame[numeric]).all(axis=1).to_numpy() & raw['_time'].notna().to_numpy()
    valid &= frame[ENERGY_TARGET].ge(0).to_numpy() & raw['Station Name'].ne('').to_numpy()
    valid &= frame['Charging Duration (seconds)'].le(frame['Connection Duration (seconds)']).to_numpy()
    train_end, validation_end = raw['_time'].quantile([.7,.85]).tolist()
    fit_mask = valid & raw['_time'].le(train_end).to_numpy()
    encoders = {}
    encoded = frame.copy()
    for c in CATEGORICAL_COLUMNS:
        # All station labels are public institutional registry data, not user IDs.
        vocab = frame[c] if c == 'Station Name' else frame.loc[fit_mask, c]
        encoder = LabelEncoder().fit(vocab.unique() if c == 'Station Name' else [*vocab.unique(),UNKNOWN,MISSING])
        encoders[c] = encoder
        frame[c] = frame[c].where(frame[c].isin(encoder.classes_),UNKNOWN)
        encoded[c] = encoder.transform(frame[c])
    scaler = MinMaxScaler().fit(encoded.loc[fit_mask,FEATURE_COLUMNS])
    target_scaler = MinMaxScaler().fit(frame.loc[fit_mask,[ENERGY_TARGET]])
    features = torch.from_numpy(scaler.transform(encoded).astype(np.float32))
    targets = torch.from_numpy(target_scaler.transform(frame[[ENERGY_TARGET]]).astype(np.float32))
    positions = []
    # Check preceding 24 public histories; target needs only Energy and time.
    target_valid = raw['_time'].notna().to_numpy() & np.isfinite(frame[ENERGY_TARGET]).to_numpy() & frame[ENERGY_TARGET].ge(0).to_numpy()
    for _, group in raw.groupby('Station Name',sort=False):
        idx = group.index.to_numpy()
        prior_complete = pd.Series(valid[idx]).rolling(WINDOW).sum().shift(1).eq(WINDOW).to_numpy()
        positions.extend(idx[prior_complete & target_valid[idx]].tolist())
    positions = np.asarray(positions,dtype=np.int64)
    t = raw.loc[positions,'_time']
    splits = {'train':positions[t.le(train_end).to_numpy()],
              'validation':positions[(t.gt(train_end)&t.le(validation_end)).to_numpy()],
              'test':positions[t.gt(validation_end).to_numpy()]}
    if not fit_mask.any() or any(len(p)==0 for p in splits.values()):
        raise ValueError('Every chronological split requires complete public windows.')
    return raw,frame,encoders,scaler,target_scaler,features,targets,splits,[str(train_end),str(validation_end)],valid


def predict_scaled(model,loader,device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x.to(device)).cpu().numpy() for x,_ in loader])


def train(csv_path,output_dir,epochs=20,patience=5,seed=42,batch_size=256):
    output_dir = Path(output_dir).resolve()
    if output_dir.name != PROFILE or (output_dir.exists() and any(output_dir.iterdir())):
        raise ValueError('Use a new empty energy_public_v1 directory; no existing artifacts may be overwritten.')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cpu')  # reproducible on audited platform
    raw,frame,encoders,scaler,target_scaler,features,targets,splits,cutoffs,valid = prepare_data(csv_path)
    loaders = {k:DataLoader(PublicWindows(features,targets,p),batch_size=batch_size,shuffle=k=='train') for k,p in splits.items()}
    config = dict(input_dim=len(FEATURE_COLUMNS),d_model=64,n_heads=4,dropout=.1,seq_len=WINDOW)
    model = Informer(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    criterion = torch.nn.MSELoss()
    best_loss,best_state,best_epoch,stale = float('inf'),None,0,0
    history = []
    print(json.dumps({'device':str(device),'windows':{k:len(p) for k,p in splits.items()},'cutoffs':cutoffs}),flush=True)
    for epoch in range(1,epochs+1):
        model.train(); total=0
        for x,y in loaders['train']:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)),y.to(device))
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total += loss.item()*len(x)
        predicted = predict_scaled(model,loaders['validation'],device)
        val_loss = float(np.mean((predicted-targets[splits['validation']].numpy())**2))
        history.append({'epoch':epoch,'train_mse_scaled':total/len(splits['train']),'validation_mse_scaled':val_loss})
        print(json.dumps(history[-1]),flush=True)
        if val_loss < best_loss:
            best_loss,best_state,best_epoch,stale = val_loss,copy.deepcopy(model.state_dict()),epoch,0
        else: stale+=1
        if stale >= patience: break
    model.load_state_dict(best_state)
    evaluation = {}
    train_rows = valid & raw['_time'].le(pd.Timestamp(cutoffs[0])).to_numpy()
    means = frame.loc[train_rows].groupby('Station Name')[ENERGY_TARGET].mean()
    global_mean = float(frame.loc[train_rows,ENERGY_TARGET].mean())
    for name in ['validation','test']:
        p = splits[name]
        actual = frame.loc[p,ENERGY_TARGET].to_numpy(float)
        predicted = target_scaler.inverse_transform(predict_scaled(model,loaders[name],device)).ravel()
        station_pred = raw.loc[p,'Station Name'].map(means).fillna(global_mean).to_numpy(float)
        evaluation[name] = {'model':metrics(actual,predicted),
                            'persistence_baseline':metrics(actual,frame.loc[p-1,ENERGY_TARGET].to_numpy(float)),
                            'station_mean_baseline':metrics(actual,station_pred),
                            'station_mean_global_fallback_count':int(raw.loc[p,'Station Name'].map(means).isna().sum()),
                            'prediction_range_kwh':[float(predicted.min()),float(predicted.max())]}
    packages = ['torch','numpy','pandas','scikit-learn','joblib','fastapi','pydantic','uvicorn']
    report = {'target_column':ENERGY_TARGET,'unit':'kWh','release_profile':PROFILE,
              'seed':seed,'random_seeds':{'python':seed,'numpy':seed,'torch':seed},
              'deterministic_algorithms':True,'device':str(device),'threads':4,
              'epochs_requested':epochs,'patience':patience,'batch_size':batch_size,'learning_rate':.001,
              'selected_epoch':best_epoch,'completed_epochs':len(history),'history':history,
              'source_rows':len(raw),'complete_public_rows':int(valid.sum()),
              'split_windows':{k:len(p) for k,p in splits.items()},
              'split_window_proportions':{k:len(p)/sum(map(len,splits.values())) for k,p in splits.items()},
              'nominal_timestamp_quantiles':[.7,.15,.15],'time_cutoffs':cutoffs,'time_column':'End Date',
              'test':evaluation['test']['model'],'validation':evaluation['validation'],
              'persistence_baseline':evaluation['test']['persistence_baseline'],
              'station_mean_baseline':evaluation['test']['station_mean_baseline'],
              'station_mean_global_fallback_count':evaluation['test']['station_mean_global_fallback_count'],
              'test_prediction_range_kwh':evaluation['test']['prediction_range_kwh'],
              'data_sha256':digest(csv_path),'training_script_sha256':digest(__file__),
              'source_hashes':{name:digest(ROOT/name) for name in ['app/public_schema.py','app/inference.py','app/preprocessing.py']},
              'feature_schema':[{'name':c,'datatype':'string' if c in CATEGORICAL_COLUMNS else 'float64',
                                 'preprocessing':'fresh LabelEncoder then MinMaxScaler' if c in CATEGORICAL_COLUMNS else 'numeric then fresh MinMaxScaler'} for c in FEATURE_COLUMNS],
              'python_version':platform.python_version(),'package_versions':{p:importlib.metadata.version(p) for p in packages},
              'prediction_semantics':SEMANTICS,'sequence_rule':SEQUENCE_RULE,
              'preprocessing':'Training-only complete-public-row scalers and equipment vocabularies; public station registry vocabulary. Durations parsed as seconds; cyclic End Date wall-clock hour and weekday. No sensitive columns loaded. No real histories exported.',
              'cohort_difference':'Public completeness replaces canonical 28-field completeness; target completeness requires Energy/time only. Cutoffs unchanged. Compare frozen internal model only on shared eligible test windows.',
              'evaluation_limitations':'One seed; adjacent rolling windows overlap. Not differential privacy, unseen-station evaluation, or fixed-origin forecasting. Test is used only after validation selection.'}
    metadata = {'schema_version':2,'release_profile':PROFILE,'feature_columns':FEATURE_COLUMNS,
                'categorical_columns':CATEGORICAL_COLUMNS,'target_column':ENERGY_TARGET,'target_unit':'kWh',
                'target_scaler':'target_scaler.joblib','input_window':WINDOW,'prediction_horizon':1,
                'input_dim':len(FEATURE_COLUMNS),'model_config':config,'training_data_sha256':report['data_sha256'],
                'prediction_semantics':SEMANTICS}
    output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'model_state_dict':best_state,'contract':metadata},output_dir/'informer_checkpoint.pth')
    joblib.dump(scaler,output_dir/'minmax_scaler.joblib')
    joblib.dump(target_scaler,output_dir/'target_scaler.joblib')
    joblib.dump(encoders,output_dir/'label_encoders.joblib')
    catalog = {'samples':[],'stationValues':{},'provenance':{'datasetSha256':report['data_sha256']},
               'sampleUnavailableReason':'Recorded session histories are intentionally excluded from the public release. Enter authorized completed-session observations.'}
    for name,value in [('feature_metadata.json',metadata),('training_report.json',report),('prediction_catalog.json',catalog)]:
        (output_dir/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    files = ['informer_checkpoint.pth','minmax_scaler.joblib','target_scaler.joblib','label_encoders.joblib','feature_metadata.json','training_report.json','prediction_catalog.json']
    (output_dir/'manifest.json').write_text(json.dumps({n:digest(output_dir/n) for n in files},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'test':report['test'],'persistence':report['persistence_baseline'],'station_mean':report['station_mean_baseline']}),flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'models'/PROFILE)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    train(args.csv,args.output,args.epochs,args.patience,args.seed)
