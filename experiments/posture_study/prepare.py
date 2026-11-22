"""Frozen, subject-disjoint study dataset; no production mutations."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, read_sensor_csv
from src.window_pipeline import WindowSpec, segment_sensor_frame, extract_window_features
from validate_personalization import audit_dataset, registration_plan

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/posture_study_20260913'
ACTIVE = ROOT / 'artifacts/models/integrated_20260908_121321_070b045a/artifacts'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def default(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default, allow_nan=False)+'\n')


def take(data, indices):
    return {k: v[indices] for k, v in data.items()}


def prepare(output=OUT):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/'data.npz').exists():
        manifest = json.loads((output/'dataset.json').read_text())
        for row in manifest['audit']['records']:
            assert digest(ROOT/'data'/row['relative_path']) == row['sha256'], 'Frozen data changed'
        for row in manifest['sidecars']:
            assert digest(ROOT/row['file']) == row['sha256'], 'Frozen sidecar changed'
        return dict(np.load(output/'data.npz', allow_pickle=False)), manifest
    records, audit = audit_dataset(ROOT/'data')
    if audit['duplicate_sensor_sequences'] or audit['duplicate_raw_files']:
        raise ValueError('Duplicate content requires identity resolution before experiment')
    baseline_evidence=json.loads((ACTIVE/'window_3s_rbf_svm_evaluation.json').read_text())
    partitions={k:v['subjects'] for k,v in baseline_evidence['partitions'].items()}
    all_subjects=sorted({r.subject for r in records})
    test=partitions['final_evaluation']; val=partitions['validation']
    train=sorted(set(all_subjects)-set(test)-set(val))
    parts=[{'id':'primary','seed':20260826,'train':train,'validation':val,'test':test}]
    dev=sorted(set(all_subjects)-set(test))
    # Two independent subject-disjoint validation resplits; held-out test unchanged.
    for i,seed in enumerate((20260913,20260914),1):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=len(val),random_state=seed).split(dev,groups=dev))
        parts.append({'id':f'robustness_{i}','seed':seed,'train':np.asarray(dev)[a].tolist(),
                      'validation':np.asarray(dev)[b].tolist(),'test':test})
    seq=[]; personal=[]; ys=[]; groups=[]; ids=[]; starts=[]; rows=[]; registrations=[]
    plans = registration_plan(records)
    if not plans:
        raise ValueError("등록용 자동 배정이 해제되어 개인 기준을 요구하는 과거 실험을 새로 준비할 수 없습니다. 일반 학습을 사용하세요.")
    for registration,evaluation in plans:
        frame=read_sensor_csv(registration.path)
        time=frame.elapsed_s.to_numpy()-frame.elapsed_s.iloc[0]
        reg=frame.loc[time<=10,SENSOR_COLUMNS].to_numpy(float)
        ref=reg.mean(0); ref_ratio=ref/max(ref.sum(),1e-6)
        registrations.append({**registration.evidence(),'use':'first10seconds_mean; entire record excluded',
                              'reference_mean':ref.tolist(),'reference_ratio':ref_ratio.tolist()})
        for record in evaluation:
            windows,meta=segment_sensor_frame(read_sensor_csv(record.path),WindowSpec(3))
            mean=windows.mean(1); ratio=mean/np.maximum(mean.sum(1,keepdims=True),1e-6)
            # All reference-derived quantities available from registration alone.
            pers=np.concatenate([np.tile(ref_ratio,(len(windows),1)), ratio-ref_ratio,
                 np.log((mean+1)/(ref+1)), ((mean.sum(1)+1)/(ref.sum()+1))[:,None]],axis=1)
            for w,p,m in zip(windows,pers,meta.to_dict('records')):
                seq.append(w);personal.append(p);ys.append(CLASS_NAMES.index(record.label));groups.append(record.subject)
                ids.append(record.relative_path);starts.append(m['start_s'])
                rows.append({'subject':record.subject,'record':record.relative_path,'sha256':record.sha256,
                             'sensor_sha256':record.sensor_sha256,'start_s':m['start_s'],'label':record.label})
    data={'seq':np.stack(seq),'personal':np.stack(personal).astype(np.float32),'y':np.array(ys),
          'groups':np.array(groups),'record_ids':np.array(ids),'start_s':np.array(starts)}
    data['features']=extract_window_features(data['seq']).astype(np.float32)
    for part in parts:
        assert not(set(part['train'])&set(part['validation']) or set(part['train'])&set(test) or set(part['validation'])&set(test))
        part['counts']={}
        for split in ('train','validation','test'):
            mask=np.isin(data['groups'],part[split])
            assert set(data['y'][mask])==set(range(7)), f'Missing class in {part["id"]}/{split}'
            part['counts'][split]={'windows':int(mask.sum()),'records':len(set(data['record_ids'][mask])),
                                  'subjects':len(part[split]),'class_counts':np.bincount(data['y'][mask],minlength=7).tolist()}
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'audit':audit,'registrations':registrations,
       'sidecars':[{'file':str(r.path.with_suffix('.manifest.json').relative_to(ROOT)),
                    'sha256':digest(r.path.with_suffix('.manifest.json'))} for r in records if r.path.with_suffix('.manifest.json').exists()],
       'partitions':parts,'rows':rows,'class_names':CLASS_NAMES,'window_seconds':3,'sampling_rate':12,
       'stride_seconds':3,'tail_fraction':.95,'feature_dimension':int(data['features'].shape[1]),
       'registration_is_retrospective':True,'physical_identity_and_chair_verified':False,
       'test_previously_inspected_in_prior_work':True,
       'baseline':{'active_artifact':str(ACTIVE/'posture_classifier_window_3s_rbf_svm.joblib'),
       'sha256':digest(ACTIVE/'posture_classifier_window_3s_rbf_svm.joblib'),
       'method':'StandardScaler + balanced RBF SVC C=1 gamma=.0005 probability=True, predict_proba argmax',
       'comparison':'retrain all arms on identical current data excluding registrations; stored baseline separately'},
       'environment':{'python':sys.version,'platform':platform.platform(),
         'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}}
    manifest['fingerprint']=hashlib.sha256(json.dumps({'audit':audit,'partitions':parts,'registrations':registrations},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    np.savez_compressed(output/'data.npz',**data);write_json(output/'dataset.json',manifest)
    return data,manifest


if __name__=='__main__':
    data,m=prepare();print(json.dumps({'shape':data['seq'].shape,'partitions':m['partitions'],'fingerprint':m['fingerprint']},ensure_ascii=False,indent=2))
