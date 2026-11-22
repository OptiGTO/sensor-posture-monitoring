from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, read_sensor_csv, feature_names
from src.window_pipeline import WindowSpec, segment_sensor_frame, extract_window_features
from validate_personalization import audit_dataset, registration_plan

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/posture_search_20260913'
ACTIVE = ROOT / 'artifacts/models/integrated_20260908_121321_070b045a/artifacts'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def default(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,Path): return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=default,allow_nan=False)+'\n')

def take(data,index):
    return {k:v[index] for k,v in data.items()}

def source_inventory():
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'data').rglob('*'))
            if p.is_file() and (p.suffix=='.csv' or p.name.endswith('.manifest.json'))}

def prepare(output=OUT):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'data.npz').exists():
        manifest=json.loads((output/'dataset.json').read_text())
        assert source_inventory()==manifest['source_inventory'],'Source files/sidecars changed since freeze'
        assert digest(output/'data.npz')==manifest['cache_sha256']
        return dict(np.load(output/'data.npz',allow_pickle=False)),manifest
    before=source_inventory()
    records,audit=audit_dataset(ROOT/'data')
    assert not audit['duplicate_raw_files'] and not audit['duplicate_sensor_sequences']
    # Read membership only, never historical outcome scores.
    old=json.loads((ACTIVE/'window_3s_rbf_svm_evaluation.json').read_text())['partitions']
    subjects=sorted({r.subject for r in records}); test=old['final_evaluation']['subjects']
    val=old['validation']['subjects']; dev=sorted(set(subjects)-set(test))
    parts=[{'id':'primary','seed':20260826,'train':sorted(set(dev)-set(val)), 'validation':val,'test':test}]
    for i,seed in enumerate((20260913,20260914),1):
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=len(val),random_state=seed).split(dev,groups=dev))
        parts.append({'id':f'group_resplit_{i}','seed':seed,'train':np.asarray(dev)[tr].tolist(),
                      'validation':np.asarray(dev)[va].tolist(),'test':test})
    seq=[];personal=[];labels=[];groups=[];ids=[];starts=[];rows=[];registrations=[]
    plans = registration_plan(records)
    if not plans:
        raise ValueError("등록용 자동 배정이 해제되어 개인 기준을 요구하는 과거 실험을 새로 준비할 수 없습니다. 일반 학습을 사용하세요.")
    for registration,evaluation in plans:
        frame=read_sensor_csv(registration.path)
        ref=frame.loc[(frame.elapsed_s-frame.elapsed_s.iloc[0])<=10,SENSOR_COLUMNS].to_numpy(float).mean(0)
        ref_ratio=ref/max(ref.sum(),1e-6)
        registrations.append({**registration.evidence(),'reference_mean':ref,'reference_ratio':ref_ratio,
            'use':'first 10s mean; entire registration file excluded from ALL model fit/evaluation'})
        for record in evaluation:
            w,meta=segment_sensor_frame(read_sensor_csv(record.path),WindowSpec(3))
            means=w.mean(1);ratio=means/np.maximum(means.sum(1,keepdims=True),1e-6)
            p=np.concatenate([np.tile(ref_ratio,(len(w),1)),ratio-ref_ratio,
                np.log((means+1)/(ref+1)),((means.sum(1)+1)/(ref.sum()+1))[:,None]],axis=1)
            seq.extend(w);personal.extend(p);labels.extend([CLASS_NAMES.index(record.label)]*len(w))
            groups.extend([record.subject]*len(w));ids.extend([record.relative_path]*len(w))
            starts.extend(meta.start_s.tolist())
            rows.extend([{**record.evidence(),'start_s':float(t)} for t in meta.start_s])
    data={'seq':np.asarray(seq,dtype=np.float32),'personal':np.asarray(personal,dtype=np.float32),
          'y':np.array(labels),'groups':np.array(groups),'record_ids':np.array(ids),'start_s':np.array(starts)}
    data['features']=extract_window_features(data['seq'])
    for p in parts:
        sets=[set(p[k]) for k in ('train','validation','test')]
        assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        p['counts']={}
        for name in ('train','validation','test'):
            ix=np.isin(data['groups'],p[name]);assert set(data['y'][ix])==set(range(7))
            p['counts'][name]={'windows':int(ix.sum()),'subjects':len(p[name]),
                'records':len(set(data['record_ids'][ix])),'class_counts':np.bincount(data['y'][ix],minlength=7).tolist()}
    assert source_inventory()==before,'Sources changed during preparation'
    manifest={'version':'1.0.0','source_inventory':before,'audit':audit,'registrations':registrations,
       'partitions':parts,'rows':rows,'class_names':CLASS_NAMES,'feature_names':feature_names(),
       'window_seconds':3,'sampling_rate':12,'stride_seconds':3,'min_tail_fraction':.95,
       'personalization':'retrospective separate recording, not prospectively registered; chair/session linkage unknown',
       'session_policy':'all recordings of one NFC-normalized subject stay together; numeric suffix is repetition, not verified session ID',
       'test_previously_inspected':True,'physically_validated':False,
       'baseline':{'artifact':str(ACTIVE/'posture_classifier_window_3s_rbf_svm.joblib'),
        'sha256':digest(ACTIVE/'posture_classifier_window_3s_rbf_svm.joblib'),
        'params':{'C':1.,'gamma':.0005,'class_weight':'balanced','probability':True},
        'decision':'predict_proba argmax as deployed adapter','comparison':'refit identical train data for all arms'},
       'environment':{'python':sys.version,'platform':platform.platform(),
          'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}}
    manifest['fingerprint']=hashlib.sha256(json.dumps({'inventory':before,'partitions':parts},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    np.savez_compressed(output/'data.npz',**data);manifest['cache_sha256']=digest(output/'data.npz')
    write_json(output/'dataset.json',manifest)
    return data,manifest

if __name__=='__main__':
    d,m=prepare();print(d['seq'].shape);print(json.dumps(m['partitions'],ensure_ascii=False,indent=2))
