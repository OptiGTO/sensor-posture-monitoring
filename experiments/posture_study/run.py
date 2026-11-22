"""Reproducible staged study. The test stage requires an immutable selection lock."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .prepare import OUT, ROOT, ACTIVE, prepare, take, write_json, digest
from .catalog import CATALOG, PROTOCOL


def metrics(y,pred,groups=None):
    p,r,f,n=precision_recall_fscore_support(y,pred,labels=np.arange(7),zero_division=0)
    out={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f.mean()),'worst_class_f1':float(f.min()),
         'precision':p.tolist(),'recall':r.tolist(),'f1':f.tolist(),'support':n.tolist(),
         'confusion_matrix':confusion_matrix(y,pred,labels=np.arange(7)).tolist()}
    if groups is not None:
        out['per_subject']={g:metrics(y[groups==g],pred[groups==g]) for g in sorted(set(groups))}
        out['subject_macro_f1_mean']=float(np.mean([v['macro_f1'] for v in out['per_subject'].values()]))
    return out


def postprocess(prob,records,starts,method):
    """Causal streaming state, reset only on observable recording boundaries/gaps."""
    out=np.empty_like(prob); prev_record=None; state=None; pending=None; count=0; last_t=None
    for i,(p,rec,t) in enumerate(zip(prob,records,starts)):
        if rec!=prev_record or last_t is None or t<=last_t or t-last_t>3.01:
            state=p.copy();pending=None;count=0
        elif method=='ema': state=.6*p+.4*state
        else:
            current=int(np.argmax(state)); candidate=int(np.argmax(p))
            if candidate==current:pending=None;count=0;state=p.copy()
            else:
                count=count+1 if pending==candidate else 1; pending=candidate
                if count>=2:state=p.copy();pending=None;count=0
        out[i]=state;prev_record=rec;last_t=t
    return out


class TemporalModel:
    def __init__(self,base,method):self.base=base;self.method=method
    def predict_proba(self,x):return postprocess(self.base.predict_proba(x),x['record_ids'],x['start_s'],self.method)
    def complexity(self):return {'type':'causal_'+self.method,'base':complexity(self.base),'state_floats':7,'maximum_confirmation_delay_s':3 if self.method=='hysteresis' else None}


class HybridModel:
    def __init__(self,seed):self.seed=seed
    def fit(self,x,y,validation=None):
        from .neural import NeuralEstimator
        self.encoder=NeuralEstimator('1dcnn',seed=self.seed,epochs=60,patience=10,batch_size=64)
        self.encoder.fit(x,y,validation=validation)
        self.head=make_pipeline(StandardScaler(),SVC(C=1,gamma='scale',class_weight='balanced',probability=True,random_state=self.seed))
        self.head.fit(self.encoder.embedding(x),y);return self
    def predict_proba(self,x):return self.head.predict_proba(self.encoder.embedding(x))
    def complexity(self):return {'encoder':complexity(self.encoder),'support_vectors':int(self.head[-1].n_support_.sum())}


class FusionModel:
    def __init__(self,models):self.models=models
    def predict_proba(self,x):return np.mean([m.predict_proba(x) for m in self.models],axis=0)
    def complexity(self):return {'type':'equal_soft_vote','members':[complexity(m) for m in self.models]}


def complexity(model):
    if hasattr(model,'complexity'):
        obj=model.complexity
        return obj() if callable(obj) else obj
    return getattr(model,'complexity_',{})


def build(spec,seed):
    if spec['family']=='classical':
        from .classical import ClassicalEstimator
        return ClassicalEstimator(spec['method'],seed=seed,components=spec.get('components'),
                                  ensemble_members=spec.get('ensemble_members'))
    if spec['family']=='neural':
        from .neural import NeuralEstimator
        return NeuralEstimator('1dcnn' if spec['method']=='aug_cnn' else spec['method'],seed=seed,
             epochs=60,patience=10,batch_size=64,augmentation=spec['method']=='aug_cnn')
    if spec['family']=='hybrid':return HybridModel(seed)
    raise ValueError(spec)


def benchmark(model,x,repetitions=35):
    one=take(x,np.array([0]));model.predict_proba(one)
    ts=[]
    for _ in range(repetitions):
        t=time.perf_counter();model.predict_proba(one);ts.append((time.perf_counter()-t)*1000)
    t=time.perf_counter();model.predict_proba(x);batch=(time.perf_counter()-t)*1000/len(x['y'])
    return {'single_window_ms_median':float(np.median(ts)),'single_window_ms_p95':float(np.percentile(ts,95)),
            'batch_ms_per_window':float(batch),'repetitions':repetitions,
            'scope':'classifier and model-specific transforms; cached legacy feature extraction excluded'}


def run_one(spec,data,manifest,output=OUT):
    output=Path(output);results=[]
    for part in manifest['partitions']:
        folder=output/spec['id']/part['id'];folder.mkdir(parents=True,exist_ok=True)
        result_path=folder/'validation.json'
        if result_path.exists():
            old=json.loads(result_path.read_text())
            assert old['spec']==spec and old['dataset_fingerprint']==manifest['fingerprint']
            assert digest(folder/'checkpoint.joblib')==old['checkpoint_sha256']
            results.append(old);continue
        x=take(data,np.isin(data['groups'],part['train']));val=take(data,np.isin(data['groups'],part['validation']))
        t=time.perf_counter()
        if spec['family']=='temporal':
            parent=spec.get('parent','B00');base=joblib.load(output/parent/part['id']/'checkpoint.joblib')
            model=TemporalModel(base,spec['method'])
        elif spec['family']=='fusion':
            model=FusionModel([joblib.load(output/id/part['id']/'checkpoint.joblib') for id in spec['members']])
        else:
            model=build(spec,part['seed']);model.fit(x,x['y'],validation=(val,val['y']))
        train_s=time.perf_counter()-t
        probs=model.predict_proba(val)
        assert probs.shape==(len(val['y']),7) and np.isfinite(probs).all()
        assert np.allclose(probs.sum(1),1,atol=1e-4)
        scores=metrics(val['y'],probs.argmax(1),val['groups'])
        joblib.dump(model,folder/'checkpoint.joblib',compress=3)
        np.savez_compressed(folder/'validation_predictions.npz',proba=probs,y=val['y'],groups=val['groups'],record_ids=val['record_ids'],start_s=val['start_s'])
        config=getattr(model,'get_config',lambda:None)()
        result={'spec':spec,'split':part,'dataset_fingerprint':manifest['fingerprint'],
             'checkpoint_sha256':digest(folder/'checkpoint.joblib'),'train_seconds':train_s,
             'checkpoint_bytes':(folder/'checkpoint.joblib').stat().st_size,'metrics':scores,
             'inference':benchmark(model,val),'complexity':complexity(model),'configuration':config,
             'architecture':getattr(model,'architecture_',repr(model)),
             'training_history':getattr(model,'training_history',None),'created_at':datetime.now(timezone.utc).isoformat()}
        write_json(result_path,result);results.append(result)
        print(f"{spec['id']} {spec['method']} {part['id']} acc={scores['accuracy']:.4f} f1={scores['macro_f1']:.4f} train={train_s:.1f}s",flush=True)
    return results


def summarize(spec,output=OUT):
    rs=[json.loads(p.read_text()) for p in sorted((Path(output)/spec['id']).glob('*/validation.json'))]
    if len(rs)!=3:return None
    primary=next(r for r in rs if r['split']['id']=='primary')
    return {'spec':spec,'accuracy':float(np.mean([r['metrics']['accuracy'] for r in rs])),
            'macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in rs])),
            'std_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in rs])),
            'worst_class_f1':float(np.mean([r['metrics']['worst_class_f1'] for r in rs])),
            'class_f1':np.mean([r['metrics']['f1'] for r in rs],axis=0).tolist(),
            'split_f1':{r['split']['id']:r['metrics']['macro_f1'] for r in rs},
            'latency_ms':primary['inference']['single_window_ms_median'],
            'size_bytes':primary['checkpoint_bytes'],'train_seconds':sum(r['train_seconds'] for r in rs),
            'primary':primary['metrics']}


def is_improvement(a,b,threshold=.005):
    return (a['macro_f1']-b['macro_f1']>=threshold and
        sum(a['split_f1'][k]>b['split_f1'][k] for k in a['split_f1'])>=2 and
        a['worst_class_f1']-b['worst_class_f1']>=-.02 and
        sum(np.asarray(a['class_f1'])>=np.asarray(b['class_f1']))>=4)


def combine(data,manifest,output=OUT):
    output=Path(output)
    if (output/'selection_lock.json').exists():return json.loads((output/'selection_lock.json').read_text())
    sums=[summarize(s,output) for s in CATALOG]
    if any(s is None for s in sums):raise ValueError('Complete every independent experiment before combination')
    base=sums[0];success=[s for s in sums[1:] if is_improvement(s,base)]
    write_json(output/'independent_analysis.json',{'summaries':sums,'successful_ids':[s['spec']['id'] for s in success],
         'rule':PROTOCOL['successful_rule'],'test_opened':False,
         'factor_groups':{'representation':['relative','spatial','personal','multiscale','select','pca'],
                          'training':['subject_weight','gain_aug'],'preprocessing':['robust'],
                          'architecture':['hierarchy','soft_hierarchy','extra_trees','1dcnn','attention'],
                          'decision':['majority','weighted','stacking','ema','hysteresis']}})
    pool=[base]+success;best=max(pool,key=lambda s:s['macro_f1'])
    near=[s for s in pool if best['macro_f1']-s['macro_f1']<=.002]
    best=min(near,key=lambda s:s['latency_ms'])
    combo_specs=[];steps=[{'step':'start','id':best['spec']['id'],'mean_validation_f1':best['macro_f1']}]
    # Representation/training components can be overlaid only on classical estimators.
    allowed={'relative','spatial','personal','multiscale','subject_weight','robust','select','pca','gain_aug'}
    for candidate in sorted(success,key=lambda s:-s['macro_f1']):
        method=candidate['spec']['method']
        if best['spec']['family']!='classical' or method not in allowed:continue
        current_components=best['spec'].get('components',[])
        if method==best['spec']['method'] or method in current_components:continue
        existing=set(current_components+[best['spec']['method']])
        if any(pair<=existing|{method} for pair in ({'relative','spatial'},{'select','pca'},{'gain_aug','personal'})):
            steps.append({'step':'incompatible_skip','parent':best['spec']['id'],'component':candidate['spec']['id'],
                          'reason':'alternative representation/projection or augmentation reference mismatch'})
            continue
        spec=deepcopy(best['spec']);spec['id']=f'C{len(combo_specs)+1:02d}'
        spec['components']=current_components+[method];spec['hypothesis']=f"{best['spec']['id']} + {candidate['spec']['id']} 검증된 구성요소"
        run_one(spec,data,manifest,output);combo_specs.append(spec);s=summarize(spec,output)
        accepted=is_improvement(s,best,.002)
        steps.append({'id':spec['id'],'parent':best['spec']['id'],'component':candidate['spec']['id'],'accepted':accepted,'delta':s['macro_f1']-best['macro_f1']})
        if accepted:best=s
    # Greedy equal-weight fusion of validated but different learners; no test-derived weights.
    for candidate in sorted(success,key=lambda s:-s['macro_f1'])[:4]:
        if candidate['spec']['id']==best['spec']['id'] or candidate['spec']['family']=='temporal':continue
        spec={'id':f'C{len(combo_specs)+1:02d}','method':'equal_soft_vote','family':'fusion',
              'members':[best['spec']['id'],candidate['spec']['id']],
              'hypothesis':f"{best['spec']['id']} + {candidate['spec']['id']}의 상보적 오류 검증"}
        run_one(spec,data,manifest,output);combo_specs.append(spec);s=summarize(spec,output);accepted=is_improvement(s,best,.002)
        steps.append({'id':spec['id'],'parent':best['spec']['id'],'component':candidate['spec']['id'],'accepted':accepted,'delta':s['macro_f1']-best['macro_f1']})
        if accepted:best=s
    for candidate in success:
        if candidate['spec']['family']!='temporal' or candidate['spec']['id']==best['spec']['id']:continue
        spec={'id':f'C{len(combo_specs)+1:02d}','family':'temporal','method':candidate['spec']['method'],
              'parent':best['spec']['id'],'hypothesis':f"{best['spec']['id']} + {candidate['spec']['method']}"}
        run_one(spec,data,manifest,output);combo_specs.append(spec);s=summarize(spec,output);accepted=is_improvement(s,best,.002)
        steps.append({'id':spec['id'],'parent':best['spec']['id'],'component':candidate['spec']['id'],'accepted':accepted,'delta':s['macro_f1']-best['macro_f1']})
        if accepted:best=s
    # Remove harmful additions before freezing; peel composite nodes and audit each retained branch.
    ablations=[];ablation_specs=[]
    spec_by_id={s['id']:s for s in CATALOG+combo_specs}
    def inspect_node(node_id,seen):
        if node_id in seen:return
        seen.add(node_id);node=spec_by_id[node_id];full=summarize(node,output)
        if node['family']=='fusion':
            for member in node['members']:
                smaller=summarize(spec_by_id[member],output)
                ablations.append({'spec':spec_by_id[member],'removed':'other fusion branch','parent':node_id,'summary':smaller,'existing':True})
                inspect_node(member,seen)
        elif node['family']=='temporal':
            parent=node.get('parent','B00');smaller=summarize(spec_by_id[parent],output)
            ablations.append({'spec':spec_by_id[parent],'removed':node['method'],'parent':node_id,'summary':smaller,'existing':True})
            inspect_node(parent,seen)
        elif node['family']=='neural' and node['method']=='aug_cnn':
            ablations.append({'spec':spec_by_id['H20'],'removed':'training gain/noise augmentation',
                              'parent':node_id,'summary':summarize(spec_by_id['H20'],output),'existing':True})
        elif node['family']=='classical':
            components=list(dict.fromkeys(node.get('components',[])+([node['method']] if node['method'] in allowed else [])))
            for component in components:
                spec=deepcopy(node);spec['id']=f'A{len(ablation_specs)+1:02d}'
                spec['components']=[c for c in components if c!=component]
                if node['method'] in allowed:spec['method']='baseline'
                spec['hypothesis']=f"{node_id}에서 {component} 제거"
                run_one(spec,data,manifest,output);ablation_specs.append(spec);spec_by_id[spec['id']]=spec
                ablations.append({'spec':spec,'removed':component,'parent':node_id,'summary':summarize(spec,output)})
            if node['method']=='stacking':
                members=node.get('ensemble_members') or ['baseline','logreg','extra_trees']
                changes=[('stacking meta learner',{'method':'weighted'})]
                if len(members)>2:
                    changes += [(f'ensemble member {member}',{'ensemble_members':[m for m in members if m!=member]}) for member in members]
                for removed,change in changes:
                    spec=deepcopy(node);spec.update(change);spec['id']=f'A{len(ablation_specs)+1:02d}'
                    spec['hypothesis']=f'{node_id}에서 {removed} 제거'
                    run_one(spec,data,manifest,output);ablation_specs.append(spec);spec_by_id[spec['id']]=spec
                    ablations.append({'spec':spec,'removed':removed,'parent':node_id,'summary':summarize(spec,output)})
    def substitute(node_id,target,replacement):
        if node_id==target:return replacement
        node=spec_by_id[node_id];spec=deepcopy(node);changed=False
        if node['family']=='fusion':
            members=[substitute(m,target,replacement) for m in node['members']]
            changed=members!=node['members'];spec['members']=members
        elif node['family']=='temporal':
            parent=substitute(node.get('parent','B00'),target,replacement)
            changed=parent!=node.get('parent','B00');spec['parent']=parent
        if not changed:return node_id
        spec['id']=f'A{len(ablation_specs)+1:02d}';spec['hypothesis']=f'전체 파이프라인 {node_id}: {target}를 {replacement}로 대체'
        run_one(spec,data,manifest,output);ablation_specs.append(spec);spec_by_id[spec['id']]=spec
        return spec['id']
    audited=set()
    while best['spec']['id'] not in audited:
        root_id=best['spec']['id'];audited.add(root_id);start=len(ablations)
        inspect_node(root_id,set())
        # Branch-level replacements are propagated through the whole final graph.
        for a in list(ablations[start:]):
            if a['parent']==root_id:continue
            replacement=substitute(root_id,a['parent'],a['spec']['id'])
            if replacement!=root_id:
                spec=spec_by_id[replacement]
                ablations.append({'spec':spec,'removed':a['removed'],'parent':root_id,
                                  'summary':summarize(spec,output),'full_pipeline':True})
        improved=[a for a in ablations[start:] if a['parent']==root_id and is_improvement(a['summary'],best,.000001)]
        if not improved:break
        a=max(improved,key=lambda a:a['summary']['macro_f1'])
        steps.append({'step':'ablation_remove','parent':root_id,'id':a['spec']['id'],
                      'component':a['removed'],'accepted':True,'delta':a['summary']['macro_f1']-best['macro_f1']})
        best=a['summary']
    all_specs=list({s['id']:s for s in CATALOG+combo_specs+ablation_specs}.values())
    checkpoints={str(p.relative_to(output)):digest(p) for p in sorted(output.glob('*/*/checkpoint.joblib'))}
    lock={'locked_at':datetime.now(timezone.utc).isoformat(),'selected_id':best['spec']['id'],
          'selected_spec':best['spec'],'successful_ids':[s['spec']['id'] for s in success],
          'selection_basis':'validation only; fixed rules in protocol.json','test_used_for_selection':False,
          'dataset_fingerprint':manifest['fingerprint'],'all_specs':all_specs,'steps':steps,'ablations':ablations,
          'checkpoints':checkpoints,'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in
              [p for p in sorted((ROOT/'experiments/posture_study').glob('*.py')) if not p.name.startswith('test_') and p.name!='report.py']+
              [ROOT/'src/posture_core.py',ROOT/'src/window_pipeline.py',ROOT/'src/app_services.py',ROOT/'validate_personalization.py']},
          'data_cache_sha256':digest(output/'data.npz'),
          'protocol_sha256':digest(output/'protocol.json')}
    write_json(output/'selection_lock.json',lock)
    return lock


def evaluate_test(data,manifest,output=OUT):
    output=Path(output);lock=json.loads((output/'selection_lock.json').read_text())
    assert lock['dataset_fingerprint']==manifest['fingerprint'],'Locked dataset differs'
    if 'data_cache_sha256' in lock:assert digest(output/'data.npz')==lock['data_cache_sha256'],'Locked window cache changed'
    assert digest(output/'protocol.json')==lock['protocol_sha256'],'Locked protocol differs'
    for p,h in lock['source_hashes'].items():assert digest(ROOT/p)==h,'Post-lock implementation changed'
    for p,h in lock['checkpoints'].items():assert digest(output/p)==h,'Post-lock model changed'
    x=take(data,np.isin(data['groups'],manifest['partitions'][0]['test']))
    summaries=[]
    for spec in lock['all_specs']:
        folder=output/spec['id']/'primary';model=joblib.load(folder/'checkpoint.joblib');probs=model.predict_proba(x)
        result={'spec':spec,'metrics':metrics(x['y'],probs.argmax(1),x['groups']),
             'selection_lock_sha256':digest(output/'selection_lock.json'),'checkpoint_sha256':digest(folder/'checkpoint.joblib')}
        write_json(folder/'test.json',result)
        np.savez_compressed(folder/'test_predictions.npz',proba=probs,y=x['y'],groups=x['groups'],record_ids=x['record_ids'],start_s=x['start_s'])
        summaries.append(result)
    bundle=joblib.load(ACTIVE/'posture_classifier_window_3s_rbf_svm.joblib')
    legacy=bundle.get('model',bundle.get('svm_model'));probs=legacy.predict_proba(x['features'])
    reference={'name':'untouched active checkpoint; original training data includes its training registrations',
      'metrics':metrics(x['y'],probs.argmax(1),x['groups']),'artifact_sha256':manifest['baseline']['sha256']}
    write_json(output/'stored_baseline_test.json',reference)
    np.savez_compressed(output/'stored_baseline_test_predictions.npz',proba=probs,y=x['y'],groups=x['groups'])
    write_json(output/'test_summary.json',{'locked_selected_id':lock['selected_id'],'evaluated_at':datetime.now(timezone.utc).isoformat(),
             'test_windows':len(x['y']),'results':summaries,'stored_baseline':reference})
    print('Locked final',lock['selected_id'],'test',next(s['metrics']['macro_f1'] for s in summaries if s['spec']['id']==lock['selected_id']),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','combine','test','all'])
    parser.add_argument('--ids',nargs='*');parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    data,manifest=prepare(args.output)
    if not (args.output/'protocol.json').exists():write_json(args.output/'protocol.json',{'protocol':PROTOCOL,'hypotheses':CATALOG})
    with threadpool_limits(limits=2):
        if args.stage in ('train','all'):
            for spec in CATALOG:
                if args.ids and spec['id'] not in args.ids:continue
                run_one(spec,data,manifest,args.output)
        if args.stage in ('combine','all'):combine(data,manifest,args.output)
        if args.stage in ('test','all'):evaluate_test(data,manifest,args.output)


if __name__=='__main__':
    # Pickled estimators must use a stable import path, never __main__.
    from experiments.posture_study.run import main as entrypoint
    entrypoint()
