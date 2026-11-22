import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import joblib
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from threadpoolctl import threadpool_limits
from .data import ROOT,OUT,ACTIVE,prepare,take,write_json,digest
from .features import transform
from .catalog import CATALOG,POLICY

def metrics(y,pred,groups=None):
    p,r,f,n=precision_recall_fscore_support(y,pred,labels=np.arange(7),zero_division=0)
    result={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f.mean()),'worst_class_f1':float(f.min()),
      'precision':p,'recall':r,'f1':f,'support':n,'confusion_matrix':confusion_matrix(y,pred,labels=np.arange(7))}
    if groups is not None:
        result['per_subject']={str(g):metrics(y[groups==g],pred[groups==g]) for g in sorted(set(groups))}
    return result

def postprocess(prob,records,starts,method):
    output=np.empty_like(prob);state=None;history=[];previous=None;last=None
    for i,(p,record,t) in enumerate(zip(prob,records,starts)):
        if record!=previous or last is None or t<=last or t-last>3.01:
            state=p.copy();history=[]
        elif method=='ema':state=.6*p+.4*state
        elif method=='confidence':
            if p.max()>=.6:state=p.copy()
        if method=='vote':
            history=(history+[int(p.argmax())])[-3:]
            votes=np.bincount(history,minlength=7).astype(float)
            state=(votes+1e-6*p)/(len(history)+1e-6)
        output[i]=state;previous=record;last=t
    return output

class Arm:
    def __init__(self,spec,seed):self.spec=spec;self.seed=seed
    def fit(self,data):
        spec=self.spec
        if spec['family']=='deep':
            from .deep import build_deep
            self.model=build_deep(spec['kind'],seed=self.seed)
        else:
            from .classical import build_classical
            self.model=build_classical(spec['kind'],seed=self.seed,baseline_params={'C':1.,'gamma':.0005},
                ensemble_members=spec.get('ensemble_members'))
        self.model.fit(transform(data,spec['features']),data['y'],groups=data['groups'])
        return self
    def predict_proba(self,data):return self.model.predict_proba(transform(data,self.spec['features']))

class Fusion:
    def __init__(self,members,weights=None):
        self.members=members;self.weights=weights if weights is not None else np.ones(len(members))/len(members)
    def predict_proba(self,data):return sum(w*m.predict_proba(data) for w,m in zip(self.weights,self.members))

class Temporal:
    def __init__(self,parent,method):self.parent=parent;self.method=method
    def predict_proba(self,data):return postprocess(self.parent.predict_proba(data),data['record_ids'],data['start_s'],self.method)

def complexity(model):
    if isinstance(model,Arm):
        m=model.model
        return {'type':type(m).__name__,'details':getattr(m,'complexity_',{}),'config':getattr(m,'config_',{}),
             'classical_counts':classical_counts(m),
             'architecture':str(getattr(m,'model_',m)), 'training_history':getattr(m,'training_history_',None)}
    if isinstance(model,Fusion):return {'members':[complexity(m) for m in model.members],'weights':list(model.weights)}
    return {'temporal':model.method,'parent':complexity(model.parent)}

def classical_counts(model):
    """Count retained fitted estimator state, avoiding references counted twice."""
    counts={'support_vectors':0,'tree_nodes':0,'trees':0,'coefficient_values':0};seen=set()
    def visit(m):
        if m is None or id(m) in seen:return
        seen.add(id(m))
        if hasattr(m,'support_vectors_'):counts['support_vectors']+=int(len(m.support_vectors_))
        if hasattr(m,'tree_'):counts['trees']+=1;counts['tree_nodes']+=int(m.tree_.node_count)
        try:
            if hasattr(m,'coef_'):counts['coefficient_values']+=int(np.asarray(m.coef_).size)
        except (AttributeError,ValueError):pass
        if hasattr(m,'steps'):
            for _,child in m.steps:visit(child)
        for name in ('estimator_','meta_','gate_','svm_'):visit(getattr(m,name,None))
        for name in ('models_','branch_models_','estimators_'):
            children=getattr(m,name,[])
            for child in np.asarray(children,dtype=object).ravel():visit(child)
    visit(model);return counts

def benchmark(model,data):
    one=take(data,np.array([0]));model.predict_proba(one);ts=[]
    for _ in range(25):
        t=time.perf_counter();model.predict_proba(one);ts.append((time.perf_counter()-t)*1000)
    uncached={k:v for k,v in one.items() if k!='features'};model.predict_proba(uncached);full=[]
    for _ in range(15):
        t=time.perf_counter();model.predict_proba(uncached);full.append((time.perf_counter()-t)*1000)
    t=time.perf_counter();model.predict_proba(data);batch=(time.perf_counter()-t)*1000/len(data['y'])
    return {'cached_single_ms':float(np.median(ts)),'raw_window_ms':float(np.median(full)),
        'raw_window_p95_ms':float(np.percentile(full,95)),'batch_ms_per_window':batch,
        'scope':'36x16 resampled window to probabilities, including feature extraction; excludes CSV/serial acquisition',
        'note':'CPU observational timing may include concurrent host activity; 3s acquisition latency separate'}

def run_one(spec,data,manifest,output=OUT):
    output=Path(output)
    for split in manifest['partitions']:
        folder=output/spec['id']/split['id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'validation.json').exists():
            old=json.loads((folder/'validation.json').read_text())
            assert old['spec']==spec and old['dataset_fingerprint']==manifest['fingerprint']
            assert old['checkpoint_sha256']==digest(folder/'model.joblib')
            continue
        train=take(data,np.isin(data['groups'],split['train']));val=take(data,np.isin(data['groups'],split['validation']))
        start=time.perf_counter()
        if spec['family']=='temporal':model=Temporal(joblib.load(output/spec.get('parent','B00')/split['id']/'model.joblib'),spec['kind'])
        elif spec['family']=='fusion':model=Fusion([joblib.load(output/i/split['id']/'model.joblib') for i in spec['members']],spec.get('weights'))
        else:model=Arm(spec,split['seed']).fit(train)
        seconds=time.perf_counter()-start;prob=model.predict_proba(val)
        assert prob.shape==(len(val['y']),7) and np.isfinite(prob).all() and np.all(prob>=0)
        assert np.allclose(prob.sum(1),1,atol=1e-5)
        score=metrics(val['y'],prob.argmax(1),val['groups'])
        joblib.dump(model,folder/'model.joblib',compress=3)
        np.savez_compressed(folder/'validation_predictions.npz',proba=prob,y=val['y'],groups=val['groups'],record_ids=val['record_ids'],start_s=val['start_s'])
        write_json(folder/'validation.json',{'spec':spec,'split':split,'dataset_fingerprint':manifest['fingerprint'],
          'metrics':score,'train_seconds':seconds,'checkpoint_bytes':(folder/'model.joblib').stat().st_size,
          'checkpoint_sha256':digest(folder/'model.joblib'),'complexity':complexity(model),'inference':benchmark(model,val),
          'created_at':datetime.now(timezone.utc).isoformat()})
        print(f"{spec['id']} {split['id']} acc={score['accuracy']:.4f} f1={score['macro_f1']:.4f} train={seconds:.1f}s",flush=True)

def summarize(spec,output=OUT):
    rows=[json.loads(p.read_text()) for p in sorted((Path(output)/spec['id']).glob('*/validation.json'))]
    if len(rows)!=3:return None
    primary=next(r for r in rows if r['split']['id']=='primary')
    return {'spec':spec,'macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in rows])),
      'accuracy':float(np.mean([r['metrics']['accuracy'] for r in rows])),
      'f1_std':float(np.std([r['metrics']['macro_f1'] for r in rows])),
      'worst_class_f1':float(np.mean([r['metrics']['worst_class_f1'] for r in rows])),
      'class_f1':np.mean([r['metrics']['f1'] for r in rows],axis=0).tolist(),
      'split_f1':{r['split']['id']:r['metrics']['macro_f1'] for r in rows},
      'latency':primary['inference']['raw_window_ms'],'train_seconds':sum(r['train_seconds'] for r in rows),
      'bytes':primary['checkpoint_bytes'],'primary':primary['metrics']}

def improves(a,b,delta=.005):
    return (a['macro_f1']-b['macro_f1']>=delta and
       sum(a['split_f1'][k]>b['split_f1'][k] for k in b['split_f1'])>=2 and
       a['worst_class_f1']-b['worst_class_f1']>=-.02 and
       sum(np.array(a['class_f1'])>=np.array(b['class_f1']))>=4)

def progressive(data,manifest,output=OUT):
    """Start a preregistered complementary pairing once >=10 hypotheses finish."""
    output=Path(output)
    if (output/'pilot_specs.json').exists():return
    rows=[summarize(s,output) for s in CATALOG];done=[s for s in rows if s]
    if len(done)<11 or rows[0] is None:return
    successful=[s for s in done[1:] if improves(s,rows[0])]
    if not {'H08','H11'}<={s['spec']['id'] for s in successful}:return
    spec={'id':'P01','kind':'subject_balanced_svm','features':'denoise','family':'classical',
       'hypothesis':'H08 피험자 균등 가중 + H11 시간 중앙값 필터; 독립 개선 요소의 순차 결합'}
    write_json(output/'interim_analysis.json',{'completed_ids':[s['spec']['id'] for s in done],
       'successful_ids':[s['spec']['id'] for s in successful],'test_opened':False,'next_spec':spec,'policy':POLICY})
    run_one(spec,data,manifest,output);summary=summarize(spec,output)
    parent=next(s for s in done if s['spec']['id']=='H08')
    write_json(output/'pilot_specs.json',{'specs':[spec],'steps':[{'id':'P01','parent':'H08','component':'H11',
       'accepted':improves(summary,parent,.002),'delta':summary['macro_f1']-parent['macro_f1']}]})

def seed_robustness(selected,specs,data,manifest,output=OUT):
    """Separate model randomness from subject split variation; validation only."""
    byid={s['id']:s for s in specs};split=manifest['partitions'][0]
    train=take(data,np.isin(data['groups'],split['train']));val=take(data,np.isin(data['groups'],split['validation']))
    rows=[]
    for seed in (101,202):
        cache={}
        def fit(id):
            if id in cache:return cache[id]
            spec=byid[id]
            if spec['family']=='fusion':model=Fusion([fit(i) for i in spec['members']],spec.get('weights'))
            elif spec['family']=='temporal':model=Temporal(fit(spec.get('parent','B00')),spec['kind'])
            else:model=Arm(spec,seed).fit(train)
            cache[id]=model;return model
        for id in dict.fromkeys(['B00',selected]):
            started=time.perf_counter();model=fit(id);seconds=time.perf_counter()-started;p=model.predict_proba(val)
            folder=Path(output)/'seed_checks'/f'{seed}_{id}';folder.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,folder/'model.joblib',compress=3)
            np.savez_compressed(folder/'predictions.npz',proba=p,y=val['y'],groups=val['groups'])
            rows.append({'id':id,'seed':seed,'split':'primary','train_seconds':seconds,
                'metrics':metrics(val['y'],p.argmax(1),val['groups'])})
    write_json(Path(output)/'seed_robustness.json',{'selection_uses_these_results':False,
        'test_opened':False,'purpose':'fixed primary subject split, two additional seeds; confirmation diagnostic', 'results':rows})

def combine(data,manifest,output=OUT):
    output=Path(output)
    if (output/'selection_lock.json').exists():return
    summaries=[summarize(s,output) for s in CATALOG]
    assert all(s is not None for s in summaries),'Complete all individual arms before combinations'
    base=summaries[0];successful=[s for s in summaries[1:] if improves(s,base)]
    write_json(output/'individual_analysis.json',{'summaries':summaries,'successful_ids':[s['spec']['id'] for s in successful],
        'test_opened':False,'selection_policy':POLICY})
    pilot=json.loads((output/'pilot_specs.json').read_text()) if (output/'pilot_specs.json').exists() else {'specs':[],'steps':[]}
    pool=[base]+successful+[summarize(s,output) for s in pilot['specs']
          if any(step['id']==s['id'] and step['accepted'] for step in pilot['steps'])]
    best=max(pool,key=lambda s:s['macro_f1'])
    best=min([s for s in pool if best['macro_f1']-s['macro_f1']<=.002],key=lambda s:s['latency'])
    specs=list(CATALOG)+pilot['specs'];steps=list(pilot['steps'])+[{'id':best['spec']['id'],
        'step':'best robust candidate including early progressive combination','macro_f1':best['macro_f1']}]
    def attempt(spec,component):
        nonlocal best
        comparable={k:v for k,v in spec.items() if k not in {'id','hypothesis'}}
        if any({k:v for k,v in old.items() if k not in {'id','hypothesis'}}==comparable for old in specs):return
        spec['id']=f'C{sum(s["id"].startswith("C") for s in specs)+1:02d}'
        parent=best['spec']['id'];run_one(spec,data,manifest,output);specs.append(spec)
        summary=summarize(spec,output);accept=improves(summary,best,.002)
        steps.append({'id':spec['id'],'parent':parent,'component':component,'accepted':accept,
            'delta':summary['macro_f1']-best['macro_f1']})
        if accept:best=summary
    # Combine independently validated feature choices with independently validated learners.
    feature_candidates=[s for s in successful if s['spec']['kind']=='baseline' and s['spec']['family']=='classical']
    for candidate in sorted(feature_candidates,key=lambda s:-s['macro_f1']):
        if best['spec']['family']=='classical' and best['spec']['features']=='baseline':
            spec=deepcopy(best['spec']);spec['features']=candidate['spec']['features']
            spec['hypothesis']=best['spec']['id']+' + '+candidate['spec']['id'];attempt(spec,candidate['spec']['id'])
    # Different successful learners are complementary candidates, empirical acceptance is required.
    for candidate in sorted(successful,key=lambda s:-s['macro_f1'])[:5]:
        if candidate['spec']['id']==best['spec']['id'] or candidate['spec']['family']=='temporal':continue
        attempt({'family':'fusion','kind':'soft_vote','features':'mixed','members':[best['spec']['id'],candidate['spec']['id']],
                 'hypothesis':'서로 다른 검증 성공 모델의 동등 가중 확률 결합'},candidate['spec']['id'])
    for candidate in successful:
        if candidate['spec']['family']=='temporal' and candidate['spec']['id']!=best['spec']['id']:
            attempt({'family':'temporal','kind':candidate['spec']['kind'],'features':'inherited','parent':best['spec']['id'],
              'hypothesis':'성공한 인과적 후처리를 선택 모델에 추가'},candidate['spec']['id'])
    # Recursive full-pipeline leave-one-component-out variants, including nested learners.
    byid={s['id']:s for s in specs};ablations=[];variant_cache={}
    def materialize(spec,reason):
        comparable={k:v for k,v in spec.items() if k not in {'id','hypothesis'}}
        for existing in specs:
            if {k:v for k,v in existing.items() if k not in {'id','hypothesis'}}==comparable:return existing['id']
        spec=deepcopy(spec);spec['id']=f'A{sum(s["id"].startswith("A") for s in specs)+1:02d}'
        spec['hypothesis']=reason;run_one(spec,data,manifest,output);specs.append(spec);byid[spec['id']]=spec
        return spec['id']
    def variants(nodeid):
        if nodeid in variant_cache:return variant_cache[nodeid]
        node=byid[nodeid];out=[]
        if node['family']=='temporal':
            parent=node.get('parent','B00');out.append((f'{nodeid} {node["kind"]}',parent))
            for reason,replacement in variants(parent):
                changed=deepcopy(node);changed['parent']=replacement
                out.append((reason,materialize(changed,f'전체 파이프라인에서 {reason} 제거')))
        elif node['family']=='fusion':
            for i,member in enumerate(node['members']):
                others=node['members'][:i]+node['members'][i+1:]
                if len(others)==1:replacement=others[0]
                else:
                    changed=deepcopy(node);changed['members']=others;changed.pop('weights',None)
                    replacement=materialize(changed,f'{nodeid}에서 분류기 {member} 제거')
                out.append((f'branch {member}',replacement))
                for reason,replacement in variants(member):
                    changed=deepcopy(node);changed['members'][i]=replacement
                    out.append((reason,materialize(changed,f'전체 파이프라인에서 {reason} 제거')))
        elif node['family']=='classical':
            if node['features']!='baseline':
                changed=deepcopy(node);changed['features']='baseline'
                out.append((f'{nodeid} features {node["features"]}',materialize(changed,'특징 표현 제거')))
            if node['kind'] in {'stacking','soft_vote','hard_vote','weighted_vote'}:
                members=node.get('ensemble_members') or ['baseline','extra_trees','logistic']
                if len(members)>1:
                    for member in members:
                        changed=deepcopy(node);changed['ensemble_members']=[m for m in members if m!=member]
                        out.append((f'{nodeid} internal {member}',materialize(changed,f'내부 분류기 {member} 제거')))
                if node['kind']=='stacking':
                    changed=deepcopy(node);changed['kind']='soft_vote'
                    out.append((f'{nodeid} learned meta classifier',materialize(changed,'학습 메타 분류기를 단순 평균으로 대체')))
            elif node['kind']!='baseline':
                changed=deepcopy(node);changed['kind']='baseline';changed.pop('ensemble_members',None)
                out.append((f'{nodeid} learner {node["kind"]}',materialize(changed,'학습 구조를 Baseline으로 대체')))
            elif node['features']=='baseline' and nodeid!='B00':out.append(('duplicate baseline','B00'))
        elif node['family']=='deep':
            if node['kind']=='hybrid':
                changed=deepcopy(node);changed['kind']='cnn1d'
                out.append(('SVM embedding head',materialize(changed,'SVM 헤드를 CNN softmax로 대체')))
            out.append((f'{nodeid} neural pipeline','B00'))
        variant_cache[nodeid]=out;return out
    audited=set()
    while best['spec']['id'] not in audited:
        full=best;selected=full['spec']['id'];audited.add(selected)
        candidates=[]
        for reason,replacement in variants(selected):
            small=summarize(byid[replacement],output)
            ablations.append({'id':replacement,'removed':reason,'parent':selected,'f1':small['macro_f1'],
                'contribution':full['macro_f1']-small['macro_f1'],'full_pipeline':True})
            if improves(small,full,.000001):candidates.append(small)
        if not candidates:break
        best=max(candidates,key=lambda s:s['macro_f1'])
        steps.append({'step':'remove harmful component after ablation','id':best['spec']['id'],'parent':selected,
            'accepted':True,'delta':best['macro_f1']-full['macro_f1']})
    seed_robustness(best['spec']['id'],specs,data,manifest,output)
    source={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'experiments/posture_search').glob('*.py'))
            if p.name not in {'report.py','test_protocol.py','test_deep.py','test_classical.py','verify.py'}}
    for relative in ('src/posture_core.py','src/window_pipeline.py','src/app_services.py','validate_personalization.py'):
        source[relative]=digest(ROOT/relative)
    write_json(output/'selection_lock.json',{'locked_at':datetime.now(timezone.utc).isoformat(),
       'selected_id':best['spec']['id'],'selected_spec':best['spec'],'dataset_fingerprint':manifest['fingerprint'],
       'successful_ids':[s['spec']['id'] for s in successful],'test_used_for_selection':False,
       'all_specs':specs,'steps':steps,'ablations':ablations,'source_hashes':source,
       'protocol_sha256':digest(output/'protocol.json'),
       'checkpoints':{str(p.relative_to(output)):digest(p) for p in sorted(output.glob('*/*/model.joblib'))}})

def test(data,manifest,output=OUT):
    output=Path(output);lock=json.loads((output/'selection_lock.json').read_text())
    assert manifest['fingerprint']==lock['dataset_fingerprint']
    assert digest(output/'protocol.json')==lock['protocol_sha256']
    for path,sha in lock['source_hashes'].items():assert digest(ROOT/path)==sha,'Post-lock source changed'
    for path,sha in lock['checkpoints'].items():assert digest(output/path)==sha,'Post-lock model changed'
    x=take(data,np.isin(data['groups'],manifest['partitions'][0]['test']));results=[]
    for spec in lock['all_specs']:
        folder=output/spec['id']/'primary';model=joblib.load(folder/'model.joblib');p=model.predict_proba(x)
        row={'spec':spec,'metrics':metrics(x['y'],p.argmax(1),x['groups']),
          'lock_sha256':digest(output/'selection_lock.json'),'model_sha256':digest(folder/'model.joblib')}
        write_json(folder/'test.json',row);results.append(row)
        np.savez_compressed(folder/'test_predictions.npz',proba=p,y=x['y'],groups=x['groups'],record_ids=x['record_ids'],start_s=x['start_s'])
    # Untouched deployed baseline supplemental reference, same exact query windows.
    bundle=joblib.load(manifest['baseline']['artifact']);model=bundle.get('svm_model',bundle.get('model'))
    p=model.predict_proba(x['features'])
    stored={'metrics':metrics(x['y'],p.argmax(1),x['groups']),'sha256':manifest['baseline']['sha256'],
       'limitation':'Original training cohort differs; matched refit B00 is primary causal comparison'}
    write_json(output/'test_summary.json',{'selected_id':lock['selected_id'],'results':results,'stored_baseline':stored,
       'test_windows':len(x['y']),'evaluated_at':datetime.now(timezone.utc).isoformat()})
    print('TEST COMPLETE; selected',lock['selected_id'],flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','combine','test'])
    parser.add_argument('--ids',nargs='*');parser.add_argument('--output',type=Path,default=OUT);a=parser.parse_args()
    data,manifest=prepare(a.output)
    if not (a.output/'protocol.json').exists():write_json(a.output/'protocol.json',{'hypotheses':CATALOG,'policy':POLICY})
    with threadpool_limits(limits=1):
        if a.stage=='train':
            for spec in CATALOG:
                if not a.ids or spec['id'] in a.ids:
                    run_one(spec,data,manifest,a.output);progressive(data,manifest,a.output)
        elif a.stage=='combine':combine(data,manifest,a.output)
        else:test(data,manifest,a.output)

if __name__=='__main__':
    from experiments.posture_search.run import main as entrypoint
    entrypoint()
