"""Retrospective equal soft voting of explicitly ranked frozen candidates."""
from __future__ import annotations
import argparse
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import joblib
from experiments.sensor_gain.run import ROOT, now, read, write_json, digest, verify, protected_snapshot, calibrate
from experiments.sensor_gain.summarize import matrix, metrics, record_predictions
from src.model_comparison import scores
from src.posture_core import CLASS_NAMES
from src.window_pipeline import extract_window_features, aligned_predict_proba

RUNS = [ROOT / 'research/results/2026-09-23_150000_sensor_gain_2s3s',
        ROOT / 'research/results/2026-09-23_181749_sensor_gain_regularized_et_2s3s']


def align(predictions, datasets, durations):
    maps = []
    for pred, data, seconds in zip(predictions, datasets, durations):
        mapping = {}
        for j, i in enumerate(pred['indices']):
            if pred['truth'][j] != data['y'][i] or pred['groups'][j] != data['groups'][i] or pred['record_ids'][j] != data['record_ids'][i]:
                raise ValueError('Prediction identity differs from cached input')
            key = (str(pred['record_ids'][j]), round(float(data['start_s'][i] + seconds), 6))
            if key in mapping: raise ValueError('Duplicate record/end identity')
            mapping[key] = j
        maps.append(mapping)
    keys = sorted(set.intersection(*(set(m) for m in maps)))
    if not keys: raise ValueError('No common endpoints')
    indices = [np.array([m[k] for k in keys]) for m in maps]
    for key in ('truth', 'groups', 'record_ids'):
        np.testing.assert_array_equal(predictions[0][key][indices[0]], predictions[1][key][indices[1]])
    return indices, keys


def equal_mean(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 7:
        raise ValueError('Class-aligned 7-column probabilities required')
    for p in (a, b):
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
            raise ValueError('Invalid probability')
    return (a + b) / 2


def intervals(y, before, after, groups):
    people = np.unique(groups)
    a = np.stack([matrix(y[groups == s], before[groups == s]) for s in people])
    b = np.stack([matrix(y[groups == s], after[groups == s]) for s in people])
    rng = np.random.default_rng(20260921); delta = []
    for _ in range(2000):
        draw = rng.integers(0, len(people), len(people))
        first, second = metrics(a[draw].sum(0)), metrics(b[draw].sum(0))
        delta.append([second[k] - first[k] for k in ('accuracy', 'macro_f1')])
    bounds = np.percentile(delta, [2.5, 97.5], axis=0)
    return {k: bounds[:, i].tolist() for i, k in enumerate(('accuracy', 'macro_f1'))}


def load_fit(run, c, stage, part):
    dest = run / stage / c['id'] / ('holdout' if stage == 'final' else part)
    m = read(dest / 'metrics.json')
    for name, sha in m['artifacts'].items():
        if digest(dest / name) != sha: raise ValueError('Saved member changed')
    pred = dict(np.load(dest / 'predictions.npz', allow_pickle=False))
    for metric in ('accuracy', 'macro_f1'):
        np.testing.assert_allclose(metrics(matrix(pred['truth'], pred['prediction']))[metric], m['evaluation'][metric], atol=1e-12)
    return dest, pred


def run(out, criterion):
    started = now(); protocols = [verify(p) for p in RUNS]
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'protected_before.json', protected_snapshot())
    rank = []
    for source, p in zip(RUNS, protocols):
        for c in p['conditions']:
            _, pred = load_fit(source, c, 'final', '')
            measured = metrics(matrix(pred['truth'], pred['prediction']))
            rank.append({**c, 'run': str(source), 'accuracy': measured['accuracy'], 'macro_f1': measured['macro_f1']})
    metric = 'accuracy' if criterion == 'accuracy' else 'macro_f1'
    rank.sort(key=lambda x: (-x[metric], x['id']))
    members = rank[:2] if criterion != 'diverse_macro_f1' else [rank[0], next(r for r in rank if r['family'] != rank[0]['family'])]
    # Historical selection and all member artifacts are locked before combining probabilities.
    evidence = {}
    for member in members:
        source = Path(member['run']); p = verify(source)
        if p['origin_dataset_sha256'] != protocols[0]['origin_dataset_sha256']:
            raise ValueError('Different underlying records')
        for actual, expected in zip(p['partitions'], protocols[0]['partitions']):
            for key in ('train', 'validation', 'test'):
                if actual[key] != expected[key]: raise ValueError('Different subject memberships')
        for name in ('protocol.json', 'protocol_lock.json', f"data_{member['seconds']}s.npz"):
            evidence[str(source / name)] = digest(source / name)
        for stage, parts in (('development', p['partitions']), ('final', p['partitions'][:1])):
            for part in parts:
                dest, _ = load_fit(source, member, stage, part['id'])
                for name in ('model.joblib', 'metrics.json', 'predictions.npz'):
                    evidence[str(dest / name)] = digest(dest / name)
    origin = read(RUNS[0] / 'origin_dataset.json')
    for row in origin['records']:
        if digest(ROOT / 'data' / row['source_relative_path']) != row['sha256']:
            raise ValueError('Historical original changed')
    files = [Path(__file__), Path(__file__).with_name('test_top_two_ensemble.py'), Path(__file__).with_name('run.py'),
             Path(__file__).with_name('summarize.py'), ROOT / 'experiments/improvement_retest/prepare.py', *(ROOT / 'src').rglob('*.py')]
    source_hashes = {}
    for f in files:
        rel = f.relative_to(ROOT); dest = out / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dest); source_hashes[str(rel)] = digest(dest)
    protocol = {'started_at': started, 'locked_at': now(), 'criterion': criterion, 'ranking': rank, 'members': members,
        'weights': [.5, .5], 'source_hashes': source_hashes, 'input_hashes': evidence,
        'partitions': protocols[0]['partitions'], 'class_names': CLASS_NAMES,
        'alignment': 'Intersect (record_id, nominal_end_s); 2s and 3s meet every 6s. Compare members and ensemble on identical endpoints.',
        'scope': 'Sixteen conditions from the two sensor-gain studies in this conversation; training and other-cohort scores are excluded.',
        'selection_bias': 'Members selected using previously observed historical holdout scores. All subsequent results are exploratory reuse, not an untouched test.',
        'success': 'Descriptive only: report Accuracy and Macro-F1 changes versus each member; no automatic adoption or weight optimization',
        'training': 'Reuse exact fitted development and 26-person final members; zero new model fits',
        'bootstrap': '2000 paired subject-cluster draws conditional on fitted and selected models; selection bias and multiplicity not corrected',
        'approved': False, 'fresh_external_test': False,
        'command': [sys.executable, *sys.argv], 'source_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
        'dirty_state': subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'environment': {'python':sys.version,'packages':{k:version(k) for k in ('numpy','scikit-learn','xgboost','joblib')}}}
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'protocol_lock.json', {'sha256':digest(out/'protocol.json'),'locked_at':now()})
    print('LOCKED', [(m['id'], m[metric]) for m in members], flush=True)
    results = []
    for stage, parts in (('development', protocol['partitions']), ('final', protocol['partitions'][:1])):
        for part in parts:
            dest = out / stage / ('holdout' if stage == 'final' else part['id']); dest.mkdir(parents=True)
            fits = [load_fit(Path(m['run']), m, stage, part['id']) for m in members]
            preds = [fit[1] for fit in fits]
            data = [np.load(Path(m['run']) / f"data_{m['seconds']}s.npz", allow_pickle=False) for m in members]
            idx, keys = align(preds, data, [m['seconds'] for m in members])
            probabilities = [p['probability'][i] for p,i in zip(preds,idx)]
            replay = []; model_refs = []
            for member, (fit, pred), cache, positions, expected in zip(members, fits, data, idx, probabilities):
                bundle = joblib.load(fit/'model.joblib')
                if bundle['label_names'] != CLASS_NAMES: raise ValueError('Member class order mismatch')
                if set(bundle['train_subjects']) & set(pred['groups'][positions]): raise ValueError('Training/evaluation subject overlap')
                seq = cache['seq'][pred['indices'][positions]]
                features = extract_window_features(calibrate(seq, bundle['gains']))
                actual = aligned_predict_proba(bundle['model'], features, 7)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                replay.append(actual)
                model_refs.append({'path':str(fit/'model.joblib'),'sha256':digest(fit/'model.joblib'),'weight':.5,'window_seconds':member['seconds']})
            ensemble = equal_mean(*probabilities)
            np.testing.assert_allclose(equal_mean(*replay), ensemble, rtol=0, atol=1e-12)
            y, groups, ids = [preds[0][k][idx[0]] for k in ('truth','groups','record_ids')]
            all_prob = [*probabilities, ensemble]; names = [m['id'] for m in members] + ['equal_ensemble']
            measured = {}; record_scores = {}
            for name, prob in zip(names, all_prob):
                pred = prob.argmax(1); measured[name] = metrics(matrix(y, pred))
                np.testing.assert_allclose(measured[name]['macro_f1'], scores(y,pred)['macro_f1'], atol=1e-12)
                rec = record_predictions({'truth':y,'groups':groups,'record_ids':ids,'probability':prob})
                record_scores[name] = metrics(matrix(rec['truth'],rec['prediction']))
            row = {'stage':stage,'partition':part['id'],'windows':len(y),'records':len(set(ids)), 'subjects':len(set(groups)),
                'metrics':measured,'record_metrics':record_scores,'raw_sequence_replay_parity':True,
                'per_subject':{str(s):{name:metrics(matrix(y[groups==s],prob[groups==s].argmax(1))) for name,prob in zip(names,all_prob)} for s in np.unique(groups)},
                'error_changes':{name:{'corrected':int(np.sum((prob.argmax(1)!=y)&(ensemble.argmax(1)==y))),
                    'introduced':int(np.sum((prob.argmax(1)==y)&(ensemble.argmax(1)!=y)))} for name,prob in zip(names[:2],probabilities)}}
            if stage == 'final':
                row['delta_ci95'] = {name:intervals(y,prob.argmax(1),ensemble.argmax(1),groups) for name,prob in zip(names[:2],probabilities)}
            np.savez_compressed(dest/'predictions.npz',truth=y,groups=groups,record_ids=ids,end_s=np.array([k[1] for k in keys]),
                member_0_probability=probabilities[0],member_1_probability=probabilities[1],probability=ensemble,prediction=ensemble.argmax(1))
            write_json(dest/'ensemble.json',{'members':model_refs,'class_names':CLASS_NAMES,'alignment':protocol['alignment'],
                'combination':'probability arithmetic mean, argmax','approved':False,'protocol_sha256':digest(out/'protocol.json')})
            row['predictions_sha256']=digest(dest/'predictions.npz');write_json(dest/'metrics.json',row);results.append(row)
            print('EVALUATED',stage,part['id'],len(y),measured['equal_ensemble']['accuracy'],measured['equal_ensemble']['macro_f1'],flush=True)
    for path,sha in evidence.items():
        if digest(path)!=sha:raise ValueError('Input evidence drifted')
    if protected_snapshot()!=read(out/'protected_before.json'):raise ValueError('Protected originals changed')
    summary={'finished_at':now(),'members':members,'criterion':criterion,'results':results,'raw_and_active_models_unchanged':True,'approved':False,'new_model_fits':0}
    write_json(out/'summary.json',summary)
    report(out,protocol,summary)


def report(out,p,s):
    names=[m['id'] for m in p['members']]+['equal_ensemble'];final=s['results'][-1];dev=s['results'][:-1]
    rel='../../research/results/'+out.name
    lines=['# 상위 두 모델의 동일 가중치 앙상블 실험','',
        f"실행 시작: {p['started_at']}; 종료: {s['finished_at']}. 기존 센서 보정 실험 16개 조건에서 `{p['criterion']}` 기준 상위 두 모델을 선택했다.",'',
        '모델마다 클래스 확률을 정렬한 뒤 `P = 0.5 × P1 + 0.5 × P2`로 결합했다. 가중치 탐색이나 재학습은 하지 않았다. 보정 사용 여부와 계수는 각 모델의 기존 설정을 유지했다.','',
        '| 선택 모델 | 원래 윈도우 | 보정 | 선택 당시 Accuracy | 선택 당시 Macro-F1 |','| --- | ---: | --- | ---: | ---: |']
    for m in p['members']:lines.append(f"| {m['family']} | {m['seconds']}초 | {m['arm']} | {100*m['accuracy']:.2f}% | {100*m['macro_f1']:.2f}% |")
    lines+=['','## 시간 정렬과 평가 범위','',
        '2초와 3초의 모든 창은 개수와 끝나는 시점이 다르다. 동일 기록·동일 종료 시점이 겹치는 6초 간격만 비교했다. 각 시점에서 2초 모델은 직전 2초, 3초 모델은 직전 3초를 본다. 이는 현재 비중첩 창 방식의 6초 간격 앙상블 평가이며 2초/3초 간격 실시간 출력 검증은 아니다.','',
        f"최종 평가는 26명으로 학습한 기존 모델을 사용해 역사적 평가자 {final['subjects']}명·{final['records']}개 기록·{final['windows']}개 공통 시점을 평가했다. 단일 모델도 정확히 같은 공통 시점에서 다시 계산했으므로 선택 당시 전체 창 점수와 다를 수 있다.",'',
        '**두 모델을 이미 확인한 최종 점수로 선택했으므로 이번 결과는 탐색적 재사용 평가다. 독립적인 새 테스트나 신규 사용자에 대한 일반화 성능을 의미하지 않는다.** 개발 분할 결과도 함께 보고하며 연구 모델 또는 운영 모델을 자동 채택하지 않는다.','',
        '## 동일 시점의 단일 모델과 앙상블','',
        '| 모델 | 개발 평균 Accuracy | 개발 평균 Macro-F1 | 최종 Accuracy | 최종 Macro-F1 | 기록 단위 Macro-F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in names:
        vals=[np.mean([r['metrics'][n][k] for r in dev]) for k in ('accuracy','macro_f1')]
        vals += [final['metrics'][n][k] for k in ('accuracy','macro_f1')]+[final['record_metrics'][n]['macro_f1']]
        lines.append('| '+n+' | '+' | '.join(f'{100*x:.2f}%' for x in vals)+' |')
    lines+=['','기록 단위 점수는 공통 시점 확률을 기록별 평균한 보조 지표다. 전체 기록을 관측한 결과이며 실시간 창 지표와 구분한다.','',
        '## 단일 모델 대비 변화','',
        '| 비교 기준 | Accuracy 변화 | 95% 구간 | Macro-F1 변화 | 95% 구간 | 고친 오류 / 새 오류 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in names[:2]:
        ds=[final['metrics']['equal_ensemble'][k]-final['metrics'][n][k] for k in ('accuracy','macro_f1')]
        ci=final['delta_ci95'][n];err=final['error_changes'][n]
        lines.append(f"| {n} | {100*ds[0]:+.2f}%p | [{100*ci['accuracy'][0]:+.2f}, {100*ci['accuracy'][1]:+.2f}] | {100*ds[1]:+.2f}%p | [{100*ci['macro_f1'][0]:+.2f}, {100*ci['macro_f1'][1]:+.2f}] | {err['corrected']} / {err['introduced']} |")
    lines+=['','95% 구간은 평가자 군집을 쌍체로 2,000회 재표집한 기술적 구간이다. 학습·모델 선택의 불확실성이나 다중비교 편향을 보정하지 않았다. 같은 사람·기록의 창은 독립 표본이 아니다.','',
        '## 자세별 최종 F1','', '| 자세 | '+' | '.join(names)+' |','| --- | ---: | ---: | ---: |']
    for label in CLASS_NAMES:lines.append('| '+label+' | '+' | '.join(f"{100*final['metrics'][n]['per_class_f1'][label]:.2f}%" for n in names)+' |')
    lines+=['','## 검증과 재현','',
        '- 원본/기존 모델의 파일 목록·SHA-256·수정시각 보존을 확인했다. 모델·프로토콜·예측 파일의 해시를 고정하고 전후 대조했다.',
        '- 각 모델의 실제 원시 창 → 해당 보정 → 특징 추출 → 저장 모델 재예측이 저장 확률과 1e-12 이내로 일치했다. 50:50 재결합도 동일했다.',
        '- 같은 사람·기록·정답·창 종료 시점을 확인했다. 모델의 학습 대상과 평가 사람의 중복이 없었다.',
        '- 혼동행렬로 Accuracy/Macro-F1을 다시 계산하고 별도 sklearn 기반 지표와 교차 검증했다.',
        '- 신규 사용자, 재착석, 물리 센서, GUI 및 앱 빌드는 검증하지 않았다.',
        f'- [프로토콜·전체 순위]({rel}/protocol.json) · [결과 JSON]({rel}/summary.json) · [테스트 로그]({rel}/validation/tests.log)',
        f'- [최종 앙상블 정의]({rel}/final/holdout/ensemble.json) · [최종 예측]({rel}/final/holdout/predictions.npz) · [혼동행렬·개인별 결과]({rel}/final/holdout/metrics.json)',
        '- 앙상블 정의는 기존 두 모델 파일과 해시를 참조한다. 원 모델을 중복 복사하거나 운영 모델로 등록하지 않았다.','',
        '```sh',f'.venv/bin/python -m experiments.sensor_gain.top_two_ensemble --criterion {p["criterion"]}','```','']
    path=ROOT/'docs/reports/2026-09-23_top_two_equal_ensemble_report.md';path.write_text('\n'.join(lines),encoding='utf-8')
    write_json(out/'analysis_verification.json',{'finished_at':now(),'report':str(path),'report_sha256':digest(path),'summary_sha256':digest(out/'summary.json'),'raw_and_active_models_unchanged':True})
    print('REPORT',path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    parser.add_argument('--criterion',choices=('accuracy','macro_f1','diverse_macro_f1'),default='accuracy')
    args=parser.parse_args();out=args.output or ROOT/'research/results'/(datetime.now().strftime('%Y-%m-%d_%H%M%S')+'_top_two_equal_ensemble')
    print('OUTPUT',out,flush=True);run(out.resolve(),args.criterion)
