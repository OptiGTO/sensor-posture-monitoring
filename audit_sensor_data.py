"""Generate a read-only audit, per-record charts, and bounded preprocessing comparison."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

from src.data_review import audit, render_plot, verify_unchanged, write_json


def prepare_windows(report, frames, output):
    """Export existing, reversible 12 Hz/5-second preprocessing with source linkage."""
    import numpy as np
    from src.posture_core import CLASS_NAMES
    from src.window_pipeline import WindowSpec, segment_sensor_frame
    arrays, labels, groups, record_ids, starts = [], [], [], [], []
    for r in report['records']:
        if not r['eligible']:
            continue
        windows, meta = segment_sensor_frame(frames[r['id']], WindowSpec(window_seconds=5))
        arrays.extend(windows);labels.extend([CLASS_NAMES.index(r['label'])]*len(windows))
        groups.extend([r['subject']]*len(windows));record_ids.extend([r['id']]*len(windows))
        starts.extend(meta['start_s'].tolist())
    if not arrays:
        return None
    target=output/'prepared_windows_5s.npz'
    np.savez_compressed(target, sequences=np.stack(arrays), labels=np.array(labels,dtype=np.int64),
                        subjects=np.array(groups), record_ids=np.array(record_ids), start_seconds=np.array(starts))
    summary={'file':target.name,'window_count':len(arrays),'shape':[len(arrays),60,16],
             'source_records':sum(r['eligible'] for r in report['records']),
             'steps':['existing canonical label aliases','12 Hz elapsed-time interpolation','nonoverlapping 5-second windows'],
             'not_applied':['human-unconfirmed label proposals','trimming','spike removal','empty-seat correction','dataset-wide learned scaling'],
             'status':'structurally_valid_review_warnings_retained',
             'source_linkage':'record_ids refer to audit.json records; keep subject split when using these windows'}
    write_json(output/'prepared_windows_5s.manifest.json',summary)
    return summary


def compare_preprocessing(report, frames, evidence_path, output):
    """Select using training-subject CV; score validation once; never score final subjects."""
    import numpy as np
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.svm import SVC
    from src.posture_core import feature_names, CLASS_NAMES
    from src.window_pipeline import WindowSpec, extract_window_features, segment_sensor_frame

    evidence = json.loads(evidence_path.read_text())
    parts = evidence['partitions']
    subjects = {k: set(v['subjects']) for k, v in parts.items()}
    assert not (subjects['train'] & subjects['validation'] or subjects['train'] & subjects['final_evaluation']
                or subjects['validation'] & subjects['final_evaluation'])
    current = {r['sha256']: r for r in report['records'] if r['eligible']}
    sequences, labels, groups, partitions, provenance = [], [], [], [], []
    for source in evidence['source_records']:
        subject = source['subject']
        if subject in subjects['final_evaluation']:
            continue
        record = current.get(source['sha256'])
        if record is None:
            raise ValueError(f"학습 당시 원본을 찾을 수 없거나 구조 오류가 있음: {source['record_id']}")
        if record['label'] != source['label']:
            raise ValueError("원본의 실제 라벨이 학습 당시와 달라졌습니다")
        windows, _ = segment_sensor_frame(frames[record['id']], WindowSpec(window_seconds=5))
        sequences.extend(windows); labels.extend([CLASS_NAMES.index(source['label'])] * len(windows))
        groups.extend([subject] * len(windows))
        part = 'train' if subject in subjects['train'] else 'validation'
        partitions.extend([part] * len(windows))
        provenance.append({'sha256': source['sha256'], 'current_file': record['file'], 'subject': subject, 'partition': part})
    X = extract_window_features(np.stack(sequences)); y = np.array(labels); groups = np.array(groups)
    partitions = np.array(partitions); train = partitions == 'train'; val = partitions == 'validation'
    names = feature_names()
    spatial = np.array([i for i,n in enumerate(names) if any(n.startswith(prefix+stat) for prefix in ('abs_','rel_')
                        for stat in ('mean_R','std_R','median_R','row_mean_','column_mean_','mean_global',
                                     'mean_left_right','mean_front_back','mean_diagonal','mean_outer_row','mean_outer_column'))])
    # Fixed, declared comparisons. All learned scaling/feature selection is fit inside each fold.
    specs = [
        ('기존 설정 / 596 / StandardScaler', 'full', 'standard', 10., .003, None),
        ('규제 비교 / 596 / StandardScaler', 'full', 'standard', 1., 'scale', None),
        ('596 / RobustScaler', 'full', 'robust', 1., 'scale', None),
        (f'공간·기본 통계 / {len(spatial)} / StandardScaler', 'spatial', 'standard', 1., 'scale', None),
        (f'공간·기본 통계 / {len(spatial)} / RobustScaler', 'spatial', 'robust', 1., 'scale', None),
        ('학습 fold에서 120개 선택 / StandardScaler', 'full', 'standard', 1., 'scale', 120),
    ]
    def pipeline(spec):
        _, _, scaler, c, gamma, k = spec
        steps = [('scale', StandardScaler() if scaler == 'standard' else RobustScaler())]
        if k:
            steps.append(('select', SelectKBest(f_classif, k=k)))
        # Probability calibration matches the existing runtime prediction route.
        steps.append(('model', SVC(C=c, gamma=gamma, class_weight='balanced', probability=True, random_state=20260826)))
        return Pipeline(steps)
    def prediction(model, x):
        return model.classes_[model.predict_proba(x).argmax(axis=1)]
    results = []
    cv = list(GroupKFold(n_splits=4).split(X[train], y[train], groups[train]))
    for spec in specs:
        name, feature_set, scaler, c, gamma, k = spec
        data = X if feature_set == 'full' else X[:, spatial]
        xt, yt = data[train], y[train]
        oof = np.full(len(yt), -1); folds = []
        for a,b in cv:
            assert not set(groups[train][a]) & set(groups[train][b])
            model = pipeline(spec); model.fit(xt[a], yt[a]); pred = prediction(model, xt[b]); oof[b] = pred
            folds.append({'held_subjects': sorted(set(groups[train][b])),
                          'accuracy': float(accuracy_score(yt[b],pred)),
                          'macro_f1': float(f1_score(yt[b],pred,labels=range(7),average='macro',zero_division=0))})
        result = {'name':name, 'feature_set':feature_set, 'scaler':scaler,'C':c,'gamma':gamma,'selected_feature_count':k or data.shape[1],
                  'cv_accuracy':float(accuracy_score(yt,oof)),
                  'cv_macro_f1':float(f1_score(yt,oof,labels=range(7),average='macro',zero_division=0)),
                  'folds':folds}
        results.append(result)
        print(f"전처리 CV: {name}: Macro F1={result['cv_macro_f1']:.4f}",flush=True)
    selected = max(range(len(results)), key=lambda i:results[i]['cv_macro_f1'])
    comparison = {'status':'cv_selection_locked', 'selection_basis':'highest pooled 4-fold training-subject OOF Macro F1',
                  'selected_index':selected, 'selected_name':results[selected]['name'],
                  'selection_locked_at':datetime.now().astimezone().isoformat(),
                  'reference_evidence':str(evidence_path),'reference_seed':evidence['random_seed'],
                  'final_evaluation_subjects_not_scored':sorted(subjects['final_evaluation']),
                  'train_subject_count':len(subjects['train']),'validation_subject_count':len(subjects['validation']),
                  'source_records':provenance,'rows':results,
                  'limitations':['Small exploratory comparison; these validation people have appeared in earlier reports.',
                                 'Signal warnings retained; no human-unverified label correction or trimming.',
                                 'No final-set scoring, production artifact, activation or improvement claim from final data.',
                                 'SVC probability fitting is internal to training; scores are not established calibrated probabilities.']}
    write_json(output/'preprocessing_comparison.json', comparison)
    for i,(spec,row) in enumerate(zip(specs,results)):
        data = X if spec[1]=='full' else X[:,spatial]
        model = pipeline(spec); model.fit(data[train], y[train])
        pred = prediction(model,data[val]); fit_pred=prediction(model,data[train])
        row.update(validation_accuracy=float(accuracy_score(y[val],pred)),
                   validation_macro_f1=float(f1_score(y[val],pred,labels=range(7),average='macro',zero_division=0)),
                   training_accuracy=float(accuracy_score(y[train],fit_pred)), selected=(i==selected))
    comparison['status']='completed';write_json(output/'preprocessing_comparison.json',comparison)
    return comparison


def render_report(report, comparison, output):
    payload = {'summary':report['summary'],'fingerprint':report['dataset_fingerprint'],
               'records':report['records'],'excluded':report['excluded'],'comparison':comparison,
               'labels':__import__('src.posture_core',fromlist=['CLASS_NAMES']).CLASS_NAMES}
    template = Path(__file__).parent/'src/data_review_report.html'
    html = template.read_text().replace('__REPORT_DATA__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
    (output/'index.html').write_text(html,encoding='utf-8')
    s=report['summary'];blockers=[r for r in report['records'] if r['blocking']]
    lines=['# 센서 데이터 자동 검사와 사용자 확인 순서','',
           f"CSV {s['csv_files']}개 / 학습 후보 {s['candidate_records']}개 / 기본 구조·라벨 통과 {s['eligible_records']}개 / 별도 유지 {s['excluded_records']}개",'',
           '원본 CSV와 메타데이터는 변경하지 않았습니다. 그래프의 색은 센서 기록값이며 힘·압력 단위나 실제 좌우 방향이 검증된 것은 아닙니다.', '',
           '1. index.html을 열고 `필수 확인`에서 실제 자세를 확인합니다. 알 수 없으면 `재수집 필요`를 선택합니다.',
           f"2. `우선 검토` {s['priority_review_count']}개에서 음영 구간과 같은 사람·자세의 비교 지도를 확인합니다. 실제 움직임이면 오류로 삭제하지 않습니다.",
           f"3. `표본 확인` {s['sample_review_count']}개를 확인합니다. {s['sample_subject_count']}명과 {s['sample_class_count']}개 자세를 포함하며 경고가 적은 기록부터 선정했습니다.",
           '4. 각 기록에 판단·확정 자세·근거 또는 문제 시간 구간을 입력하고 `검토 결과 JSON 저장`을 누릅니다.',
           '5. 저장한 JSON 파일 경로를 이 작업에 알려주면, 확인된 결정에 맞춰 원본과 분리된 처리본 및 학습 목록을 만들 수 있습니다.',
           f"1차 검토는 중복을 제외하면 {s['first_pass_unique_records']}개입니다. `신호 확인`의 나머지 경고는 1차 결과에서 문제가 확인된 피험자·자세를 중심으로 추가 확인합니다.", '',
           '## 먼저 확인할 라벨', '', '| 파일 | 현재 라벨 | 제안 (미확정) |','|---|---|---|']
    for r in blockers:
        lines.append(f"| {r['subject']} / {r['filename']} | {', '.join(r['raw_labels'])} | {r.get('suggested_label') or '수집 기록 확인 필요'} |")
    lines+=['','## 검증 경계','','신호 경고는 검사 후보입니다. 실제 센서 고장·라벨 오류로 자동 확정하지 않습니다.',
            '센서의 명목 범위·단위·빈 좌석 기준이 없어 물리 범위 초과·포화·실제 무게 보정은 판정하지 않았습니다.',
            '모델 오답이나 낮은 신뢰도는 품질 제외 기준으로 사용하지 않았습니다.',
            '전처리 실험은 기존 학습 피험자 내부 교차검증으로 선택하고 검증 피험자에서 확인했습니다. 최종 평가 피험자는 다시 채점하지 않았습니다.']
    if comparison:
        lines += ['', '## 전처리 비교', '', '| 방식 | 학습 피험자 CV Macro F1 | 검증 Accuracy | 검증 Macro F1 |', '|---|---:|---:|---:|']
        for row in comparison['rows']:
            lines.append(f"| {row['name']} | {row['cv_macro_f1']:.4f} | {row['validation_accuracy']:.2%} | {row['validation_macro_f1']:.4f} |")
        chosen=comparison['rows'][comparison['selected_index']]
        base=comparison['rows'][0]
        if chosen['validation_macro_f1'] <= base['validation_macro_f1']:
            lines += ['', 'CV 최고 설정도 기존 설정보다 검증 결과가 개선되지는 않았습니다. 이번 자동 비교로 성능 개선이 입증된 변경은 없습니다.']
    (output/'사용자_확인순서.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(output/'training_candidates.json',{'dataset_fingerprint':report['dataset_fingerprint'],
               'status':'structural_checks_only_not_human_approved','source_root':report['source_root'],
               'records':[{'file':r['file'],'sha256':r['sha256'],'subject':r['subject'],'label':r['label'],
                           'pending_signal_warnings':[f['code'] for f in r['flags']]} for r in report['records'] if r['eligible']]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-evaluation',type=Path)
    parser.add_argument('--reuse-comparison',type=Path,help='Reuse an unchanged completed comparison when only review heuristics or plots were updated')
    parser.add_argument('--skip-plots',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False);(output/'plots').mkdir()
    print('전체 파일 자동 검사 시작',flush=True)
    report,frames=audit(args.data.resolve());write_json(output/'audit.json',report)
    print(json.dumps(report['summary'],ensure_ascii=False),flush=True)
    report['prepared_data']=prepare_windows(report,frames,output)
    comparison=None
    if args.reference_evaluation:
        comparison=compare_preprocessing(report,frames,args.reference_evaluation.resolve(),output)
    elif args.reuse_comparison:
        comparison=json.loads(args.reuse_comparison.read_text())
        if comparison.get('status')!='completed':raise ValueError('완료되지 않은 비교 결과입니다')
        available={r['sha256'] for r in report['records'] if r['eligible']}
        if any(r['sha256'] not in available for r in comparison['source_records']):
            raise ValueError('전처리 비교에 사용한 원본이나 구조 검증 결과가 달라졌습니다')
        comparison['reused_from']=str(args.reuse_comparison.resolve())
        write_json(output/'preprocessing_comparison.json',comparison)
    if not args.skip_plots:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import font_manager
        fonts={f.name for f in font_manager.fontManager.ttflist}
        matplotlib.rcParams['font.family']=next((x for x in ['AppleGothic','NanumGothic','Noto Sans CJK KR'] if x in fonts),'DejaVu Sans')
        matplotlib.rcParams['axes.unicode_minus']=False
        for i,r in enumerate(report['records'],1):
            if r['id'] in frames:render_plot(r,frames[r['id']],output/'plots')
            if i%50==0: print(f'그래프 생성 {i}/{len(report["records"])}',flush=True)
    verify_unchanged(report);write_json(output/'audit.json',report)
    render_report(report,comparison,output)
    print('완료:',output,flush=True)


if __name__=='__main__':
    main()
