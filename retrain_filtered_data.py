"""Retrain after excluding every flagged audit record while preserving subject roles."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil

import joblib
import numpy as np

from src.app_services import TrainingJob
from src.data_review import digest, write_json
from src.model_comparison import DEFAULT_MODEL_FAMILIES, scores, train_comparison
from src.posture_core import CLASS_NAMES, read_sensor_csv, recording_label, training_csv_paths
from src.window_pipeline import WindowSpec, aligned_predict_proba, extract_window_features, segment_sensor_frame


def stage_selection(audit_path, evidence_path, output):
    audit=json.loads(audit_path.read_text());evidence=json.loads(evidence_path.read_text())
    root=Path(audit['source_root']).resolve()
    current={p.resolve() for p in training_csv_paths(root)}
    expected={(root/r['file']).resolve() for r in audit['records']}
    if current!=expected:
        raise ValueError('학습 후보 파일 목록이 검사 때와 달라졌습니다. 자동 검사를 갱신하세요.')
    kept=[];excluded=[]
    for r in audit['records']:
        path=(root/r['file']).resolve()
        if not path.is_relative_to(root) or digest(path)!=r['sha256']:
            raise ValueError(f"검사 당시 원본과 다름: {r['file']}")
        if r['eligible'] and not r['flags']:
            kept.append(r)
        else:
            excluded.append({'file':r['file'],'sha256':r['sha256'],'subject':r['subject'],
                             'label':r['label'],'reasons':r['flags']})
    if not kept:raise ValueError('경고를 제외한 학습 기록이 없습니다.')
    ref_by_hash={r['sha256']:r for r in evidence['source_records']}
    for r in kept:
        reference=ref_by_hash.get(r['sha256'])
        if not reference or reference['subject']!=r['subject'] or reference['label']!=r['label']:
            raise ValueError('기존 피험자 분리와 원본·피험자·라벨을 대응할 수 없습니다.')
    present={r['subject'] for r in kept}
    fixed={name:sorted(set(part['subjects'])&present) for name,part in evidence['partitions'].items()}
    flattened=[s for group in fixed.values() for s in group]
    if len(flattened)!=len(set(flattened)) or set(flattened)!=present:
        raise ValueError('기존 피험자 분리를 보존할 수 없습니다.')
    coverage={}
    for name,members in fixed.items():
        rows=[r for r in kept if r['subject'] in members]
        counts=Counter(r['label'] for r in rows)
        if set(counts)!=set(CLASS_NAMES):raise ValueError(f'{name}에 7개 자세가 모두 남지 않습니다.')
        coverage[name]={'subjects':members,'record_count':len(rows),'class_record_counts':dict(counts)}
    stage=output/'training_data';stage.mkdir(parents=True)
    for r in kept:
        source=root/r['file'];target=stage/r['file'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        sidecar=source.with_suffix('.manifest.json')
        if sidecar.is_file():shutil.copy2(sidecar,target.with_suffix('.manifest.json'))
        if digest(target)!=r['sha256']:raise RuntimeError('학습 사본의 해시가 일치하지 않습니다.')
    manifest={'schema_version':1,'user_instruction':'의심이 되는 자세는 다 제외하고 학습을 다시 진행',
              'interpretation':'Exclude individual recording files with any audit warning or blocking issue; preserve all seven posture classes.',
              'audit_file':str(audit_path),'audit_sha256':digest(audit_path),'audit_fingerprint':audit['dataset_fingerprint'],
              'reference_evidence':str(evidence_path),'reference_evidence_sha256':digest(evidence_path),
              'source_root':str(root),'included_record_count':len(kept),'excluded_record_count':len(excluded),
              'included':[{'file':r['file'],'sha256':r['sha256'],'subject':r['subject'],'label':r['label']} for r in kept],
              'excluded':excluded,'fixed_subject_partitions':fixed,'coverage':coverage,
              'excluded_is_human_confirmed_fault':False,'source_deletion_performed':False}
    write_json(output/'exclusion.manifest.json',manifest)
    return audit,evidence,manifest,stage


def evaluate_same_records(report,evidence,evidence_path,selection,output):
    chosen=next(r for r in report['models'] if r['selected'])
    new_path=output/'artifacts'/chosen['artifact_file']
    old_config_path=evidence_path.parent/'window_5s_rbf_svm_model_config.json'
    old_config=json.loads(old_config_path.read_text());old_path=old_config_path.parent/old_config['model_file']
    old=joblib.load(old_path)['model'];new=joblib.load(new_path)['model']
    final_subjects=set(evidence['partitions']['final_evaluation']['subjects'])
    clean_hashes={r['sha256'] for r in selection['included']}
    root=Path(selection['source_root']);current_by_hash={digest(p):p for p in training_csv_paths(root)}
    windows=[];truth=[];clean=[];subjects=[];sources=[];record_ids=[]
    for r in evidence['source_records']:
        if r['subject'] not in final_subjects:continue
        source=current_by_hash.get(r['sha256'])
        if source is None:raise ValueError('기존 전체 평가 기록의 원본을 찾을 수 없습니다.')
        frame=read_sensor_csv(source)
        if recording_label(frame)[1]!=r['label']:raise ValueError('평가 라벨이 달라졌습니다.')
        seq,_=segment_sensor_frame(frame,WindowSpec(window_seconds=5))
        windows.extend(seq);truth.extend([CLASS_NAMES.index(r['label'])]*len(seq))
        clean.extend([r['sha256'] in clean_hashes]*len(seq));subjects.extend([r['subject']]*len(seq))
        record_ids.extend([r['record_id']]*len(seq));sources.append({'record_id':r['record_id'],'sha256':r['sha256'],'subject':r['subject'],'retained_by_filter':r['sha256'] in clean_hashes})
    X=extract_window_features(np.stack(windows));y=np.array(truth);clean=np.array(clean);subjects=np.array(subjects);record_ids=np.array(record_ids)
    result={'status':'completed','selection_already_locked':report['selection_locked'],
            'selection_locked_at':report['selection_locked_at'],'selection_basis':report['selection_policy'],
            'evaluation_role':'Diagnostic reuse of historical final subjects; not a new independent external test.',
            'new_model':{'name':chosen['name'],'file':str(new_path),'sha256':digest(new_path)},
            'baseline_model':{'name':'기존 5초 RBF SVM','file':str(old_path),'sha256':digest(old_path)},
            'source_records':sources,'cohorts':{}}
    predictions={name:aligned_predict_proba(model,X,7).argmax(1) for name,model in [('baseline',old),('retrained',new)]}
    if not np.array_equal(scores(y,predictions['baseline'])['confusion_matrix'],evidence['metrics']['confusion_matrix']):
        raise ValueError('기존 전체 평가 결과를 재현하지 못했습니다.')
    for name,mask in [('original_full_final',np.ones(len(y),bool)),('retained_clean_final',clean),('excluded_flagged_final',~clean)]:
        result['cohorts'][name]={'record_count':len(set(record_ids[mask])),'window_count':int(mask.sum()),
                                'subject_count':len(set(subjects[mask])),
                                **{kind:scores(y[mask],p[mask]) for kind,p in predictions.items()}}
    final_saved=chosen['final_evaluation']
    if not np.array_equal(result['cohorts']['retained_clean_final']['retrained']['confusion_matrix'],final_saved['confusion_matrix']):
        raise ValueError('필터 적용 평가 결과가 학습 보고서와 일치하지 않습니다.')
    result['baseline_full_confusion_matrix_reproduced']=True
    result['retrained_clean_confusion_matrix_reproduced']=True
    write_json(output/'same_record_comparison.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--reference-evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent;output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    audit,evidence,selection,stage=stage_selection(args.audit.resolve(),args.reference_evaluation.resolve(),output)
    print(f"경고 제외: {selection['excluded_record_count']}개 / 유지: {selection['included_record_count']}개",flush=True)
    job=TrainingJob(root,(),evidence['random_seed'],windows=(5,),state_root=root);job.run_dir=output
    log=output/'training.log'
    def progress(message):
        with log.open('a') as file:file.write(message+'\n')
        print(message,flush=True)
    run={'status':'running','approval_status':'pending','windows':[5],'models':list(DEFAULT_MODEL_FAMILIES),'seed':evidence['random_seed'],
         'selected_records':[str(Path(selection['source_root'])/r['file']) for r in selection['included']],
         'filter_manifest':'exclusion.manifest.json','comparison_report':str(job.comparison_report_path)}
    write_json(output/'training.manifest.json',run)
    try:
        report=train_comparison(stage,output,(5,),evidence['random_seed'],fixed_subject_partitions=selection['fixed_subject_partitions'],progress=progress)
        comparison=evaluate_same_records(report,evidence,args.reference_evaluation.resolve(),selection,output)
        job.candidate_manifest_paths=job._write_pending_candidate_manifests()
        for path in job.candidate_manifest_paths:
            payload=json.loads(path.read_text());payload['display_name']='경고 제외 · '+payload['display_name']
            payload['provenance']['created_by']='Filtered data retraining'
            payload['metadata'].update(training_filter='exclude every audit flag',excluded_record_count=selection['excluded_record_count'])
            write_json(path,payload)
        validations=job._validate_candidates()
        for r in audit['records']:
            if digest(Path(selection['source_root'])/r['file'])!=r['sha256']:raise RuntimeError('원본 학습 후보가 실행 중 변경되었습니다.')
        run.update(status='completed',exit_code=0,completed_at=datetime.now().astimezone().isoformat(),
                   candidate_manifests=[str(p) for p in job.candidate_manifest_paths],validation_results=validations,
                   raw_training_candidates_unchanged=True,selected=report['selected'])
        write_json(output/'training.manifest.json',run)
        lines=['# 자동 경고 기록 제외 후 재학습','',f"학습 후보 371개 중 {selection['excluded_record_count']}개 제외, {selection['included_record_count']}개 유지. 원본을 삭제하거나 수정하지 않았습니다.",
               '', '기존 피험자 분리를 유지하고 5초 조건으로 4개 모델을 비교했습니다. 최종 평가 점수로 모델을 다시 고르지 않았습니다.',
               '', '| 구분 | 기록 수 | 피험자 수 |','|---|---:|---:|']
        for part,info in selection['coverage'].items():lines.append(f"| {part} | {info['record_count']} | {len(info['subjects'])} |")
        lines+=['',f"선택 모델: {report['selected']['name']} / 설정: {report['selected']['params']}",'',
                '| 동일 평가 기록 | 기록 수 | 기존 SVM Accuracy | 재학습 Accuracy | 기존 Macro F1 | 재학습 Macro F1 |','|---|---:|---:|---:|---:|---:|']
        for name,label in [('retained_clean_final','경고 없는 평가 기록'),('original_full_final','기존 전체 평가 기록'),('excluded_flagged_final','경고로 제외한 평가 기록')]:
            row=comparison['cohorts'][name]
            lines.append(f"| {label} | {row['record_count']} | {row['baseline']['accuracy']:.2%} | {row['retrained']['accuracy']:.2%} | {row['baseline']['macro_f1']:.4f} | {row['retrained']['macro_f1']:.4f} |")
        lines+=['','점수는 5초 구간 단위입니다. 기존 평가 피험자를 다시 사용한 진단 결과이며 새 외부 평가가 아닙니다.',
                '경고가 없는 기록도 실제 라벨·센서 상태가 사람에 의해 확인된 것은 아닙니다.',
                '새 모델은 후보로 저장했으며 활성 모델을 변경하지 않았습니다.',
                f"Runtime·배포 소프트웨어 검사: {sum(bool(v['passed']) for v in validations)}/{len(validations)}개 통과. 실제 센서 사용 검증은 포함하지 않습니다."]
        (output/'재학습_결과.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps({'selected':report['selected'],'cohorts':{k:{'baseline':v['baseline']['accuracy'],'retrained':v['retrained']['accuracy']} for k,v in comparison['cohorts'].items()},'validation_passed':[v['passed'] for v in validations],'output':str(output)},ensure_ascii=False),flush=True)
    except BaseException as error:
        run.update(status='failed',failure_reason=str(error));write_json(output/'training.manifest.json',run);raise


if __name__=='__main__':main()
