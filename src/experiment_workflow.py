"""Configuration registry and adapters for the shared experiment workflow.

Definitions describe existing trainers; adding a definition does not invent a new
learning algorithm. Saved reports are adapted in memory, never rewritten.
"""
from dataclasses import dataclass
from pathlib import Path
import json
from .model_catalog import MODEL_NAMES
from .posture_core import CLASS_NAMES
from .soft_voting import validate_members
from .training_methods import validate_training_method


@dataclass(frozen=True)
class Configuration:
    id: str
    label: str
    description: str
    editor: str
    personalized: bool = False
    ensemble: bool = False


SELECTION_METRIC_LABELS = {
    'loso': 'LOSO · 개발 피험자 한 명씩 제외 (기본)',
    'validation': '단일 검증 분할 · 빠름',
}


CONFIGURATIONS = {
    item.id: item for item in (
        Configuration('individual', '개별 모델', '기존 모델·분석 창 비교', 'models'),
        Configuration('ensemble', '앙상블', '동일 가중 Soft voting', 'models', ensemble=True),
        Configuration('personal', '개인 표준화', '등록 기준을 사용하는 기존 2단계 구성', 'personal', personalized=True),
    )
}


def build_configuration(state, *, allow_retrospective=False):
    definition = CONFIGURATIONS[state.configuration_type]
    if definition.personalized and allow_retrospective:
        raise ValueError('과거 정자세 기록의 등록용 배정은 해제되었습니다. 실제 등록 정보가 연결된 데이터를 사용하세요.')
    if not state.experiment_name.strip():
        raise ValueError('실험 이름을 입력하세요.')
    if isinstance(state.seed, bool) or not 0 <= state.seed <= 2**32-1:
        raise ValueError('난수 시드는 0~4294967295 사이 정수여야 합니다.')
    # Personal training owns its fixed candidate grid; unrelated draft options
    # must never masquerade as settings used by that trainer.
    windows = (3,) if definition.personalized else state.windows
    models = ('rbf_svm', 'logistic_regression') if definition.personalized else state.models
    training_method = 'standard' if definition.personalized else state.training_method
    method = validate_training_method(training_method, models, windows)
    members = state.soft_voting_members if definition.ensemble else ()
    if definition.ensemble and len(members) < 2:
        raise ValueError('앙상블 구성 모델을 2개 이상 선택하세요.')
    validate_members(members, models, windows)
    selection_metric = getattr(state, 'selection_metric', 'loso')
    if selection_metric not in SELECTION_METRIC_LABELS:
        raise ValueError('모델 선택 기준은 LOSO 또는 검증 분할이어야 합니다.')
    return {'schema_version': 1, 'name': state.experiment_name.strip(), 'type_id': definition.id,
            'label': definition.label, 'seed': state.seed, 'windows': list(windows), 'models': list(models),
            'training_method': training_method, 'training_method_label': method['label'],
            'training_method_evidence': method.get('evidence'),
            **({'checkpoint': method['checkpoint']} if 'checkpoint' in method else {}),
            'soft_voting_members': list(members),
            'selection_metric': None if definition.personalized else selection_metric,
            'selection_metric_label': '개인 기준 고정 구성' if definition.personalized else SELECTION_METRIC_LABELS[selection_metric],
            'personalized': definition.personalized,
            'allow_retrospective': bool(allow_retrospective) if definition.personalized else False,
            'source_experiment': state.source_experiment,
            'settings_policy': 'fixed existing personal grid' if definition.personalized else 'selected models and windows'}


def validate_registration(records, configuration):
    if not configuration['personalized']:
        return
    by_subject = {}
    for record in records:
        sidecar = record.csv_path.with_suffix('.manifest.json')
        payload = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        linked = bool(payload.get('guidance', {}).get('reference_profile'))
        by_subject.setdefault(record.subject, []).append(linked)
    for subject, linked in by_subject.items():
        if any(linked) and not all(linked):
            raise ValueError(f'{subject}: 등록 정보가 연결된 기록과 없는 기록을 함께 사용할 수 없습니다.')
        if not any(linked):
            raise ValueError(f'{subject}: 등록 정보가 없습니다. 실제 등록 정보가 연결된 데이터를 선택하세요.')


def apply_configuration(state, configuration):
    if configuration.get('schema_version') != 1 or configuration.get('type_id') not in CONFIGURATIONS:
        raise ValueError('지원하지 않는 실험 구성입니다.')
    training_method = configuration.get('training_method', 'standard')
    if CONFIGURATIONS[configuration['type_id']].personalized and training_method != 'standard':
        raise ValueError('개인 표준화 구성은 기본 학습 방식만 지원합니다.')
    validate_training_method(training_method, configuration['models'], configuration['windows'])
    state.configuration_type = configuration['type_id']
    state.experiment_name = configuration['name'] + ' · 복제'
    state.seed = int(configuration['seed'])
    state.training_method = training_method
    if not CONFIGURATIONS[state.configuration_type].personalized:
        state.configure(configuration['windows'], state.seed, configuration['models'])
    # Older saved experiments predate LOSO and were selected on one validation split.
    state.selection_metric = configuration.get('selection_metric') or 'validation'
    if state.selection_metric not in SELECTION_METRIC_LABELS:
        raise ValueError('지원하지 않는 모델 선택 기준입니다.')
    state.soft_voting_enabled = CONFIGURATIONS[state.configuration_type].ensemble
    if CONFIGURATIONS[state.configuration_type].ensemble:
        state.soft_voting_members = tuple(configuration.get('soft_voting_members', ()))
        state.voter_selection_initialized = bool(state.soft_voting_members)


def adapt_report(report):
    if not isinstance(report, dict):
        raise ValueError('모델 비교 보고서 형식이 올바르지 않습니다.')
    if 'results' not in report or 'registration_records' not in report:
        return report
    names = {'single_7class': '단일 모델 (비교군)', 'generic_hierarchy': '일반 2단계 (비교군)',
             'personal_hierarchy': '개인 표준화 2단계'}
    rows = []
    for key, name in names.items():
        row = {'family': key, 'display_label': name, 'window_seconds': 3}
        for split in ('validation', 'final_evaluation'):
            metrics = dict(report.get('results', {}).get(split, {}).get(key) or {})
            f1s = [item['f1'] for item in metrics.get('per_class', {}).values() if 'f1' in item]
            if f1s:
                metrics['worst_class_f1'] = min(f1s)
            row[split] = metrics
        rows.append(row)
    return {**report, 'models': rows, 'labels': list(CLASS_NAMES), 'configuration_type': 'personal',
            'status': report.get('status', 'completed'),
            'summary_text': '개인 표준화 · 3초 고정 · 등록 기록을 분리한 동일 평가 창\n'
                + ('과거 기록 실험 · 빈 좌석 0 가정' if report.get('evidence_kind') == 'retrospective_zero_empty_baseline'
                   else '연결된 정자세 등록 기록 사용')
                + ' · 미확정 출력을 평가에서 제외하지 않습니다.'}
