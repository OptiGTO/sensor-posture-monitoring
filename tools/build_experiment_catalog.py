"""Build read-only artifact indexes; only docs/catalog is written. No ML imports."""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'catalog'
SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', 'build', 'dist', '.cache', '.pyinstaller-cache', '.next', 'catalog'}
TITLES = {'data': '측정 원본·측정 부속 기록', 'models': '모델 파일', 'reports': '보고서·문서', 'figures': '그래프·도표', 'results': '평가 결과·예측·가공 데이터', 'metadata': '모델 설정·실험 메타데이터', 'code': '실험 실행 코드', 'archives': '압축 보관본'}
MODEL_EXT = {'.joblib', '.pkl', '.pickle', '.pt', '.pth', '.onnx', '.h5', '.keras', '.ckpt'}
ROOTS = {'data', 'research', 'docs', 'experiments', 'example'}
DATE = re.compile(r'(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)')
TIME_KEYS = ('started_at', 'recorded_at', 'measured_at', 'finished_at', 'completed_at', 'trained_at', 'created_at', 'generated_at', 'timestamp')


def walk(directory):
    for p in sorted(directory.iterdir()):
        if p.is_symlink() or p.name.startswith('.') or p.name in SKIP:
            continue
        if p.is_dir():
            yield from walk(p)
        elif p.is_file():
            yield p


def category(p):
    rel = p.relative_to(ROOT)
    if rel.parts[0] == 'data':
        return 'data'
    if p.suffix.lower() in MODEL_EXT:
        return 'models'
    if p.suffix.lower() == '.zip':
        return 'archives'
    if p.suffix.lower() in {'.md', '.html', '.pdf', '.docx', '.hwp'} or (rel.parts[0] == 'docs' and p.suffix.lower() == '.txt'):
        return 'reports'
    if p.suffix.lower() in {'.png', '.svg', '.jpg', '.jpeg'}:
        return 'figures'
    if rel.parts[0] == 'experiments' and p.suffix == '.py':
        return 'code'
    if p.suffix in {'.json', '.yaml', '.yml'}:
        return 'metadata' if rel.parts[:2] == ('research', 'models') or any(s in p.name for s in ('manifest', 'config', 'dataset', 'environment', 'selection')) else 'results'
    if p.suffix in {'.csv', '.jsonl', '.npz', '.npy', '.log', '.txt'}:
        return 'results'
    return None


def metadata(p):
    if p.suffix != '.json' or p.stat().st_size > 10_000_000:
        return {}
    try:
        value = json.loads(p.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (ValueError, UnicodeError):
        return {}


def time_evidence(p):
    candidates = [p] if p.suffix == '.json' else []
    # Only a same-stem measurement sidecar is associated; never infer from neighbours.
    if p.suffix == '.csv':
        sidecar = p.with_suffix('.manifest.json')
        if sidecar.is_file():
            candidates.append(sidecar)
    evidence = []
    for source in candidates:
        d = metadata(source)
        for key in TIME_KEYS:
            value = d.get(key)
            if isinstance(value, str):
                try:
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    continue
                evidence.append(f'{key}={value} ({source.name})')
    return '; '.join(evidence) or '미기록'


def path_date(path):
    dates = []
    for m in DATE.finditer(path):
        try:
            dates.append(datetime(*map(int, m.groups())).date().isoformat())
        except ValueError:
            pass
    return ', '.join(dict.fromkeys(dates)) or '미상'


def link(path):
    return f'[{path}]({quote("../../" + path, safe="/")})'


def cell(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


EXCLUSIONS = ROOT / 'research' / 'models' / 'training_exclusions.json'


def training_exclusions():
    """NFC data-relative path -> (status, entry). Same rule as src.training_exclusions:
    applied only when both path and SHA-256 match; read only."""
    import hashlib
    import unicodedata
    if not EXCLUSIONS.is_file():
        return {}
    result = {}
    for entry in json.loads(EXCLUSIONS.read_text(encoding='utf-8')).get('exclusions', []):
        relative = unicodedata.normalize('NFC', entry['source_file'])
        source = ROOT / 'data' / relative
        actual = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        status = '제외' if actual == entry['source_sha256'] else '제외 목록 해시 불일치(적용 안 됨)'
        result['data/' + relative] = (status, entry)
    return result


def main():
    files = []
    for p in sorted(ROOT.iterdir()):
        if p.is_symlink():
            continue
        if p.name in ROOTS and p.is_dir():
            files.extend(walk(p))
        elif p.is_file() and (p.suffix in MODEL_EXT | {'.zip', '.md', '.jsonl'}):
            files.append(p)
    review = ROOT / 'src/data_review_report.html'
    if review.exists():
        files.append(review)
    rows = []
    excluded = training_exclusions()
    import unicodedata
    for p in files:
        kind = category(p)
        if kind:
            rel = p.relative_to(ROOT).as_posix()
            rows.append(dict(category=kind, path=rel, recorded_time=time_evidence(p), path_date=path_date(rel), modified_kst=datetime.fromtimestamp(p.stat().st_mtime, ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'), bytes=p.stat().st_size))
            if kind == 'data' and unicodedata.normalize('NFC', rel) in excluded:
                status, entry = excluded[unicodedata.normalize('NFC', rel)]
                rows[-1]['model_training'] = {'status': status, 'reason': entry['reason'], 'requested_at': entry['requested_at']}
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    (OUT / 'inventory.json').write_text(json.dumps({'generated_at': stamp, 'files': rows}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    intro = f'생성 시각: {stamp}. 현재 실제 저장 경로를 색인합니다. data/는 읽기만 합니다.\n\n기록 시각은 JSON의 명시적 필드이며 `created_at`은 저장/생성 시각으로 측정 시작과 다를 수 있습니다. 시간대 없는 값은 원문 그대로입니다. 경로 날짜는 이름에 포함된 날짜일 뿐 실제 실행·측정일을 보증하지 않습니다. 파일 수정 시각(KST)은 별도 참고값입니다. 미기록/미상은 추정하지 않습니다.\n\n'
    for kind, title in TITLES.items():
        subset = [r for r in rows if r['category'] == kind]
        lines = [f'# {title}\n', intro, '[통합 안내](README.md) · [날짜별 찾기](timeline.md)\n', f'총 {len(subset)}개. 경로를 클릭하면 원본을 엽니다.\n']
        if kind == 'data':
            marked = [r for r in subset if 'model_training' in r]
            lines += [f'## 모델 학습 제외 기록: {len(marked)}개\n',
                      '`research/models/training_exclusions.json`에서 경로와 SHA-256이 모두 일치하는 기록입니다. 앱 학습·실험 코호트에서 빠지며, 원본 CSV는 삭제·수정하지 않습니다. 아래 표의 `모델 학습` 열은 이 목록만 반영합니다. 빈칸이 학습 포함을 뜻하지는 않습니다(측정 전용, 품질 실패, 7자세 미충족 등은 앱이 따로 판정합니다).\n',
                      '| 파일 위치 | 상태 | 사유 | 요청 시각 |', '| --- | --- | --- | --- |']
            lines += ['| ' + ' | '.join([link(r['path']), r['model_training']['status'], cell(r['model_training']['reason']),
                                         r['model_training']['requested_at']]) + ' |' for r in marked]
            lines += ['\n## 전체 목록\n', '| 파일 위치 | 모델 학습 | 기록 시각과 근거 | 경로 날짜 | 수정 시각(KST) |', '| --- | --- | --- | --- | --- |']
            for r in subset:
                lines.append('| ' + ' | '.join([link(r['path']), r.get('model_training', {}).get('status', ''), cell(r['recorded_time']), r['path_date'], r['modified_kst']]) + ' |')
        else:
            lines += ['| 파일 위치 | 기록 시각과 근거 | 경로 날짜 | 수정 시각(KST) |', '| --- | --- | --- | --- |']
            for r in subset:
                lines.append('| ' + ' | '.join([link(r['path']), cell(r['recorded_time']), r['path_date'], r['modified_kst']]) + ' |')
        (OUT / f'{kind}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    groups = defaultdict(list)
    for r in rows:
        # Timeline explicitly groups by naming convention, not a claimed execution date.
        groups[r['path_date']].append(r)
    lines = ['# 경로에 표시된 날짜별 찾기\n', intro, '[통합 안내](README.md)\n', '이 목록은 경로 날짜별 탐색입니다. 실제 시각 근거는 각 범주 색인에서 확인하세요. 날짜 없는 과거 원본은 범주 색인에 모두 포함됩니다.\n']
    for date, group in sorted(groups.items(), reverse=True):
        if date == '미상':
            lines.append(f'\n## 날짜 없는 경로: {len(group)}개\n\n범주별 색인 참조. 수정일로 측정일을 대체하지 않았습니다.\n')
            continue
        lines.append(f'\n## {date} ({len(group)}개)\n')
        for r in group:
            lines.append(f'- {TITLES[r["category"]]}: {link(r["path"])}')
    (OUT / 'timeline.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    counts = Counter(r['category'] for r in rows)
    lines = ['# 실험 자료 통합 안내\n', intro, 'research/의 실제 저장 위치와 data/의 측정 원본을 한곳에서 찾는 색인입니다. 새 실험 후 `python3 tools/build_experiment_catalog.py`로 갱신합니다. 이 명령은 `docs/catalog/`의 생성 파일만 씁니다.\n', '| 찾을 자료 | 파일 수 | 주요 원본 위치 |', '| --- | ---: | --- |']
    locations = {'data':'data/', 'models':'research/models/, research/results/ 실험별 폴더', 'reports':'docs/reports/, docs/research/, docs/ 설계 문서, 루트 안내 문서', 'figures':'research/figures/, research/results/', 'results':'research/results/, research/models/legacy_outputs/, example/, realtime_predictions.jsonl', 'metadata':'research/models/, research/results/', 'code':'experiments/posture_search/, experiments/posture_study/', 'archives':'research/archives/'}
    for kind, title in TITLES.items():
        lines.append(f'| [{title}]({kind}.md) | {counts[kind]} | {locations[kind]} |')
    lines.extend(['\n[날짜별 찾기](timeline.md) · [전체 기계 판독 목록](inventory.json) · [현재 구조와 운영 규칙](STRUCTURE_PROPOSAL.md)\n', '## 읽는 순서\n', '1. 측정 파일은 측정 원본 색인에서 이름을 찾습니다. `.manifest.json`이 있으면 CSV 행에도 해당 시각을 연결합니다.\n2. 실험 보고서는 보고서 색인에서 찾고, 동일 실험 폴더 이름으로 모델·결과 색인을 검색합니다.\n3. 모델의 사용/승인 여부는 `artifacts/active_model.json`, `release_selection.json`, 개별 manifest와 배포 기록을 확인합니다. 파일이 존재하거나 최신이라고 승인 모델인 것은 아닙니다.\n', '## 범위와 한계\n', '프로젝트의 data/research/docs/experiments/example 및 루트 문서·압축본·모델·예측 로그와 src의 데이터 검토 보고서를 포함합니다. 의존성, 빌드/배포 복제본, 숨김 파일, 캐시, 심볼릭 링크는 제외합니다. 압축본 내부와 설치 앱의 외부 사용자 상태 폴더는 조사하지 않습니다. 모델은 역직렬화하지 않습니다. 같은 범주로 묶었으며 내용 중복을 의미하지 않습니다.\n', '`results/`, `outputs/` 및 일부 학습 복사본은 Git 제외 대상이므로 다른 컴퓨터에는 색인 링크의 원본이 없을 수 있습니다. 색인은 로컬 스냅샷이며 결과 백업을 대체하지 않습니다. 측정 전용 기록이나 training_data 복사본을 학습 적격 원본으로 판정하지 않습니다.\n'])
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(total=len(rows), categories=counts), ensure_ascii=False))


if __name__ == '__main__':
    main()
