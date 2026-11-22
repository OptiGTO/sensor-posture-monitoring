"""Guard direct CLI input against canonical, hash-bound training exclusions."""
from pathlib import Path
import unicodedata

from .training_exclusions import load_training_exclusions


def validate_cli_training_input(data_root: str | Path) -> None:
    """Require excluded originals to be omitted before a direct training call.

    The app stages only eligible records. A CLI caller supplying the original
    tree must instead prepare such a staging tree; this guard never rewrites or
    removes sources. Copied inputs retain their own caller-provided provenance.
    """
    root = Path(data_root).expanduser().resolve()
    sources = {path.resolve() for path in root.glob('**/*.csv')}
    source_identities = {unicodedata.normalize('NFC', str(path)) for path in sources}
    directories = {root}
    for path in sources:
        directories.update(parent for parent in path.parents if parent.name == 'data')
    excluded = []
    for directory in sorted(directories):
        for item in load_training_exclusions(directory).values():
            source = (directory / item['source_file']).resolve()
            if unicodedata.normalize('NFC', str(source)) in source_identities:
                excluded.append(source)
    if excluded:
        names = ', '.join(path.name for path in sorted(set(excluded)))
        raise ValueError(
            f'학습 제외 목록에 등록된 원본 {len(set(excluded))}개가 입력에 포함되어 있습니다: {names}. '
            '모델 추가 화면에서 데이터를 선택하거나 DatasetService.stage_for_training()으로 '
            '제외 기록을 뺀 별도 학습 폴더를 준비한 뒤 --data로 지정하세요. 원본은 변경하지 마세요.'
        )
