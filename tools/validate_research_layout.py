"""Verify alias routing and all generated catalog targets without touching data."""
from pathlib import Path
import json
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    'example': 'research/example',
    'artifacts': 'research/models',
    'outputs': 'research/models/legacy_outputs',
    'results': 'research/results',
    'figures': 'research/figures',
    'docs/experiments': 'docs/reports',
    'research/reports': 'docs/reports',
    'research/README.md': 'docs/research/README.md',
    'archive.zip': 'research/archives/archive.zip',
    'data.zip': 'research/archives/data.zip',
}


def main():
    for old, new in ALIASES.items():
        alias, target = ROOT / old, ROOT / new
        assert alias.is_symlink(), f'Legacy path is not a symlink: {old}'
        assert target.exists() and alias.resolve() == target.resolve(), old
        if target.is_dir():
            with tempfile.TemporaryDirectory(prefix='.layout-check-', dir=alias) as temporary:
                path = Path(temporary) / 'probe.txt'
                path.write_text('routing check', encoding='utf-8')
                assert path.resolve().is_relative_to(target.resolve())
                assert (target / Path(temporary).name / 'probe.txt').read_text() == 'routing check'
    document_aliases = [p for p in (ROOT / 'research').rglob('*')
                        if p.is_symlink() and p.suffix in {'.md', '.html', '.pdf', '.docx', '.hwp'}]
    for path in document_aliases:
        assert path.is_file() and path.resolve().is_relative_to(ROOT / 'docs'), str(path)
    inventory = json.loads((ROOT / 'docs/catalog/inventory.json').read_text())
    paths = [item['path'] for item in inventory['files']]
    assert len(paths) == len(set(paths)), 'Duplicate catalog paths'
    for rel in paths:
        path = ROOT / rel
        assert path.is_file(), f'Missing catalog target: {rel}'
        assert not any(rel == old or rel.startswith(old + '/') for old in ALIASES), rel
    print(json.dumps({'status': 'passed', 'aliases': len(ALIASES), 'catalog_targets': len(paths),
                      'new_file_routing': 'passed', 'document_aliases': len(document_aliases), 'data_writes': False}))


if __name__ == '__main__':
    main()
