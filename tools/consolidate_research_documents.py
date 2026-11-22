"""Move research prose into docs; preserve legacy readers via relative symlinks."""
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.md', '.html', '.pdf', '.docx', '.hwp'}


def main():
    research, docs = ROOT / 'research', ROOT / 'docs'
    reports = research / 'reports'
    reports_moved = not reports.is_symlink()
    if not reports.is_symlink():
        destination = docs / 'reports'
        if destination.exists():
            raise ValueError('docs/reports already exists; refusing to overwrite')
        reports.rename(destination)
        reports.symlink_to('../docs/reports', target_is_directory=True)
    legacy = docs / 'experiments'
    if legacy.is_symlink():
        legacy.unlink()
        legacy.symlink_to('reports', target_is_directory=True)
    paths = [p for p in research.rglob('*') if p.is_file() and not p.is_symlink()
             and p.suffix.lower() in EXTENSIONS and 'source_snapshot' not in p.parts]
    mapping = {p: docs / 'research' / p.relative_to(research) for p in paths}
    for source, target in mapping.items():
        if target.exists():
            raise ValueError(f'Destination already exists: {target}')
    for source, target in mapping.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        source.symlink_to(os.path.relpath(target, source.parent))
    # HTML dashboards use dynamic relative plot paths embedded in JSON. Preserve
    # sibling asset lookup using links, without altering the embedded experiment data.
    for source, target in mapping.items():
        if target.suffix == '.html':
            for sibling in source.parent.iterdir():
                dest = target.parent / sibling.name
                if not dest.exists() and not dest.is_symlink():
                    dest.symlink_to(os.path.relpath(sibling, dest.parent), target_is_directory=sibling.is_dir())
    # Rebase Markdown's relative links. Reports moved reports/ -> docs/reports/
    # retain the same depth, but ../results and ../models refer back to research.
    pairs = list(mapping.items())
    if reports_moved:
        pairs += [(research/'reports'/p.name, p) for p in (docs/'reports').glob('*.md')]
    for source, target in pairs:
        if target.suffix != '.md':
            continue
        text = target.read_text(encoding='utf-8')
        def rebase(match):
            raw = match.group(1)
            url = urlsplit(raw)
            if url.scheme or raw.startswith(('/', '#')):
                return match.group(0)
            # Only rebase existing old targets; leave already canonical links alone.
            old = Path(os.path.normpath(source.parent / unquote(url.path)))
            if not old.exists():
                return match.group(0)
            dest = mapping.get(old, old.resolve())
            rel = quote(os.path.relpath(dest, target.parent), safe='/')
            return '](' + rel + ('?' + url.query if url.query else '') + ('#' + url.fragment if url.fragment else '') + ')'
        target.write_text(re.sub(r'\]\(([^)\s]+)\)', rebase, text), encoding='utf-8')
    print(json.dumps({'moved_documents': len(mapping), 'reports_directory': 'docs/reports',
                      'source_snapshot_preserved': True}))


if __name__ == '__main__':
    main()
