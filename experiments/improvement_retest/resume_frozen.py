"""Resume the preregistered study using its original immutable model sources.

Other tasks may continue editing the shared working tree. Never rewrite their
changes or the original protocol to make a previously running study pass.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_execution(out):
    source = out / "execution_source"
    if (out / "execution_source_manifest.json").exists():
        frozen = json.loads((out / "execution_source_manifest.json").read_text())
        for name, sha in frozen["hashes"].items():
            if digest(source / name) != sha:
                raise ValueError(f"Execution source changed: {name}")
        return source
    p = json.loads((out / "protocol.json").read_text())
    source.mkdir(exist_ok=False)
    for directory in ("src", "experiments/improvement_retest"):
        for original in (ROOT / directory).rglob("*.py"):
            destination = source / original.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
    (source / "experiments/__init__.py").touch()
    # Override every frozen training/data dependency with the audited originals.
    for name, sha in p["source_hashes"].items():
        original = out / "source_snapshot" / name
        if digest(original) != sha:
            raise ValueError(f"Original study snapshot changed: {name}")
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    hashes = {str(path.relative_to(source)): digest(path) for path in source.rglob("*.py")}
    manifest = {"created_at": datetime.now().astimezone().isoformat(),
        "reason": "Concurrent working-tree edits after protocol freeze; original candidate algorithms and criteria preserved",
        "protocol_sha256": digest(out / "protocol.json"), "hashes": hashes,
        "live_changes": [name for name, sha in p["source_hashes"].items() if digest(ROOT / name) != sha],
        "scope": "Frozen sources override live model dependencies; supporting modules copied and hashed; original raw data and active models not written"}
    (out / "execution_source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["development", "select", "final", "summarize"])
    parser.add_argument("--audit", choices=["development", "selection", "final"])
    args = parser.parse_args()
    out = args.output.resolve()
    source = prepare_execution(out)
    if args.audit:
        command = [sys.executable, "-m", "experiments.improvement_retest.audit", "--output", str(out), "--stage", args.audit]
    elif args.phase:
        command = [sys.executable, "-m", "experiments.improvement_retest.run", "--output", str(out), "--phase", args.phase]
    else:
        return
    raise SystemExit(subprocess.call(command, cwd=source))


if __name__ == "__main__":
    main()
