"""Finish interrupted fits using the exact immutable source snapshot.

This coordinator does not overwrite worktree changes or change the cohort. Every
worker runs with the snapshot as cwd, so its existing fail-closed source checks
verify the locked source bytes, and it reads the original frozen feature cache.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    snapshot = out / "source_snapshot"
    p = json.loads((out / "protocol.json").read_text())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    def execute(code):
        subprocess.run([sys.executable, "-c", code, str(out)], cwd=snapshot, env=env, check=True)
    execute("from experiments.consensus_filter.run import *; import sys; verify(Path(sys.argv[1])); print('Frozen source verified')")
    def fit(job):
        scope, condition = job
        log = out / "logs" / f"frozen_{scope}_{condition}.log"
        with log.open("w") as f:
            subprocess.run([sys.executable, "-m", "experiments.consensus_filter.run", "--output", str(out),
                            "--worker", condition, "--scope", scope, "--stage", "evaluate"],
                           cwd=snapshot, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(log.read_text().strip(), flush=True)
    jobs = []
    for split in p["outer"]:
        for c in p["conditions"]:
            paths = [out / split["id"] / "evaluate" / c / arm / "metrics.json" for arm in p["arms"]]
            if not all(path.exists() for path in paths):
                jobs.append((split["id"], c))
    print('Remaining outer conditions', jobs, flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fit, jobs))
    execute("from experiments.consensus_filter.run import *; import sys; exclusions(Path(sys.argv[1]), 'full')")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fit, [("full", c) for c in p["conditions"]]))
    execute("from experiments.consensus_filter.run import *; import sys; out=Path(sys.argv[1]); verify(out); write_json(out/'execution_complete.json', {'finished_at':now(), 'frozen_source_recovery':True, 'fits_expected':192, 'protection_note':'See concurrent_workspace_changes.json; canonical data and preexisting model artifacts audited separately'})")
    print('FROZEN RECOVERY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
